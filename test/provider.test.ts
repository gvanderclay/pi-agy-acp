import assert from "node:assert/strict";
import { test } from "node:test";
import {
	createInitialSystemMessage,
	type AssistantMessage,
	type Message,
	type StreamOptions,
	type SystemMessage,
	type Tool,
	type TranscriptContext,
} from "@earendil-works/pi-ai";
import type {
	AcpPromptCall,
	AcpSessionResumeOptions,
	AcpSessionStartOptions,
	AcpTransport,
} from "../src/acp-client.ts";
import type { AcpAgentInfo, AcpPromptResult, AcpSessionConfig } from "../src/acp-wire.ts";
import { McpBridgeServer, MCP_TOKEN_HEADER } from "../src/mcp-server.ts";
import {
	AGY_ACP_API,
	AGY_ACP_PROVIDER_ID,
	createAgyAcpProvider,
	modelsFromSessionConfig,
	normalizeTurnUsage,
	type AgyAcpProviderHandle,
	type PersistedAcpBinding,
} from "../src/provider.ts";

function makeContext(systemPrompt: string | SystemMessage, tools: Tool[], messages: Message[]): TranscriptContext {
	const sys = typeof systemPrompt === "string" ? createInitialSystemMessage(systemPrompt, tools) : systemPrompt;
	return { messages: sys ? [sys, ...messages] : [...messages] } as TranscriptContext;
}

const providerStreamObserver: NonNullable<StreamOptions["onProviderStreamEvent"]> = () => {};

function liveRequest(handle: AgyAcpProviderHandle, signal = new AbortController().signal) {
	handle.expectLiveRequest(signal);
	return { signal, onProviderStreamEvent: providerStreamObserver };
}

function makeMockConfig(): AcpSessionConfig {
	return {
		model: {
			configId: "model",
			currentValue: "gemini-4-neon",
			choices: [
				{ value: "gemini-4-neon", name: "Gemini 4 Neon", description: undefined },
				{ value: "gemini-3.8-flash", name: "Gemini 3.8 Flash", description: undefined },
				{ value: "claude-opus-5-5", name: "Claude Opus 5.5", description: undefined },
				{ value: "gpt-oss-120b-medium", name: "GPT-OSS 120B", description: undefined },
			],
		},
		thoughtLevel: {
			configId: "thought_level",
			currentValue: "high",
			choices: [
				{ value: "low", name: "Low", description: undefined },
				{ value: "medium", name: "Medium", description: undefined },
				{ value: "high", name: "High", description: undefined },
				{ value: "xhigh", name: "Extra", description: undefined },
			],
		},
		availableModelVariants: [
			{ value: "gemini-4-neon-low", name: "Neon Low", description: undefined },
			{ value: "gemini-4-neon-medium", name: "Neon Medium", description: undefined },
			{ value: "gemini-4-neon-high", name: "Neon High", description: undefined },
			{ value: "gemini-4-neon-xhigh", name: "Neon Extra", description: undefined },
			{ value: "gemini-3.8-flash-low", name: "Flash Low", description: undefined },
			{ value: "gemini-3.8-flash-medium", name: "Flash Medium", description: undefined },
			{ value: "gemini-3.8-flash-high", name: "Flash High", description: undefined },
			{ value: "claude-opus-5-5-low", name: "Opus Low", description: undefined },
			{ value: "claude-opus-5-5-high", name: "Opus High", description: undefined },
			{ value: "gpt-oss-120b-medium", name: "GPT-OSS", description: undefined },
		],
	};
}

class MockAcpTransport implements AcpTransport {
	readonly createdSessions: AcpSessionStartOptions[] = [];
	readonly resumedSessions: AcpSessionResumeOptions[] = [];
	readonly closedSessions: string[] = [];
	readonly configCalls: Array<{ sessionId: string; configId: string; value: string }> = [];
	readonly cancelledSessions: string[] = [];
	readonly prompts: AcpPromptCall[] = [];
	resumeError?: Error;
	disposed = false;
	private seq = 0;
	private configs = new Map<string, AcpSessionConfig>();
	promptImpl: (req: AcpPromptCall) => Promise<AcpPromptResult> = async () => ({
		stopReason: "end_turn",
		usage: { inputTokens: 100, outputTokens: 20, cachedReadTokens: 0, cachedWriteTokens: 0, thoughtTokens: 0, totalTokens: 120 },
	});

	async ensureReady(): Promise<AcpAgentInfo> {
		return {
			protocolVersion: 1,
			name: "mock-agy-acp",
			version: "1.5.17",
			authMethods: [],
			supportsResume: true,
			supportsClose: true,
			supportsSystemPromptAppend: true,
		};
	}

	async createSession(options: AcpSessionStartOptions) {
		this.createdSessions.push(options);
		const sessionId = `acp-sess-${++this.seq}`;
		const config = makeMockConfig();
		this.configs.set(sessionId, config);
		return { sessionId, config };
	}

	async resumeSession(options: AcpSessionResumeOptions) {
		this.resumedSessions.push(options);
		if (this.resumeError) throw this.resumeError;
		const config = makeMockConfig();
		this.configs.set(options.sessionId, config);
		return { sessionId: options.sessionId, config };
	}

	async closeSession(sessionId: string) {
		this.closedSessions.push(sessionId);
		this.configs.delete(sessionId);
	}

	sessionConfig(sessionId: string) {
		return this.configs.get(sessionId);
	}

	async setConfigOption(sessionId: string, configId: string, value: string) {
		this.configCalls.push({ sessionId, configId, value });
		const cfg = this.configs.get(sessionId) ?? makeMockConfig();
		if (configId === "model" && cfg.model) cfg.model = { ...cfg.model, currentValue: value };
		if (configId === "thought_level" && cfg.thoughtLevel) cfg.thoughtLevel = { ...cfg.thoughtLevel, currentValue: value };
		this.configs.set(sessionId, cfg);
		return cfg;
	}

	async prompt(request: AcpPromptCall): Promise<AcpPromptResult> {
		this.prompts.push(request);
		const onAbort = () => this.cancel(request.sessionId);
		request.signal?.addEventListener("abort", onAbort, { once: true });
		try {
			return await this.promptImpl(request);
		} finally {
			request.signal?.removeEventListener("abort", onAbort);
		}
	}

	cancel(sessionId: string) {
		this.cancelledSessions.push(sessionId);
	}

	dispose() {
		this.disposed = true;
	}
}

test("modelsFromSessionConfig maps per-model thinking levels and context windows accurately", () => {
	const models = modelsFromSessionConfig(makeMockConfig());
	const byId = new Map(models.map((m) => [m.id, m]));

	assert.equal(byId.get("gemini-4-neon")?.reasoning, true);
	assert.equal(byId.get("gemini-4-neon")?.thinkingLevelMap?.xhigh, "xhigh");
	assert.equal(byId.get("gemini-4-neon")?.contextWindow, 1_048_576);

	assert.equal(byId.get("claude-opus-5-5")?.reasoning, true);
	assert.equal(byId.get("claude-opus-5-5")?.thinkingLevelMap?.xhigh, null);
	assert.equal(byId.get("claude-opus-5-5")?.contextWindow, 200_000);

	assert.equal(byId.get("gpt-oss-120b-medium")?.reasoning, false);
	assert.equal(byId.get("gpt-oss-120b-medium")?.thinkingLevelMap, undefined);
	assert.equal(byId.get("gpt-oss-120b-medium")?.contextWindow, 128_000);
});

test("normalizeTurnUsage scales multi-step cumulative usage to final context occupancy while preserving cache ratio", () => {
	const model = modelsFromSessionConfig(makeMockConfig())[1];
	const normalized = normalizeTurnUsage(
		model,
		{ inputTokens: 6000, outputTokens: 200, cachedReadTokens: 21000, cachedWriteTokens: 3000, thoughtTokens: 50, totalTokens: 30200 },
		{ used: 10200, size: 1_048_576 },
		3,
	);
	// Target prompt context = 10200 - 200 = 10000; 20% input (2000) + 70% cacheRead (7000) + 10% cacheWrite (1000)
	assert.equal(normalized.input, 2000);
	assert.equal(normalized.cacheRead, 7000);
	assert.equal(normalized.cacheWrite, 1000);
	assert.equal(normalized.output, 200);
	assert.equal(normalized.reasoning, 50);
	assert.equal(normalized.totalTokens, 10200);
});

test("same-session extension call leaves the live ACP turn parked and subsequent turns use only the delta", async () => {
	const mainTransport = new MockAcpTransport();
	const auxTransport = new MockAcpTransport();
	let mcpEndpoint: { url: string; token: string } | undefined;
	const bindings: PersistedAcpBinding[] = [];

	const handle = createAgyAcpProvider({
		cacheDir: "/tmp/pi-agy-acp-unit-test",
		parkBatchMs: 0,
		checkAvailable: () => true,
		createMcpServer: () => new McpBridgeServer(),
		createMainTransport: (ep) => {
			mcpEndpoint = ep;
			return mainTransport;
		},
		createAuxTransport: () => auxTransport,
		onBindingChange: (b) => bindings.push(b),
	});

	const codemodeTool: Tool = {
		name: "codemode",
		description: "Run JS in Pi",
		parameters: { type: "object", properties: { code: { type: "string" } }, required: ["code"] } as never,
	};

	let mcpToolResponse: unknown;
	mainTransport.promptImpl = async (req) => {
		req.onUpdate({ type: "agent_thought_chunk", text: "planning..." });
		req.onUpdate({ type: "agent_message_chunk", text: "Running tool. " });
		req.onUpdate({ type: "usage_update", used: 500, size: 1_048_576 });

		// Simulate `agy acp` invoking our localhost MCP server mid-turn
		const res = await fetch(mcpEndpoint!.url, {
			method: "POST",
			headers: { "content-type": "application/json", [MCP_TOKEN_HEADER]: mcpEndpoint!.token },
			body: JSON.stringify({
				jsonrpc: "2.0",
				id: 99,
				method: "tools/call",
				params: { name: "codemode", arguments: { code: "return 21 * 2" } },
			}),
		});
		mcpToolResponse = await res.json();

		req.onUpdate({ type: "agent_message_chunk", text: "Done: 42" });
		req.onUpdate({ type: "usage_update", used: 900, size: 1_048_576 });
		return {
			stopReason: "end_turn",
			usage: { inputTokens: 200, outputTokens: 40, cachedReadTokens: 660, cachedWriteTokens: 0, thoughtTokens: 15, totalTokens: 900 },
		};
	};

	try {
		handle.syncSession({ piSessionId: "pi-1", cwd: "/work" });
		const model = handle.provider.getModels().find((m) => m.id === "gemini-3.8-flash")!;
		const user1: Message = { role: "user", content: "compute 21*2", timestamp: 1 };
		const signal = new AbortController().signal;
		const liveOptions = { ...liveRequest(handle, signal), sessionId: "pi-1", reasoning: "low" as const };
		const auxContext = makeContext("Summarize", [], [{ role: "user", content: "summarize this", timestamp: 4 }]);

		const beforeLive = handle.provider.streamSimple(model, auxContext, { signal, sessionId: "pi-1" });
		assert.equal((await beforeLive.result()).stopReason, "stop");
		assert.equal(mainTransport.createdSessions.length, 0);

		// Phase 1: Pi receives stopReason: "toolUse" + toolCall
		const stream1 = handle.provider.streamSimple(
			model,
			makeContext("Be helpful.", [codemodeTool], [user1]),
			liveOptions,
		);
		const msg1: AssistantMessage = await stream1.result();
		assert.equal(msg1.stopReason, "toolUse");
		const toolCall = msg1.content.find((b) => b.type === "toolCall");
		assert.ok(toolCall && toolCall.type === "toolCall");
		assert.equal(toolCall.name, "codemode");
		assert.deepEqual(toolCall.arguments, { code: "return 21 * 2" });

		const auxStream = handle.provider.streamSimple(model, auxContext, { ...liveOptions });
		assert.equal((await auxStream.result()).stopReason, "stop");
		assert.deepEqual(mainTransport.cancelledSessions, []);
		assert.equal(mainTransport.createdSessions.length, 1);
		assert.equal(auxTransport.createdSessions.length, 2);
		assert.equal(auxTransport.closedSessions.length, 2);
		assert.equal(bindings.length, 0);

		// Phase 2: Pi executes tool, appends toolResult + user steering, calls streamSimple again
		const toolRes: Message = {
			role: "toolResult",
			toolCallId: toolCall.id,
			toolName: "codemode",
			content: [{ type: "text", text: "42" }],
			isError: false,
			timestamp: 2,
		};
		const steer: Message = { role: "user", content: "keep it brief", timestamp: 3 };

		const stream2 = handle.provider.streamSimple(
			model,
			makeContext("Be helpful. Tools changed.", [codemodeTool], [user1, msg1, toolRes, steer]),
			{ ...liveRequest(handle, signal), sessionId: "pi-1", reasoning: "low" },
		);
		const msg2: AssistantMessage = await stream2.result();
		assert.equal(msg2.stopReason, "stop");
		const textBlock = msg2.content.find((b) => b.type === "text");
		assert.ok(textBlock && textBlock.type === "text");
		assert.equal(textBlock.text, "Done: 42");
		assert.equal(msg2.usage.cacheRead, 660);
		assert.equal(bindings.length, 1);

		// Verify steering was delivered inside the MCP tool result
		const mcpContent = (mcpToolResponse as { result: { content: Array<{ text: string }> } }).result.content;
		assert.equal(mcpContent[0].text, "42");
		assert.match(mcpContent[1].text, /keep it brief/);

		assert.equal(mainTransport.prompts.length, 1);
		assert.deepEqual(mainTransport.resumedSessions, []);
		mainTransport.promptImpl = async (req) => {
			req.onUpdate({ type: "agent_message_chunk", text: "Still warm" });
			return { stopReason: "end_turn", usage: undefined };
		};
		const user2: Message = { role: "user", content: "follow-up", timestamp: 5 };
		const msg3 = await handle.provider.streamSimple(
			model,
			makeContext("Be helpful. Tools changed.", [codemodeTool], [user1, msg1, toolRes, steer, msg2, user2]),
			{ ...liveRequest(handle), sessionId: "pi-1", reasoning: "low" },
		).result();
		assert.equal(msg3.stopReason, "stop");
		assert.equal(mainTransport.createdSessions.length, 1);
		assert.equal(mainTransport.prompts[1].sessionId, mainTransport.prompts[0].sessionId);
		assert.deepEqual(
			mainTransport.resumedSessions.map(({ sessionId, systemPromptAppend }) => ({ sessionId, systemPromptAppend })),
			[{ sessionId: mainTransport.prompts[0].sessionId, systemPromptAppend: "Be helpful. Tools changed." }],
		);
		assert.deepEqual(mainTransport.prompts[1].prompt, [{ type: "text", text: "follow-up" }]);

		// Simulate Pi restart: a fresh provider instance receives the persisted binding -> resumes via session/resume!
		const mainTransport2 = new MockAcpTransport();
		mainTransport2.promptImpl = async (req) => {
			req.onUpdate({ type: "agent_message_chunk", text: "Resumed ok" });
			return {
				stopReason: "end_turn",
				usage: { inputTokens: 100, outputTokens: 10, cachedReadTokens: 800, cachedWriteTokens: 0, thoughtTokens: 0, totalTokens: 910 },
			};
		};
		const handle2 = createAgyAcpProvider({
			cacheDir: "/tmp/pi-agy-acp-unit-test",
			parkBatchMs: 0,
			checkAvailable: () => true,
			createMainTransport: () => mainTransport2,
		});
		try {
			const binding = bindings.at(-1)!;
			handle2.syncSession({ piSessionId: "pi-1", cwd: "/work", persistedBinding: binding });
			const user3: Message = { role: "user", content: "follow-up after restart", timestamp: 6 };
			const restarted = await handle2.provider.streamSimple(
				model,
				makeContext("Be helpful. Tools changed.", [codemodeTool], [user1, msg1, toolRes, steer, msg2, user2, msg3, user3]),
				{ ...liveRequest(handle2), sessionId: "pi-1", reasoning: "low" },
			).result();
			assert.equal(restarted.stopReason, "stop");
			assert.equal(mainTransport2.resumedSessions.length, 1);
			assert.equal(mainTransport2.resumedSessions[0].sessionId, binding.acpSessionId);
			// Delta prompt only contains the new user message (no <conversation_history> wrapper)
			assert.deepEqual(mainTransport2.prompts[0].prompt, [{ type: "text", text: "follow-up after restart" }]);
		} finally {
			await handle2.dispose();
		}
	} finally {
		await handle.dispose();
	}
});

const user = (content: string, timestamp: number): Message => ({ role: "user", content, timestamp });

function mockConversation() {
	const transport = new MockAcpTransport();
	const logs: string[] = [];
	const handle = createAgyAcpProvider({
		cacheDir: "/tmp/pi-agy-acp-unit-test",
		createMainTransport: () => transport,
		log: (message) => logs.push(message),
	});
	handle.syncSession({ piSessionId: "pi-live", cwd: "/work" });
	const model = handle.provider.getModels()[0];
	const turn = (system: string | SystemMessage, messages: Message[]) => handle.provider.streamSimple(
		model,
		makeContext(system, [], messages),
		{ ...liveRequest(handle), sessionId: "pi-live", reasoning: "low" },
	).result();
	return { transport, logs, handle, turn };
}

const stockSystem = (tools: string, rules: string): SystemMessage => ({
	role: "system",
	content: "",
	sections: {
		preamble: "You are a coding assistant running inside pi, a coding agent harness.",
		tools,
		rules,
		project_context: "<project_context>AGENTS.md</project_context>",
	},
	timestamp: 0,
});

test("a Pi tools-only system patch keeps the ACP session warm and sends a delta without resume", async (t) => {
	const { transport, handle, turn } = mockConversation();
	t.after(() => handle.dispose());
	const system = stockSystem("<tools>codemode</tools>", "<rules>Be concise.</rules>");
	const first = user("first question", 1);
	const reply = await turn(system, [first]);
	const toolPatch: SystemMessage = {
		role: "system",
		content: "",
		sections: { tools: "<tools>codemode, fetch_content</tools>" },
		timestamp: 2,
	};
	const next = await turn(system, [first, reply, toolPatch, user("next question", 3)]);
	assert.equal(next.stopReason, "stop");
	assert.match(transport.createdSessions[0].systemPromptAppend ?? "", /<tools>codemode<\/tools>/);
	assert.equal(transport.createdSessions.length, 1);
	assert.equal(transport.prompts[1].sessionId, transport.prompts[0].sessionId);
	assert.deepEqual(transport.prompts[1].prompt, [{ type: "text", text: "next question" }]);
	assert.deepEqual(transport.resumedSessions, []);
});

test("a rules change resumes with the full append, then tools-only changes keep that session warm", async (t) => {
	const { transport, handle, turn } = mockConversation();
	t.after(() => handle.dispose());
	const first = user("first question", 1);
	const reply = await turn(stockSystem("<tools>codemode</tools>", "<rules>Be concise.</rules>"), [first]);
	const second = user("next question", 2);
	const next = await turn(stockSystem("<tools>codemode, fetch_content</tools>", "<rules>Give examples.</rules>"), [first, reply, second]);
	assert.equal(next.stopReason, "stop");
	const sid = transport.prompts[0].sessionId;
	assert.deepEqual(
		transport.resumedSessions.map(({ sessionId, systemPromptAppend }) => ({ sessionId, systemPromptAppend })),
		[{
			sessionId: sid,
			systemPromptAppend: "<tools>codemode, fetch_content</tools>\n\n<rules>Give examples.</rules>\n\n<project_context>AGENTS.md</project_context>",
		}],
	);
	const third = await turn(stockSystem("<tools>codemode</tools>", "<rules>Give examples.</rules>"), [
		first, reply, second, next, user("third question", 3),
	]);
	assert.equal(third.stopReason, "stop");
	assert.equal(transport.createdSessions.length, 1);
	assert.equal(transport.prompts[2].sessionId, sid);
	assert.deepEqual(transport.prompts[2].prompt, [{ type: "text", text: "third question" }]);
	assert.equal(transport.resumedSessions.length, 1);
});

test("changed system append resumes the same ACP session with only the next user prompt", async (t) => {
	const { transport, handle, turn } = mockConversation();
	t.after(() => handle.dispose());
	const first = user("first question", 1);
	const reply = await turn("original system", [first]);
	transport.configCalls.length = 0;
	const followUp = user("next question", 2);
	const next = await turn("updated system", [first, reply, followUp]);
	assert.equal(next.stopReason, "stop");

	assert.equal(transport.createdSessions.length, 1);
	assert.deepEqual(transport.closedSessions, []);
	const sessionId = transport.prompts[0].sessionId;
	assert.deepEqual(
		transport.resumedSessions.map(({ sessionId, systemPromptAppend }) => ({ sessionId, systemPromptAppend })),
		[{ sessionId, systemPromptAppend: "updated system" }],
	);
	assert.equal(transport.prompts[1].sessionId, sessionId);
	assert.deepEqual(transport.prompts[1].prompt, [{ type: "text", text: "next question" }]);
	assert.deepEqual(
		Object.fromEntries(transport.configCalls.filter((call) => call.sessionId === sessionId).map((call) => [call.configId, call.value])),
		{ model: "gemini-4-neon", thought_level: "low" },
	);

	const third = await turn("updated system", [first, reply, followUp, next, user("third question", 3)]);
	assert.equal(third.stopReason, "stop");
	assert.equal(transport.resumedSessions.length, 1);
	assert.deepEqual(transport.prompts[2].prompt, [{ type: "text", text: "third question" }]);
});

test("failed append refresh keeps the warm ACP session and retries the update on the next turn", async (t) => {
	const { transport, logs, handle, turn } = mockConversation();
	t.after(() => handle.dispose());
	const first = user("first question", 1);
	const reply = await turn("original system", [first]);
	const second = user("next question", 2);
	transport.resumeError = new Error("resume not available");
	const next = await turn("updated system", [first, reply, second]);
	assert.equal(next.stopReason, "stop");

	const sessionId = transport.prompts[0].sessionId;
	assert.equal(transport.createdSessions.length, 1);
	assert.deepEqual(transport.closedSessions, []);
	assert.equal(transport.prompts[1].sessionId, sessionId);
	assert.deepEqual(transport.prompts[1].prompt, [{ type: "text", text: "next question" }]);
	assert.match(logs.join("\n"), /keeping previous system append: resume not available/);

	transport.resumeError = undefined;
	const third = await turn("updated system", [first, reply, second, next, user("third question", 3)]);
	assert.equal(third.stopReason, "stop");
	assert.deepEqual(
		transport.resumedSessions.map(({ sessionId, systemPromptAppend }) => ({ sessionId, systemPromptAppend })),
		[{ sessionId, systemPromptAppend: "updated system" }, { sessionId, systemPromptAppend: "updated system" }],
	);
});

test("a converted Pi custom message between turns is sent as context delta, not replayed history", async (t) => {
	const { transport, handle, turn } = mockConversation();
	t.after(() => handle.dispose());
	const first = user("old question", 1);
	const reply = await turn("sys", [first]);
	const notice = user("background task completed", 2);
	const next = await turn("sys", [first, reply, notice, user("next question", 3)]);
	assert.equal(next.stopReason, "stop");

	assert.equal(transport.createdSessions.length, 1);
	assert.equal(transport.prompts[1].sessionId, transport.prompts[0].sessionId);
	const [context, question] = transport.prompts[1].prompt;
	assert.equal(context.type, "text");
	assert.match(context.text, /background task completed/);
	assert.doesNotMatch(context.text, /old question|conversation_history/);
	assert.deepEqual(question, { type: "text", text: "next question" });
});

test("a context-only Pi continuation stays warm without replaying history", async (t) => {
	const { transport, handle, turn } = mockConversation();
	t.after(() => handle.dispose());
	const first = user("first question", 1);
	const reply = await turn("sys", [first]);
	const next = await turn("sys", [first, reply]);
	assert.equal(next.stopReason, "stop");
	assert.equal(transport.createdSessions.length, 1);
	assert.equal(transport.prompts[1].sessionId, transport.prompts[0].sessionId);
	assert.deepEqual(transport.prompts[1].prompt, [{ type: "text", text: "Continue the conversation." }]);
});

test("an edited prefix starts a new ACP session, replays the edited history, and logs why", async (t) => {
	const { transport, logs, handle, turn } = mockConversation();
	t.after(() => handle.dispose());
	const reply = await turn("sys", [user("original question", 1)]);
	const next = await turn("sys", [user("edited question", 1), reply, user("next question", 2)]);
	assert.equal(next.stopReason, "stop");

	assert.equal(transport.createdSessions.length, 2);
	assert.notEqual(transport.prompts[1].sessionId, transport.prompts[0].sessionId);
	const [history, question] = transport.prompts[1].prompt;
	assert.equal(history.type, "text");
	assert.match(history.text, /<conversation_history>[\s\S]*edited question/);
	assert.doesNotMatch(history.text, /original question/);
	assert.deepEqual(question, { type: "text", text: "next question" });
	assert.match(logs.join("\n"), /full transcript replay: diverged prefix/);
	assert.match(logs.join("\n"), /raw ACP usage[^\n]*"inputTokens":100/);
});

test("refreshModels restores stored catalog offline and persists discovered models online", async () => {
	const auxTransport = new MockAcpTransport();
	const handle = createAgyAcpProvider({
		cacheDir: "/tmp/pi-agy-acp-unit-test-2",
		checkAvailable: () => true,
		createAuxTransport: () => auxTransport,
	});

	try {
		const storedModel = {
			id: "gemini-4-neon",
			name: "Stored Neon",
			api: AGY_ACP_API,
			provider: AGY_ACP_PROVIDER_ID,
			baseUrl: "acp://agy-acp",
			reasoning: true,
			input: ["text", "image"] as ("text" | "image")[],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 1_048_576,
			maxTokens: 65_536,
		};

		// Offline startup restore
		await handle.provider.refreshModels!({
			stored: { models: [storedModel], checkedAt: 1000 },
			allowNetwork: false,
			signal: new AbortController().signal,
			publish: async (pub) => {
				pub.update?.();
				return true;
			},
		});
		assert.equal(handle.provider.getModels().length, 1);
		assert.equal(handle.provider.getModels()[0].name, "Stored Neon");

		// Online discovery persists full catalog
		let persistedCount = 0;
		await handle.provider.refreshModels!({
			stored: { models: [storedModel], checkedAt: 1000 },
			allowNetwork: true,
			signal: new AbortController().signal,
			publish: async (pub) => {
				if (pub.persist) persistedCount = pub.persist.models.length;
				pub.update?.();
				return true;
			},
		});
		assert.equal(persistedCount, 4);
		assert.equal(handle.provider.getModels().length, 4);
		assert.equal(auxTransport.closedSessions.length, 1);
		assert.equal(auxTransport.disposed, true);
	} finally {
		await handle.dispose();
	}
});

test("abandonParkedTurn cancels remote ACP session and rejects open MCP call", async () => {
	const mainTransport = new MockAcpTransport();
	let mcpEndpoint: { url: string; token: string } | undefined;
	const handle = createAgyAcpProvider({
		cacheDir: "/tmp/pi-agy-acp-unit-test-3",
		parkBatchMs: 0,
		checkAvailable: () => true,
		createMcpServer: () => new McpBridgeServer(),
		createMainTransport: (ep) => {
			mcpEndpoint = ep;
			return mainTransport;
		},
	});

	const codemodeTool: Tool = {
		name: "codemode",
		description: "Run JS",
		parameters: { type: "object", properties: {} } as never,
	};

	let mcpErrorResult: unknown;
	mainTransport.promptImpl = async () => {
		const res = await fetch(mcpEndpoint!.url, {
			method: "POST",
			headers: { "content-type": "application/json", [MCP_TOKEN_HEADER]: mcpEndpoint!.token },
			body: JSON.stringify({
				jsonrpc: "2.0",
				id: 55,
				method: "tools/call",
				params: { name: "codemode", arguments: {} },
			}),
		});
		mcpErrorResult = await res.json();
		return { stopReason: "cancelled", usage: undefined };
	};

	try {
		handle.syncSession({ piSessionId: "pi-2", cwd: "/work" });
		const model = handle.provider.getModels()[0];
		const stream1 = handle.provider.streamSimple(
			model,
			makeContext("sys", [codemodeTool], [{ role: "user", content: "go", timestamp: 1 }]),
			{ ...liveRequest(handle), sessionId: "pi-2" },
		);
		const msg1 = await stream1.result();
		assert.equal(msg1.stopReason, "toolUse");

		// Simulate Pi aborting during tool execution (agent_end fires without follow-up streamSimple)
		handle.abandonParkedTurn("User aborted");
		await new Promise((r) => setTimeout(r, 30));

		assert.equal(mainTransport.cancelledSessions.length, 1);
		assert.equal((mcpErrorResult as { result: { isError: boolean } }).result.isError, true);
	} finally {
		await handle.dispose();
	}
});

test("aborting a main turn aborts the underlying ACP prompt signal so a follow-up turn runs cleanly", async (t) => {
	const mainTransport = new MockAcpTransport();
	const handle = createAgyAcpProvider({
		cacheDir: "/tmp/pi-agy-acp-unit-test-4",
		checkAvailable: () => true,
		createMainTransport: () => mainTransport,
	});
	t.after(() => handle.dispose());
	handle.syncSession({ piSessionId: "pi-abort", cwd: "/work" });
	const model = handle.provider.getModels()[0];

	let firstSignalAborted = false;
	mainTransport.promptImpl = (req) =>
		new Promise((resolve) => {
			if (mainTransport.prompts.length === 1) {
				req.signal?.addEventListener(
					"abort",
					() => {
						firstSignalAborted = true;
						resolve({ stopReason: "cancelled", usage: undefined });
					},
					{ once: true },
				);
			} else {
				req.onUpdate({ type: "agent_message_chunk", text: "Recovered" });
				resolve({ stopReason: "end_turn", usage: undefined });
			}
		});

	const ac = new AbortController();
	const firstStream = handle.provider.streamSimple(
		model,
		makeContext("sys", [], [user("slow task", 1)]),
		{ ...liveRequest(handle, ac.signal), sessionId: "pi-abort" },
	);
	await new Promise((r) => setTimeout(r, 20));
	ac.abort();
	const abortedMsg = await firstStream.result();
	assert.equal(abortedMsg.stopReason, "aborted");
	assert.equal(firstSignalAborted, true);
	assert.equal(mainTransport.cancelledSessions.length, 1);

	const nextMsg = await handle.provider.streamSimple(
		model,
		makeContext("sys", [], [user("try again", 2)]),
		{ ...liveRequest(handle), sessionId: "pi-abort" },
	).result();
	assert.equal(nextMsg.stopReason, "stop");
	assert.equal(mainTransport.prompts.length, 2);
});

test("auxTransport is disposed after auxIdleMs of inactivity", async (t) => {
	const created: MockAcpTransport[] = [];
	const handle = createAgyAcpProvider({
		cacheDir: "/tmp/pi-agy-acp-unit-test-5",
		auxIdleMs: 25,
		checkAvailable: () => true,
		createAuxTransport: () => {
			const transport = new MockAcpTransport();
			created.push(transport);
			return transport;
		},
	});
	t.after(() => handle.dispose());
	const model = handle.provider.getModels()[0];

	const msg1 = await handle.provider.streamSimple(
		model,
		makeContext("Summarize", [], [user("text 1", 1)]),
		{},
	).result();
	assert.equal(msg1.stopReason, "stop");
	assert.equal(created.length, 1);
	assert.equal(created[0].disposed, false);

	await new Promise((r) => setTimeout(r, 60));
	assert.equal(created[0].disposed, true);

	const msg2 = await handle.provider.streamSimple(
		model,
		makeContext("Summarize", [], [user("text 2", 2)]),
		{},
	).result();
	assert.equal(msg2.stopReason, "stop");
	assert.equal(created.length, 2);
});
