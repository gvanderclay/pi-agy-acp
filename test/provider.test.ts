import assert from "node:assert/strict";
import { test } from "node:test";
import {
	createInitialSystemMessage,
	type AssistantMessage,
	type Message,
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
	type PersistedAcpBinding,
} from "../src/provider.ts";

function makeContext(systemPrompt: string, tools: Tool[], messages: Message[]): TranscriptContext {
	const sys = createInitialSystemMessage(systemPrompt, tools);
	return { messages: sys ? [sys, ...messages] : [...messages] } as TranscriptContext;
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
		return this.promptImpl(request);
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
		{ inputTokens: 6000, outputTokens: 200, cachedReadTokens: 24000, cachedWriteTokens: 0, thoughtTokens: 50, totalTokens: 30200 },
		{ used: 10200, size: 1_048_576 },
		3,
	);
	// Target prompt context = 10200 - 200 = 10000; 20% input (2000) + 80% cacheRead (8000)
	assert.equal(normalized.input, 2000);
	assert.equal(normalized.cacheRead, 8000);
	assert.equal(normalized.output, 200);
	assert.equal(normalized.reasoning, 50);
	assert.equal(normalized.totalTokens, 10200);
});

test("provider executes full parked Pi tool loop with steering, usage, and session/resume across restart", async () => {
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

		// Phase 1: Pi receives stopReason: "toolUse" + toolCall
		const stream1 = handle.provider.streamSimple(
			model,
			makeContext("Be helpful.", [codemodeTool], [user1]),
			{ sessionId: "pi-1", reasoning: "low" },
		);
		const msg1: AssistantMessage = await stream1.result();
		assert.equal(msg1.stopReason, "toolUse");
		const toolCall = msg1.content.find((b) => b.type === "toolCall");
		assert.ok(toolCall && toolCall.type === "toolCall");
		assert.equal(toolCall.name, "codemode");
		assert.deepEqual(toolCall.arguments, { code: "return 21 * 2" });

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
			makeContext("Be helpful.", [codemodeTool], [user1, msg1, toolRes, steer]),
			{ sessionId: "pi-1", reasoning: "low" },
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

		// Auxiliary call (e.g. handoff-lite with fresh sessionId) uses auxTransport, NOT mainTransport!
		const auxStream = handle.provider.streamSimple(
			model,
			makeContext("Summarize", [], [{ role: "user", content: "summarize this", timestamp: 4 }]),
			{ sessionId: "other-session-uuid", disableContinuation: true },
		);
		const auxMsg = await auxStream.result();
		assert.equal(auxMsg.stopReason, "stop");
		assert.equal(auxTransport.createdSessions.length, 1);
		assert.equal(auxTransport.closedSessions.length, 1);
		assert.equal(mainTransport.createdSessions.length, 1);

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
			handle2.syncSession({ piSessionId: "pi-1", cwd: "/work", persistedBinding: bindings[0] });
			const user2: Message = { role: "user", content: "follow-up after restart", timestamp: 5 };
			const stream3 = handle2.provider.streamSimple(
				model,
				makeContext("Be helpful.", [codemodeTool], [user1, msg1, toolRes, steer, msg2, user2]),
				{ sessionId: "pi-1", reasoning: "low" },
			);
			const msg3 = await stream3.result();
			assert.equal(msg3.stopReason, "stop");
			assert.equal(mainTransport2.resumedSessions.length, 1);
			assert.equal(mainTransport2.resumedSessions[0].sessionId, bindings[0].acpSessionId);
			// Delta prompt only contains the new user message (no <conversation_history> wrapper)
			assert.deepEqual(mainTransport2.prompts[0].prompt, [{ type: "text", text: "follow-up after restart" }]);
		} finally {
			await handle2.dispose();
		}
	} finally {
		await handle.dispose();
	}
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
			{ sessionId: "pi-2" },
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
