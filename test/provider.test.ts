import assert from "node:assert/strict";
import { test } from "node:test";
import {
	createInitialSystemMessage,
	type AssistantMessage,
	type Message,
	type StreamOptions,
	type Tool,
	type ToolResultMessage,
	type TranscriptContext,
} from "@earendil-works/pi-ai";
import type {
	AcpPromptCall,
	AcpSessionResumeOptions,
	AcpSessionStartOptions,
	AcpTransport,
	AcpUpdateListener,
} from "../src/acp-client.ts";
import type { AcpAgentInfo, AcpPromptResult, AcpSessionConfig, AcpSessionUpdate } from "../src/acp-wire.ts";
import { GATE_URL_ENV, type HookToolCall } from "../src/hook-gate.ts";
import {
	agyRoleInstructions,
	approvalToolCall,
	createAgyNativeProvider,
	modelsFromSessionConfig,
	normalizeTurnUsage,
	type AgyNativeProviderHandle,
	type AgyNativeProviderOptions,
	type SessionView,
} from "../src/provider.ts";

// Codemode can hide Pi's read declaration; Agy reads must still go through Pi's read.
function makeContext(messages: Message[], tools: Tool[] = []): TranscriptContext {
	const sys = createInitialSystemMessage("Pi prompt", tools);
	return { messages: sys ? [sys, ...messages] : [...messages] } as TranscriptContext;
}

const user = (content: string, timestamp: number): Message => ({ role: "user", content, timestamp });

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

const USAGE = { inputTokens: 100_000, outputTokens: 20, cachedReadTokens: 0, cachedWriteTokens: 0, thoughtTokens: 0, totalTokens: 100_020 };

class MockAcpTransport implements AcpTransport {
	readonly createdSessions: AcpSessionStartOptions[] = [];
	readonly resumedSessions: AcpSessionResumeOptions[] = [];
	readonly cancelledSessions: string[] = [];
	readonly prompts: AcpPromptCall[] = [];
	gateUrl = "";
	allowPermission: (toolCallId: string | undefined) => boolean = () => false;
	private seq = 0;
	private readonly configs = new Map<string, AcpSessionConfig>();
	private readonly listeners = new Map<string, AcpUpdateListener>();
	promptImpl: (req: AcpPromptCall) => Promise<AcpPromptResult> = async () => ({ stopReason: "end_turn", usage: USAGE });

	async ensureReady(): Promise<AcpAgentInfo> {
		return { protocolVersion: 1, name: "mock", version: "1.5.18", authMethods: [], supportsResume: true, supportsClose: true, supportsSystemPromptAppend: true };
	}

	async createSession(options: AcpSessionStartOptions) {
		this.createdSessions.push(options);
		const sessionId = `acp-${++this.seq}`;
		this.configs.set(sessionId, makeMockConfig());
		return { sessionId, config: makeMockConfig() };
	}

	async resumeSession(options: AcpSessionResumeOptions) {
		this.resumedSessions.push(options);
		this.configs.set(options.sessionId, makeMockConfig());
		return { sessionId: options.sessionId, config: makeMockConfig() };
	}

	async closeSession(sessionId: string) {
		this.configs.delete(sessionId);
	}

	sessionConfig(sessionId: string) {
		return this.configs.get(sessionId);
	}

	async setConfigOption(sessionId: string, configId: string, value: string) {
		const cfg = this.configs.get(sessionId) ?? makeMockConfig();
		if (configId === "model" && cfg.model) cfg.model = { ...cfg.model, currentValue: value };
		if (configId === "thought_level" && cfg.thoughtLevel) cfg.thoughtLevel = { ...cfg.thoughtLevel, currentValue: value };
		this.configs.set(sessionId, cfg);
		return cfg;
	}

	setUpdateListener(sessionId: string, listener: AcpUpdateListener | undefined) {
		if (listener) this.listeners.set(sessionId, listener);
		else this.listeners.delete(sessionId);
	}

	emit(sessionId: string, update: AcpSessionUpdate) {
		this.listeners.get(sessionId)?.(update, { sessionId, update });
	}

	text(sessionId: string, text: string) {
		this.emit(sessionId, { type: "agent_message_chunk", text });
	}

	/** Runs the pi-gate hook the way Agy would, returning Pi's decision. */
	async hook(toolCall: { id: string; name: string; args: Record<string, unknown> }, conversationId: string) {
		const response = await fetch(this.gateUrl, { method: "POST", body: JSON.stringify({ toolCall, conversationId }) });
		return (await response.json()) as { decision: string; reason?: string };
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

	dispose() {}
}

const promptText = (call: AcpPromptCall) => call.prompt.map((block) => (block.type === "text" ? block.text : `[${block.type}]`)).join("\n");

function setup(overrides: Partial<AgyNativeProviderOptions> = {}) {
	const transport = new MockAcpTransport();
	const notices: string[] = [];
	const handle = createAgyNativeProvider({
		cacheDir: "/tmp/pi-agy-native-unit-test",
		geminiDir: "/tmp/pi-agy-native-unit-test/gemini",
		parkBatchMs: 0,
		checkAvailable: () => true,
		installGate: () => {},
		notify: (message) => notices.push(message),
		createTransport: (env, allowPermission) => {
			transport.gateUrl = env[GATE_URL_ENV];
			transport.allowPermission = allowPermission;
			return transport;
		},
		...overrides,
	});
	handle.syncSession({ piSessionId: "pi-1", cwd: "/work" });
	const model = handle.provider.getModels().find((m) => m.id === "gemini-4-neon")!;
	const send = (messages: Message[], options: Partial<StreamOptions> = {}) =>
		handle.provider.streamSimple(model, makeContext(messages), live(handle, options)).result();
	return { handle, transport, model, send, notices };
}

function live(handle: AgyNativeProviderHandle, options: Partial<StreamOptions> = {}): StreamOptions {
	const signal = options.signal ?? new AbortController().signal;
	handle.expectLiveRequest(signal);
	return { signal, sessionId: "pi-1", onProviderStreamEvent: () => {}, ...options };
}

const textOf = (message: AssistantMessage) =>
	message.content.map((part) => (part.type === "text" ? part.text : "")).join("");

const toolResult = (call: AssistantMessage, text: string, isError: boolean, timestamp: number): ToolResultMessage => {
	const toolCall = call.content.find((part) => part.type === "toolCall");
	assert.ok(toolCall && toolCall.type === "toolCall");
	return { role: "toolResult", toolCallId: toolCall.id, toolName: toolCall.name, content: [{ type: "text", text }], isError, timestamp };
};

function deferred<T>() {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((r) => (resolve = r));
	return { promise, resolve };
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

test("a typed message reaches Agy and the next typed message continues the same Agy session", async () => {
	const { transport, send, handle } = setup();
	transport.promptImpl = async (req) => {
		transport.text(req.sessionId, `reply ${transport.prompts.length}`);
		return { stopReason: "end_turn", usage: USAGE };
	};
	try {
		const first = user("hello", 1);
		const reply1 = await send([first]);
		assert.equal(reply1.stopReason, "stop");
		assert.equal(textOf(reply1), "reply 1");

		const reply2 = await send([first, reply1, user("and again", reply1.timestamp + 1)]);
		assert.equal(textOf(reply2), "reply 2");
		assert.equal(transport.createdSessions.length, 1);
		assert.deepEqual(transport.prompts.map((p) => p.sessionId), ["acp-1", "acp-1"]);
		assert.deepEqual(transport.prompts.map(promptText), ["hello", "and again"]);
	} finally {
		await handle.dispose();
	}
});

test("Pi's context meter counts Pi's transcript, not Agy's own context", async () => {
	const { send, handle } = setup();
	try {
		const reply = await send([user("hello", 1)]);
		assert.ok(reply.usage.input === USAGE.inputTokens);
		assert.ok(reply.usage.totalTokens > 0 && reply.usage.totalTokens < 1_000, `totalTokens ${reply.usage.totalTokens}`);
	} finally {
		await handle.dispose();
	}
});

test("an allowed Agy command runs and its output becomes the Pi tool result", async () => {
	const { transport, send, handle } = setup();
	const decisions: unknown[] = [];
	transport.promptImpl = async (req) => {
		transport.text(req.sessionId, "Running. ");
		decisions.push(await transport.hook({ id: "tc-1", name: "run_command", args: { CommandLine: "echo hi", Cwd: "/work" } }, req.sessionId));
		transport.emit(req.sessionId, { type: "tool_call_update", toolCallId: "tc-1", title: "echo hi", status: "completed", rawOutput: { output: "hi\n", exit_code: 0 }, content: undefined });
		transport.text(req.sessionId, "Printed hi.");
		return { stopReason: "end_turn", usage: USAGE };
	};
	try {
		const first = user("say hi in the shell", 1);
		const parked = await send([first]);
		assert.equal(parked.stopReason, "toolUse");
		const call = parked.content.find((part) => part.type === "toolCall");
		assert.ok(call && call.type === "toolCall");
		assert.equal(call.name, "agy_command");
		assert.deepEqual(call.arguments, { command: "echo hi", cwd: "/work" });

		const output = await handle.runApprovedTool(call.id, undefined);
		assert.deepEqual(output, { text: "hi\n", isError: false });
		assert.deepEqual(decisions, [{ decision: "auto_approve" }]);
		assert.equal(transport.allowPermission("tc-1"), true);
		assert.equal(transport.allowPermission("tc-other"), false);

		const done = await send([first, parked, toolResult(parked, "hi\n", false, 3)]);
		assert.equal(done.stopReason, "stop");
		assert.equal(textOf(done), "Printed hi.");
		assert.equal(transport.prompts.length, 1);
	} finally {
		await handle.dispose();
	}
});

test("a call Pi blocks is denied to Agy with Pi's reason", async () => {
	const { transport, send, handle } = setup();
	const decisions: unknown[] = [];
	transport.promptImpl = async (req) => {
		decisions.push(await transport.hook({ id: "tc-1", name: "run_command", args: { CommandLine: "printenv MY_KEY" } }, req.sessionId));
		transport.text(req.sessionId, "Blocked.");
		return { stopReason: "end_turn", usage: USAGE };
	};
	try {
		const first = user("print the key", 1);
		const parked = await send([first]);
		const done = await send([first, parked, toolResult(parked, "Blocked by rule: secrets", true, 3)]);
		assert.equal(textOf(done), "Blocked.");
		assert.deepEqual(decisions, [{ decision: "deny", reason: "Blocked by rule: secrets" }]);
	} finally {
		await handle.dispose();
	}
});

test("Agy reads become Pi read calls, and Agy reads the file once Pi allows it", async () => {
	const { transport, send, handle } = setup();
	const decisions: unknown[] = [];
	transport.promptImpl = async (req) => {
		decisions.push(await transport.hook({ id: "tc-1", name: "read_file", args: { Path: "/work/notes.txt" } }, req.sessionId));
		return { stopReason: "end_turn", usage: USAGE };
	};
	try {
		const first = user("read notes", 1);
		const parked = await send([first]);
		const call = parked.content.find((part) => part.type === "toolCall");
		assert.ok(call && call.type === "toolCall");
		assert.equal(call.name, "read");
		assert.deepEqual(call.arguments, { path: "/work/notes.txt" });
		await send([first, parked, toolResult(parked, "file text", false, 3)]);
		assert.deepEqual(decisions, [{ decision: "auto_approve" }]);
	} finally {
		await handle.dispose();
	}
});

test("Agy writes and other calls on a file carry the path Pi's path rules check", async () => {
	const { transport, send, handle } = setup();
	transport.promptImpl = async (req) => {
		await transport.hook({ id: "tc-1", name: "write_file", args: { Path: "/work/a.txt", Content: "hi" } }, req.sessionId);
		await transport.hook({ id: "tc-2", name: "list_dir", args: { AbsolutePath: "/home/me/.ssh" } }, req.sessionId);
		return { stopReason: "end_turn", usage: USAGE };
	};
	try {
		const first = user("write", 1);
		const parked = await send([first]);
		const write = parked.content.find((part) => part.type === "toolCall");
		assert.ok(write && write.type === "toolCall");
		assert.equal(write.name, "agy_edit");
		assert.deepEqual(write.arguments, { path: "/work/a.txt" });
		const parked2 = await send([first, parked, toolResult(parked, "ok", false, 3)]);
		const list = parked2.content.find((part) => part.type === "toolCall");
		assert.ok(list && list.type === "toolCall");
		assert.deepEqual(list.arguments, { tool: "list_dir", arguments: { AbsolutePath: "/home/me/.ssh" }, path: "/home/me/.ssh" });
	} finally {
		await handle.dispose();
	}
});

test("subagent and unknown tool calls are shown as such and return without output", async () => {
	const { transport, send, handle } = setup();
	transport.promptImpl = async () => {
		await transport.hook({ id: "tc-1", name: "search_web", args: { query: "pi" } }, "subagent-conversation");
		return { stopReason: "end_turn", usage: USAGE };
	};
	try {
		const parked = await send([user("research", 1)]);
		const call = parked.content.find((part) => part.type === "toolCall");
		assert.ok(call && call.type === "toolCall");
		assert.equal(call.name, "agy_search_web");
		assert.deepEqual(call.arguments, { query: "pi", subagent: true });
		const output = await handle.runApprovedTool(call.id, undefined);
		assert.equal(output.isError, false);
		assert.match(output.text, /subagent/);
	} finally {
		await handle.dispose();
	}
});

test("each known Agy tool gets its own approval tool, so Pi rules can tell them apart", () => {
	const call = (name: string, args: Record<string, unknown>) => approvalToolCall({ toolCallId: "tc", toolName: name, args } as HookToolCall, false, "/work");
	assert.deepEqual(call("search_web", { query: "pi" }), { name: "agy_search_web", arguments: { query: "pi" } });
	assert.deepEqual(call("invoke_subagent", { Prompt: "go", Role: "r" }), { name: "agy_invoke_subagent", arguments: { arguments: { Prompt: "go", Role: "r" } } });
	assert.deepEqual(call("send_message", { Message: "hi" }), { name: "agy_send_message", arguments: { arguments: { Message: "hi" } } });
	assert.deepEqual(call("new_tool", { X: 1 }), { name: "agy_tool", arguments: { tool: "new_tool", arguments: { X: 1 } } });
});

test("a hook call with no Pi turn running is denied", async () => {
	const { transport, send, handle } = setup();
	try {
		await send([user("hello", 1)]);
		const decision = await transport.hook({ id: "late", name: "run_command", args: { CommandLine: "ls" } }, "acp-1");
		assert.equal(decision.decision, "deny");
	} finally {
		await handle.dispose();
	}
});

test("a side request runs on its own Agy session with its instructions, without tools, and leaves the conversation's session alone", async () => {
	const { transport, send, handle, model } = setup();
	let sideHook: { decision: string; reason?: string } | undefined;
	try {
		const first = user("hello", 1);
		const reply = await send([first]);
		transport.promptImpl = async (req) => {
			sideHook = await transport.hook({ id: "side-1", name: "run_command", args: { CommandLine: "ls" } }, req.sessionId);
			transport.text(req.sessionId, "the summary");
			return { stopReason: "end_turn", usage: USAGE };
		};
		const sideContext = {
			messages: [createInitialSystemMessage("Summarize the conversation.", [])!, user("<conversation>hello</conversation>", 5)],
		} as TranscriptContext;
		const side = await handle.provider.streamSimple(model, sideContext, { signal: new AbortController().signal }).result();
		assert.equal(side.stopReason, "stop", side.errorMessage);
		assert.equal(textOf(side), "the summary");
		assert.equal(sideHook?.decision, "deny");
		const sideCall = transport.prompts[1];
		assert.notEqual(sideCall.sessionId, "acp-1");
		assert.match(promptText(sideCall), /Summarize the conversation\.[\s\S]*<conversation>hello<\/conversation>/);
		assert.equal(transport.sessionConfig(sideCall.sessionId), undefined);

		transport.promptImpl = async () => ({ stopReason: "end_turn", usage: USAGE });
		await send([first, reply, user("next", reply.timestamp + 1)]);
		assert.equal(transport.prompts[2].sessionId, "acp-1");
		assert.equal(promptText(transport.prompts[2]), "next");
	} finally {
		await handle.dispose();
	}
});

test("turns another model answered since Agy's last reply are sent as context", async () => {
	const { transport, send, handle } = setup();
	try {
		const first = user("hello", 1);
		const reply = await send([first]);
		const other: AssistantMessage = { ...reply, provider: "anthropic", model: "claude", responseId: undefined, content: [{ type: "text", text: "Claude answered" }], timestamp: reply.timestamp + 2 };
		await send([first, reply, user("question for claude", reply.timestamp + 1), other, user("back to agy", reply.timestamp + 3)]);
		assert.equal(transport.createdSessions.length, 1);
		const sent = promptText(transport.prompts[1]);
		assert.match(sent, /<conversation_context>[\s\S]*question for claude[\s\S]*Claude answered[\s\S]*<\/conversation_context>\nback to agy$/);
		assert.doesNotMatch(sent, /hello/);
	} finally {
		await handle.dispose();
	}
});

test("a branch that dropped Agy's last reply starts a fresh Agy session with the kept conversation", async () => {
	let view: SessionView | undefined;
	const { transport, send, handle, notices } = setup({ getSessionView: () => view });
	try {
		const first = user("hello", 1);
		const reply = await send([first]);
		const mark = { acpSessionId: "acp-1", timestamp: reply.timestamp };
		view = { branchMark: mark, latestTimestamp: () => reply.timestamp + 100, retired: new Set() };
		await send([first, reply, user("different path", reply.timestamp + 1)]);
		assert.equal(transport.createdSessions.length, 2);
		assert.equal(transport.prompts[1].sessionId, "acp-2");
		assert.match(promptText(transport.prompts[1]), /<conversation_history>[\s\S]*hello[\s\S]*<\/conversation_history>\ndifferent path$/);
		assert.equal(notices.length, 1);

		view = { branchMark: mark, latestTimestamp: () => reply.timestamp, retired: new Set(["acp-1"]) };
		await send([first, reply, user("forked", reply.timestamp + 1)]);
		assert.equal(transport.createdSessions.length, 3);
	} finally {
		await handle.dispose();
	}
});

test("the Pi session's role instructions are appended to Agy's prompt when its session starts or resumes", async () => {
	const fresh = setup({ getSystemPromptAppend: () => "ROLE" });
	const resumed = setup({
		getSystemPromptAppend: () => "ROLE",
		getSessionView: () => ({ branchMark: { acpSessionId: "old-acp", timestamp: 10 }, latestTimestamp: () => 10, retired: new Set() }),
	});
	try {
		await fresh.send([user("hello", 1)]);
		assert.deepEqual(fresh.transport.createdSessions.map((s) => s.systemPromptAppend), ["ROLE"]);
		const earlier: AssistantMessage = { role: "assistant", content: [{ type: "text", text: "old" }], api: "agy-native", provider: "agy-native", model: "gemini-4-neon", usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: "stop", timestamp: 10 };
		await resumed.send([user("hello", 1), earlier, user("still there?", 11)]);
		assert.deepEqual(resumed.transport.resumedSessions.map((s) => s.systemPromptAppend), ["ROLE"]);
	} finally {
		await fresh.handle.dispose();
		await resumed.handle.dispose();
	}
});

test("Agy gets the appended system prompt and Squire's delegate section, not the rest of Pi's prompt", () => {
	const role = agyRoleInstructions({
		appendSystemPrompt: "Find the evidence the parent needs.",
		sections: { squire_delegate_task: "You are a delegate.", posthorse: "Pi rolls context over automatically." },
	});
	assert.equal(role, "You are a delegate.\n\nFind the evidence the parent needs.");
	assert.equal(agyRoleInstructions({ appendSystemPrompt: "", sections: { posthorse: "x" } }), undefined);
});

test("an Agy session from an earlier Pi process is resumed, not replayed", async () => {
	const { transport, send, handle } = setup({
		getSessionView: () => ({ branchMark: { acpSessionId: "old-acp", timestamp: 10 }, latestTimestamp: () => 10, retired: new Set() }),
	});
	try {
		const earlier: AssistantMessage = { role: "assistant", content: [{ type: "text", text: "old" }], api: "agy-native", provider: "agy-native", model: "gemini-4-neon", usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: "stop", timestamp: 10 };
		await send([user("hello", 1), earlier, user("still there?", 11)]);
		assert.deepEqual(transport.resumedSessions.map((s) => s.sessionId), ["old-acp"]);
		assert.equal(transport.createdSessions.length, 0);
		assert.equal(promptText(transport.prompts[0]), "still there?");
	} finally {
		await handle.dispose();
	}
});

test("Esc cancels the Agy turn and denies the parked tool call", async () => {
	const { transport, send, handle } = setup();
	const decision = deferred<unknown>();
	transport.promptImpl = async (req) => {
		decision.resolve(await transport.hook({ id: "tc-1", name: "run_command", args: { CommandLine: "sleep 60" } }, req.sessionId));
		return { stopReason: "cancelled", usage: undefined };
	};
	try {
		const controller = new AbortController();
		const first = user("wait", 1);
		const parked = await send([first], { signal: controller.signal });
		assert.equal(parked.stopReason, "toolUse");
		controller.abort();
		const resumed = await send([first, parked, toolResult(parked, "aborted", true, 3)], { signal: controller.signal });
		assert.equal(resumed.stopReason, "aborted");
		assert.equal((await decision.promise as { decision: string }).decision, "deny");
		assert.deepEqual(transport.cancelledSessions, ["acp-1"]);
	} finally {
		await handle.dispose();
	}
});

test("a message typed during tool execution is sent to Agy as a steering prompt", async () => {
	const { transport, send, handle } = setup();
	const steered = deferred<void>();
	transport.promptImpl = async (req) => {
		if (req.steer) {
			transport.text(req.sessionId, "Switched.");
			steered.resolve();
			return { stopReason: "end_turn", usage: USAGE };
		}
		await transport.hook({ id: "tc-1", name: "run_command", args: { CommandLine: "ls" } }, req.sessionId);
		await steered.promise;
		return { stopReason: "end_turn", usage: USAGE };
	};
	try {
		const first = user("list files", 1);
		const parked = await send([first]);
		const done = await send([first, parked, toolResult(parked, "denied", true, 3), user("do something else", 4)]);
		assert.equal(textOf(done), "Switched.");
		assert.equal(transport.prompts.at(-1)?.steer, true);
		assert.equal(promptText(transport.prompts.at(-1)!), "do something else");
	} finally {
		await handle.dispose();
	}
});

test("mid-stream quota retry notices cancel the turn and fail without leaking warning text", async () => {
	const { transport, send, handle } = setup();
	const cancelled = deferred<void>();
	transport.promptImpl = async (req) => {
		transport.text(req.sessionId, "⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): Quota reached\n");
		req.signal?.addEventListener("abort", () => cancelled.resolve(), { once: true });
		if (req.signal?.aborted) cancelled.resolve();
		await cancelled.promise;
		return { stopReason: "cancelled", usage: undefined };
	};
	try {
		const reply = await send([user("hello", 1)]);
		assert.equal(reply.stopReason, "error");
		assert.match(reply.errorMessage ?? "", /Antigravity rate limited:.*RESOURCE_EXHAUSTED/);
		assert.equal(textOf(reply), "");
		assert.deepEqual(transport.cancelledSessions, ["acp-1"]);
	} finally {
		await handle.dispose();
	}
});

test("final quota warning line fails the turn even without a code 429 suffix", async () => {
	const { transport, send, handle } = setup();
	transport.promptImpl = async (req) => {
		transport.text(req.sessionId, "⚠ Individual quota reached. Resets in 2h.");
		return { stopReason: "end_turn", usage: USAGE };
	};
	try {
		const reply = await send([user("hello", 1)]);
		assert.equal(reply.stopReason, "error");
		assert.match(reply.errorMessage ?? "", /Individual quota reached/);
		assert.equal(textOf(reply), "");
	} finally {
		await handle.dispose();
	}
});

test("effort-suffixed model rosters select the matching variant and skip thought_level", async () => {
	const setCalls: Array<{ configId: string; value: string }> = [];
	const { transport, handle } = setup();
	const suffixedConfig = (): AcpSessionConfig => ({
		model: {
			configId: "model",
			currentValue: "gemini-4-hydrogen-medium",
			choices: [
				{ value: "gemini-4-hydrogen-low", name: "Hydrogen Low", description: undefined },
				{ value: "gemini-4-hydrogen-medium", name: "Hydrogen Medium", description: undefined },
				{ value: "gemini-4-hydrogen-high", name: "Hydrogen High", description: undefined },
			],
		},
		thoughtLevel: {
			configId: "thought_level",
			currentValue: "medium",
			choices: [
				{ value: "low", name: "Low", description: undefined },
				{ value: "medium", name: "Medium", description: undefined },
				{ value: "high", name: "High", description: undefined },
			],
		},
		availableModelVariants: [],
	});
	transport.createSession = async (options) => {
		transport.createdSessions.push(options);
		const cfg = suffixedConfig();
		return { sessionId: "acp-1", config: cfg };
	};
	transport.sessionConfig = () => suffixedConfig();
	transport.setConfigOption = async (_sessionId, configId, value) => {
		setCalls.push({ configId, value });
		const cfg = suffixedConfig();
		if (configId === "model" && cfg.model) cfg.model.currentValue = value;
		return cfg;
	};
	try {
		const hydrogen = handle.provider.getModels().find((m) => m.id === "gemini-4-hydrogen")!;
		const reply = await handle.provider
			.streamSimple(hydrogen, makeContext([user("hi", 1)]), { ...live(handle), reasoning: "high" } as StreamOptions)
			.result();
		assert.equal(reply.stopReason, "stop", reply.errorMessage);
		assert.deepEqual(setCalls, [{ configId: "model", value: "gemini-4-hydrogen-high" }]);
	} finally {
		await handle.dispose();
	}
});

test("a rewind or fork resumes a native SQLite fork when available, and falls back to replay if resume fails", async () => {
	let view: SessionView | undefined;
	const recorded: Array<{ sessionId: string; timestamp: number }> = [];
	const deleted: string[] = [];
	let nextForkId: string | undefined = "fork-acp-1";
	const { transport, send, handle, notices } = setup({
		getSessionView: () => view,
		conversationStore: {
			recordCheckpoint: (_dir, sessionId, timestamp) => {
				recorded.push({ sessionId, timestamp });
				return { steps: 1, generations: 1, executions: 1 };
			},
			forkConversation: () => nextForkId,
			deleteConversation: (_dir, id) => {
				deleted.push(id);
			},
		},
	});
	try {
		const first = user("hello", 1);
		const reply = await send([first]);
		assert.deepEqual(recorded, [{ sessionId: "acp-1", timestamp: reply.timestamp }]);

		// Rewind: native fork succeeds -> session/resume on fork-acp-1, only new message sent, no replay notice.
		const mark = { acpSessionId: "acp-1", timestamp: reply.timestamp };
		view = { branchMark: mark, latestTimestamp: () => reply.timestamp + 100, retired: new Set() };
		await send([first, reply, user("branched question", reply.timestamp + 1)]);
		assert.equal(transport.createdSessions.length, 1);
		assert.deepEqual(transport.resumedSessions.map((s) => s.sessionId), ["fork-acp-1"]);
		assert.equal(promptText(transport.prompts[1]), "branched question");
		assert.equal(notices.length, 0);

		// If resumeSession rejects the fork, delete the orphaned fork and fall back cleanly to session/new + replay.
		nextForkId = "fork-acp-bad";
		transport.resumeSession = async () => {
			throw new Error("corrupt fork");
		};
		await send([first, reply, user("fallback question", reply.timestamp + 2)]);
		assert.deepEqual(deleted, ["fork-acp-bad"]);
		assert.equal(transport.createdSessions.length, 2);
		assert.match(promptText(transport.prompts[2]), /<conversation_history>[\s\S]*hello[\s\S]*<\/conversation_history>\nfallback question$/);
		assert.equal(notices.length, 1);
	} finally {
		await handle.dispose();
	}
});

test("configured Pi tools are projected over MCP, auto-approved in PreToolUse, and executed through Pi", async () => {
	const delegateTool: Tool = {
		name: "delegate",
		description: "Delegate a task to a Squire worker",
		parameters: { type: "object", properties: { task: { type: "string" } }, required: ["task"] } as never,
	};
	const readTool: Tool = {
		name: "read",
		description: "Pi read",
		parameters: { type: "object", properties: { path: { type: "string" } } } as never,
	};
	const { transport, send, handle } = setup({
		mcpToolPatterns: ["delegate", "read"],
		getAvailableTools: () => [delegateTool, readTool],
	});
	let hookDecision: { decision: string; reason?: string } | undefined;
	let mcpResponse: unknown;
	transport.promptImpl = async (req) => {
		const mcp = transport.createdSessions[0]?.mcpServers?.[0];
		assert.ok(mcp, "expected HTTP MCP server on live session");
		const headers = { "content-type": "application/json", [mcp.headers[0].name]: mcp.headers[0].value };
		// Verify only `delegate` (not reserved `read`) is advertised.
		const listRes = await fetch(mcp.url, {
			method: "POST",
			headers,
			body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/list" }),
		});
		const listJson = (await listRes.json()) as { result: { tools: Array<{ name: string }> } };
		assert.deepEqual(listJson.result.tools.map((t) => t.name), ["delegate"]);

		// Agy's PreToolUse hook fires first and should auto-approve our own MCP tool without double-gating.
		hookDecision = await transport.hook({ id: "tc-mcp-1", name: "mcp_pi-tools_delegate", args: { task: "fix bug" } }, req.sessionId);

		// Then Agy invokes the MCP server, which parks until Pi executes the real `delegate` tool.
		const callRes = await fetch(mcp.url, {
			method: "POST",
			headers,
			body: JSON.stringify({
				jsonrpc: "2.0",
				id: 2,
				method: "tools/call",
				params: { name: "delegate", arguments: { task: "fix bug" } },
			}),
		});
		mcpResponse = await callRes.json();
		transport.text(req.sessionId, "Delegated.");
		return { stopReason: "end_turn", usage: USAGE };
	};
	try {
		const first = user("delegate fixing the bug", 1);
		const parked = await send([first]);
		assert.equal(parked.stopReason, "toolUse");
		const call = parked.content.find((part) => part.type === "toolCall");
		assert.ok(call && call.type === "toolCall");
		assert.equal(call.name, "delegate");
		assert.deepEqual(call.arguments, { task: "fix bug" });
		assert.deepEqual(hookDecision, { decision: "auto_approve" });

		const done = await send([first, parked, toolResult(parked, "worker-123 launched", false, 3)]);
		assert.equal(done.stopReason, "stop");
		assert.equal(textOf(done), "Delegated.");
		assert.deepEqual(mcpResponse, {
			jsonrpc: "2.0",
			id: 2,
			result: { content: [{ type: "text", text: "worker-123 launched" }] },
		});
	} finally {
		await handle.dispose();
	}
});
