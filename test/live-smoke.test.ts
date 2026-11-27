import assert from "node:assert/strict";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { createInitialSystemMessage, type Message, type TranscriptContext } from "@earendil-works/pi-ai";
import { createAgyNativeProvider, type AgyNativeProviderHandle } from "../src/provider.ts";

const LIVE = process.env.PI_AGY_ACP_LIVE === "1";
const MODEL = "gemini-3.8-flash";

function live(handle: AgyNativeProviderHandle) {
	const signal = new AbortController().signal;
	handle.expectLiveRequest(signal);
	return { signal, onProviderStreamEvent: () => {}, sessionId: "live-pi-1", reasoning: "medium" as const };
}

const context = (messages: Message[]): TranscriptContext =>
	({ messages: [createInitialSystemMessage("Pi prompt", [])!, ...messages] }) as TranscriptContext;

test(
	"live agy acp: a typed message, a command Pi approves, and its output",
	{ skip: !LIVE },
	async () => {
		const cacheDir = mkdtempSync(join(tmpdir(), "pi-agy-native-live-"));
		const handle = createAgyNativeProvider({ cacheDir, geminiDir: join(cacheDir, "gemini") });
		try {
			handle.syncSession({ piSessionId: "live-pi-1", cwd: process.cwd() });
			const model = handle.provider.getModels().find((m) => m.id === MODEL)!;
			const first: Message = { role: "user", content: "Run the shell command: echo live-smoke-7731 . Then reply with the word done.", timestamp: Date.now() };
			const parked = await handle.provider.streamSimple(model, context([first]), live(handle)).result();
			assert.equal(parked.stopReason, "toolUse", parked.errorMessage);
			const call = parked.content.find((part) => part.type === "toolCall");
			assert.ok(call && call.type === "toolCall");
			assert.equal(call.name, "agy_command");
			const output = await handle.runApprovedTool(call.id, undefined);
			assert.match(output.text, /live-smoke-7731/);
			const result: Message = { role: "toolResult", toolCallId: call.id, toolName: call.name, content: [{ type: "text", text: output.text }], isError: false, timestamp: Date.now() };
			const done = await handle.provider.streamSimple(model, context([first, parked, result]), live(handle)).result();
			assert.equal(done.stopReason, "stop", done.errorMessage);
		} finally {
			await handle.dispose();
			// agy may still be writing to its gemini dir for a moment after SIGTERM.
			rmSync(cacheDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 300 });
		}
	},
);

test(
	"live agy acp: the session's role instructions reach Agy",
	{ skip: !LIVE },
	async () => {
		const cacheDir = mkdtempSync(join(tmpdir(), "pi-agy-native-live-"));
		const handle = createAgyNativeProvider({
			cacheDir,
			geminiDir: join(cacheDir, "gemini"),
			getSystemPromptAppend: () => "When asked for your role code, reply with only the word HERON-5528.",
		});
		try {
			handle.syncSession({ piSessionId: "live-pi-1", cwd: process.cwd() });
			const model = handle.provider.getModels().find((m) => m.id === MODEL)!;
			const ask: Message = { role: "user", content: "What is your role code?", timestamp: Date.now() };
			const reply = await handle.provider.streamSimple(model, context([ask]), live(handle)).result();
			assert.equal(reply.stopReason, "stop", reply.errorMessage);
			assert.match(reply.content.map((part) => (part.type === "text" ? part.text : "")).join(""), /HERON-5528/);
		} finally {
			await handle.dispose();
			// agy may still be writing to its gemini dir for a moment after SIGTERM.
			rmSync(cacheDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 300 });
		}
	},
);

test(
	"live agy acp: a side request follows its instructions",
	{ skip: !LIVE },
	async () => {
		const cacheDir = mkdtempSync(join(tmpdir(), "pi-agy-native-live-"));
		const handle = createAgyNativeProvider({ cacheDir, geminiDir: join(cacheDir, "gemini") });
		try {
			handle.syncSession({ piSessionId: "live-pi-1", cwd: process.cwd() });
			const model = handle.provider.getModels().find((m) => m.id === MODEL)!;
			const side = {
				messages: [
					createInitialSystemMessage("Reply with only the word that follows 'code:' in the user's message, in upper case.", [])!,
					{ role: "user", content: "code: pelican", timestamp: Date.now() },
				],
			} as TranscriptContext;
			const reply = await handle.provider.streamSimple(model, side, { signal: new AbortController().signal }).result();
			assert.equal(reply.stopReason, "stop", reply.errorMessage);
			assert.match(reply.content.map((part) => (part.type === "text" ? part.text : "")).join(""), /PELICAN/);
		} finally {
			await handle.dispose();
			// agy may still be writing to its gemini dir for a moment after SIGTERM.
			rmSync(cacheDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 300 });
		}
	},
);

test(
	"live agy acp: a side request's tool calls are recognised as its own and denied",
	{ skip: !LIVE },
	async () => {
		const cacheDir = mkdtempSync(join(tmpdir(), "pi-agy-native-live-"));
		const marker = join(cacheDir, "side-request-ran");
		const logs: string[] = [];
		const handle = createAgyNativeProvider({ cacheDir, geminiDir: join(cacheDir, "gemini"), log: (message) => logs.push(message) });
		try {
			handle.syncSession({ piSessionId: "live-pi-1", cwd: process.cwd() });
			const model = handle.provider.getModels().find((m) => m.id === MODEL)!;
			const side = {
				messages: [
					createInitialSystemMessage("Use your shell tool exactly once, then reply with one sentence saying whether it ran.", [])!,
					{ role: "user", content: `Run the shell command: touch ${marker}`, timestamp: Date.now() },
				],
			} as TranscriptContext;
			const reply = await handle.provider.streamSimple(model, side, { signal: new AbortController().signal }).result();
			assert.equal(reply.stopReason, "stop", reply.errorMessage);
			assert.equal(existsSync(marker), false);
			assert.ok(logs.some((line) => /^hook \S+ denied: side request$/.test(line)), logs.join("\n"));
		} finally {
			await handle.dispose();
			// agy may still be writing to its gemini dir for a moment after SIGTERM.
			rmSync(cacheDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 300 });
		}
	},
);
