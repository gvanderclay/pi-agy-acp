import assert from "node:assert/strict";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { createInitialSystemMessage, type Message, type TranscriptContext } from "@earendil-works/pi-ai";
import { createAgyNativeProvider, type AgyNativeProviderHandle } from "../src/provider.ts";

const LIVE = process.env.PI_AGY_ACP_LIVE === "1";

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
		const handle = createAgyNativeProvider({ cacheDir });
		try {
			handle.syncSession({ piSessionId: "live-pi-1", cwd: process.cwd() });
			const model = handle.provider.getModels().find((m) => m.id === "gemini-4-neon")!;
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
			rmSync(cacheDir, { recursive: true, force: true });
		}
	},
);

test(
	"live agy acp: a side request follows its instructions",
	{ skip: !LIVE },
	async () => {
		const cacheDir = mkdtempSync(join(tmpdir(), "pi-agy-native-live-"));
		const handle = createAgyNativeProvider({ cacheDir });
		try {
			handle.syncSession({ piSessionId: "live-pi-1", cwd: process.cwd() });
			const model = handle.provider.getModels().find((m) => m.id === "gemini-4-neon")!;
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
			rmSync(cacheDir, { recursive: true, force: true });
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
		const handle = createAgyNativeProvider({ cacheDir, log: (message) => logs.push(message) });
		try {
			handle.syncSession({ piSessionId: "live-pi-1", cwd: process.cwd() });
			const model = handle.provider.getModels().find((m) => m.id === "gemini-4-neon")!;
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
			rmSync(cacheDir, { recursive: true, force: true });
		}
	},
);
