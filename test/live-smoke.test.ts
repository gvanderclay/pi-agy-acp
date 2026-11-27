import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { createInitialSystemMessage, type Message, type Tool, type TranscriptContext } from "@earendil-works/pi-ai";
import { createAgyAcpProvider, type AgyAcpProviderHandle, type PersistedAcpBinding } from "../src/provider.ts";

const LIVE = process.env.PI_AGY_ACP_LIVE === "1";

function liveRequest(handle: AgyAcpProviderHandle, signal = new AbortController().signal) {
	handle.expectLiveRequest(signal);
	return { signal, onProviderStreamEvent: () => {}, sessionId: "live-pi-1", reasoning: "medium" as const };
}

function makeContext(systemPrompt: string, tools: Tool[], messages: Message[]): TranscriptContext {
	const sys = createInitialSystemMessage(systemPrompt, tools);
	return { messages: sys ? [sys, ...messages] : [...messages] } as TranscriptContext;
}

test(
	"live agy acp end-to-end: model discovery, parked Pi tool execution, and cross-process session/resume",
	{ skip: !LIVE },
	async () => {
		const cacheDir = mkdtempSync(join(tmpdir(), "pi-agy-acp-live-"));
		let savedBinding: PersistedAcpBinding | undefined;

		const codemodeTool: Tool = {
			name: "codemode",
			description: "Execute JavaScript in Pi and return the computed result.",
			parameters: {
				type: "object",
				properties: { code: { type: "string", description: "JS expression" } },
				required: ["code"],
			} as never,
		};

		const handle1 = createAgyAcpProvider({
			cacheDir,
			onBindingChange: (b) => {
				savedBinding = b;
			},
		});

		try {
			// 1. Live model discovery
			await handle1.provider.refreshModels!({
				allowNetwork: true,
				signal: new AbortController().signal,
				publish: async (pub) => {
					pub.update?.();
					return true;
				},
			});
			const flash = handle1.provider.getModels().find((m) => m.id === "gemini-4-neon");
			assert.ok(flash, `expected gemini-4-neon in discovered models (${handle1.provider.getModels().map((m) => m.id).join(", ")})`);

			// 2. Parked Pi tool execution round-trip
			handle1.syncSession({ piSessionId: "live-pi-1", cwd: process.cwd() });
			const sysPrompt = "Project rule: Secret word: VELVET-9012. Always call codemode when asked.";
			const user1: Message = {
				role: "user",
				content: "Call the codemode tool with code `21 * 2` and then state the returned number.",
				timestamp: Date.now(),
			};

			const signal = new AbortController().signal;
			const s1 = handle1.provider.streamSimple(flash, makeContext(sysPrompt, [codemodeTool], [user1]), liveRequest(handle1, signal));
			const m1 = await s1.result();
			assert.equal(m1.stopReason, "toolUse", `expected toolUse, got ${m1.stopReason}: ${m1.errorMessage ?? ""}`);
			const call = m1.content.find((b) => b.type === "toolCall");
			assert.ok(call && call.type === "toolCall");
			assert.equal(call.name, "codemode");

			const tr1: Message = {
				role: "toolResult",
				toolCallId: call.id,
				toolName: "codemode",
				content: [{ type: "text", text: "42" }],
				isError: false,
				timestamp: Date.now(),
			};

			const s2 = handle1.provider.streamSimple(flash, makeContext(sysPrompt, [codemodeTool], [user1, m1, tr1]), liveRequest(handle1, signal));
			const m2 = await s2.result();
			assert.equal(m2.stopReason, "stop");
			const text2 = m2.content.filter((b) => b.type === "text").map((b) => b.text).join("");
			assert.match(text2, /42/);
			assert.ok(m2.usage.totalTokens > 0);
			assert.ok(savedBinding, "expected persisted binding after completed turn");

			await handle1.dispose();

			// 3. Cross-process session/resume with a brand-new provider + fresh agy acp process + new MCP port
			const handle2 = createAgyAcpProvider({ cacheDir });
			try {
				handle2.syncSession({ piSessionId: "live-pi-1", cwd: process.cwd(), persistedBinding: savedBinding });
				const user2: Message = {
					role: "user",
					content: "Without calling any tool, repeat the secret word from your system instructions and the number codemode returned.",
					timestamp: Date.now(),
				};
				const s3 = handle2.provider.streamSimple(
					flash,
					makeContext(sysPrompt, [codemodeTool], [user1, m1, tr1, m2, user2]),
					liveRequest(handle2),
				);
				const m3 = await s3.result();
				assert.equal(m3.stopReason, "stop");
				const text3 = m3.content.filter((b) => b.type === "text").map((b) => b.text).join("");
				assert.match(text3, /VELVET-9012/);
				assert.match(text3, /42/);
			} finally {
				await handle2.dispose();
			}
		} finally {
			await handle1.dispose();
			rmSync(cacheDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
		}
	},
);
