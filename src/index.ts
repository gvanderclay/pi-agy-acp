/**
 * Pi extension entry point: registers the `agy-acp` provider backed by
 * Antigravity CLI ACP (`agy acp`) with full Pi tool execution.
 */

import { appendFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { getAgentDir, type ExtensionAPI, type SessionEntry } from "@earendil-works/pi-coding-agent";
import { isRecord } from "./acp-wire.ts";
import { createAgyAcpProvider, type PersistedAcpBinding } from "./provider.ts";

const BINDING_ENTRY_TYPE = "agy-acp-state";

function extractPersistedBinding(branch: readonly SessionEntry[]): PersistedAcpBinding | undefined {
	for (let i = branch.length - 1; i >= 0; i--) {
		const entry = branch[i];
		if (entry.type !== "custom" || entry.customType !== BINDING_ENTRY_TYPE) continue;
		const data = entry.data;
		if (
			isRecord(data) &&
			typeof data.piSessionId === "string" &&
			typeof data.acpSessionId === "string" &&
			typeof data.syncedFingerprint === "string" &&
			typeof data.systemPromptHash === "string" &&
			typeof data.cwd === "string" &&
			typeof data.modelId === "string"
		) {
			return {
				piSessionId: data.piSessionId,
				acpSessionId: data.acpSessionId,
				syncedFingerprint: data.syncedFingerprint,
				systemPromptHash: data.systemPromptHash,
				cwd: data.cwd,
				modelId: data.modelId,
			};
		}
	}
	return undefined;
}

function createDebugLogger(cacheDir: string): (message: string) => void {
	const logPath = join(cacheDir, "debug.log");
	let target: "unopened" | "file" | "stderr" = "unopened";
	return (message) => {
		const line = `${new Date().toISOString()} [agy-acp:${process.pid}] ${message}\n`;
		if (target !== "stderr") {
			try {
				if (target === "unopened") mkdirSync(cacheDir, { recursive: true, mode: 0o700 });
				appendFileSync(logPath, line, { mode: 0o600 });
				target = "file";
				return;
			} catch (error) {
				target = "stderr";
				process.stderr.write(`[agy-acp] Cannot write ${logPath}: ${error instanceof Error ? error.message : String(error)}\n`);
			}
		}
		process.stderr.write(line);
	};
}

export default function agyAcpExtension(pi: ExtensionAPI): void {
	const cacheDir = join(getAgentDir(), "cache", "agy-acp");
	const log = process.env.PI_AGY_ACP_DEBUG === "1" ? createDebugLogger(cacheDir) : undefined;
	log?.(`Loaded provider from ${import.meta.url}`);

	let canAppendEntries = false;

	const handle = createAgyAcpProvider({
		cacheDir,
		log,
		onBindingChange: (binding) => {
			if (!canAppendEntries) return;
			try {
				pi.appendEntry(BINDING_ENTRY_TYPE, binding);
			} catch (error) {
				log?.(`appendEntry failed: ${error instanceof Error ? error.message : String(error)}`);
			}
		},
	});

	pi.registerProvider(handle.provider);

	pi.on("session_start", (event, ctx) => {
		canAppendEntries = true;
		const persistedBinding = extractPersistedBinding(ctx.sessionManager.getBranch());
		handle.syncSession({
			piSessionId: ctx.sessionManager.getSessionId(),
			cwd: ctx.cwd,
			persistedBinding,
			isFork: event.reason === "fork",
		});
	});

	pi.on("session_tree", (_event, ctx) => {
		const persistedBinding = extractPersistedBinding(ctx.sessionManager.getBranch());
		handle.syncSession({
			piSessionId: ctx.sessionManager.getSessionId(),
			cwd: ctx.cwd,
			persistedBinding,
		});
	});

	pi.on("session_compact", () => {
		handle.invalidateSession("compact");
	});

	pi.on("context_with_system", (_event, ctx) => {
		handle.expectLiveRequest(ctx.signal);
	});

	pi.on("agent_end", () => {
		handle.abandonParkedTurn();
	});

	pi.on("agent_settled", () => {
		handle.abandonParkedTurn();
	});

	pi.on("session_shutdown", async () => {
		canAppendEntries = false;
		await handle.dispose();
	});
}
