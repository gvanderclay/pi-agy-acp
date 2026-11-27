/**
 * Pi extension entry point: registers the `agy-native` provider, which runs the
 * stock Antigravity agent (`agy acp`) and routes its tool calls through Pi.
 */

import { appendFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import type { Message } from "@earendil-works/pi-ai";
import {
	getAgentDir,
	type BuildSystemPromptOptions,
	type ExtensionAPI,
	type ExtensionContext,
	type SessionEntry,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { loadAgyNativeConfig } from "./config.ts";
import {
	AGY_COMMAND_TOOL,
	AGY_EDIT_TOOL,
	AGY_MESSAGE_TOOL,
	AGY_OTHER_TOOL,
	AGY_PROVIDER_ID,
	AGY_SEARCH_TOOL,
	AGY_SUBAGENT_TOOL,
	AGY_TOOL_NAMES,
	agyTurnMark,
	agyRoleInstructions,
	createAgyNativeProvider,
	type AgyTurnMark,
	type SessionView,
} from "./provider.ts";
import { isRecord } from "./acp-wire.ts";

/** Agy sessions a forked Pi session must not continue; the session it was forked from owns them. */
const RETIRED_ENTRY_TYPE = "agy-native-retired";
const COMPACTION_SUMMARY = "Pi compacted its copy of this conversation without a summary. Antigravity keeps the full conversation in its own session.";

function createDebugLogger(cacheDir: string): (message: string) => void {
	const logPath = join(cacheDir, "debug.log");
	let target: "unopened" | "file" | "stderr" = "unopened";
	return (message) => {
		const line = `${new Date().toISOString()} [agy-native:${process.pid}] ${message}\n`;
		if (target !== "stderr") {
			try {
				if (target === "unopened") mkdirSync(cacheDir, { recursive: true, mode: 0o700 });
				appendFileSync(logPath, line, { mode: 0o600 });
				target = "file";
				return;
			} catch (error) {
				target = "stderr";
				process.stderr.write(`[agy-native] Cannot write ${logPath}: ${error instanceof Error ? error.message : String(error)}\n`);
			}
		}
		process.stderr.write(line);
	};
}

function turnMarks(entries: readonly SessionEntry[]): AgyTurnMark[] {
	return entries.flatMap((entry) => (entry.type === "message" ? (agyTurnMark(entry.message as Message) ?? []) : []));
}

function retiredSessions(entries: readonly SessionEntry[]): Set<string> {
	const retired = new Set<string>();
	for (const entry of entries) {
		if (entry.type !== "custom" || entry.customType !== RETIRED_ENTRY_TYPE || !isRecord(entry.data)) continue;
		const ids = entry.data.acpSessionIds;
		if (Array.isArray(ids)) for (const id of ids) if (typeof id === "string") retired.add(id);
	}
	return retired;
}

export default function agyNativeExtension(pi: ExtensionAPI): void {
	const cacheDir = join(getAgentDir(), "cache", "agy-native");
	const log = process.env.PI_AGY_ACP_DEBUG === "1" ? createDebugLogger(cacheDir) : undefined;
	const config = loadAgyNativeConfig(join(getAgentDir(), "agy-native.json"), log);
	log?.(`Loaded provider from ${import.meta.url} (mcpTools=${config.mcpTools.length})`);

	let session: ExtensionContext | undefined;
	// The run's prompt options; handlers that run after this one still edit the same object.
	let promptOptions: BuildSystemPromptOptions | undefined;

	const handle = createAgyNativeProvider({
		cacheDir,
		geminiDir: join(getAgentDir(), "agy-native"),
		log,
		notify: (message) => session?.ui.notify(message, "warning"),
		mcpToolPatterns: config.mcpTools,
		getAvailableTools: () => {
			const active = new Set(pi.getActiveTools());
			return pi.getAllTools().filter((tool) => tool.exposure !== "hidden" && active.has(tool.name));
		},
		getSystemPromptAppend: () =>
			promptOptions ? agyRoleInstructions(promptOptions, config.systemPromptSections) : undefined,
		getSessionView: (): SessionView | undefined => {
			const manager = session?.sessionManager;
			if (!manager) return undefined;
			const all = manager.getEntries();
			const latest = new Map<string, number>();
			for (const mark of turnMarks(all)) latest.set(mark.acpSessionId, Math.max(latest.get(mark.acpSessionId) ?? 0, mark.timestamp));
			return {
				branchMark: turnMarks(manager.getBranch()).at(-1),
				latestTimestamp: (acpSessionId) => latest.get(acpSessionId),
				retired: retiredSessions(all),
			};
		},
	});

	pi.registerProvider(handle.provider);

	const run = (toolCallId: string, signal: AbortSignal | undefined) =>
		handle.runApprovedTool(toolCallId, signal).then(({ text, isError }) => {
			if (isError) throw new Error(text);
			return { content: [{ type: "text" as const, text }], details: {} };
		});
	const subagent = Type.Optional(Type.Boolean({ description: "True when an Agy subagent made the call" }));
	// A hidden tool is registered but never declared or callable. An inactive one is not enough: when an
	// extension registers a tool, Pi re-activates every inactive tool a `--tools` allowlist names.
	const registerApprovalTools = (exposure: "direct" | "hidden"): void => {
		pi.registerTool({
			name: AGY_COMMAND_TOOL,
			label: "Agy command",
			description: "A shell command Antigravity wants to run. Only the agy-native provider calls this tool.",
			parameters: Type.Object({
				command: Type.String({ description: "Command line" }),
				cwd: Type.String({ description: "Working directory" }),
				subagent,
			}),
			defaultActive: false,
			exposure,
			execute: (toolCallId, _params, signal) => run(toolCallId, signal),
		});
		pi.registerTool({
			name: AGY_EDIT_TOOL,
			label: "Agy edit",
			description: "A file Antigravity wants to edit. Only the agy-native provider calls this tool.",
			parameters: Type.Object({ path: Type.String({ description: "File path" }), subagent }),
			defaultActive: false,
			exposure,
			execute: (toolCallId, _params, signal) => run(toolCallId, signal),
		});
		pi.registerTool({
			name: AGY_SEARCH_TOOL,
			label: "Agy web search",
			description: "A web search Antigravity wants to run. Only the agy-native provider calls this tool.",
			parameters: Type.Object({ query: Type.String({ description: "Search query" }), subagent }),
			defaultActive: false,
			exposure,
			execute: (toolCallId, _params, signal) => run(toolCallId, signal),
		});
		const agyArguments = Type.Record(Type.String(), Type.Unknown(), { description: "Antigravity tool arguments" });
		pi.registerTool({
			name: AGY_SUBAGENT_TOOL,
			label: "Agy subagent",
			description: "An Antigravity subagent Antigravity wants to start. Only the agy-native provider calls this tool.",
			parameters: Type.Object({ arguments: agyArguments, subagent }),
			defaultActive: false,
			exposure,
			execute: (toolCallId, _params, signal) => run(toolCallId, signal),
		});
		pi.registerTool({
			name: AGY_MESSAGE_TOOL,
			label: "Agy message",
			description: "A message between Antigravity's agent and its subagents. Only the agy-native provider calls this tool.",
			parameters: Type.Object({ arguments: agyArguments, subagent }),
			defaultActive: false,
			exposure,
			execute: (toolCallId, _params, signal) => run(toolCallId, signal),
		});
		pi.registerTool({
			name: AGY_OTHER_TOOL,
			label: "Agy tool",
			description: "Another Antigravity tool call. Only the agy-native provider calls this tool.",
			parameters: Type.Object({
				tool: Type.String({ description: "Antigravity tool name" }),
				arguments: Type.Record(Type.String(), Type.Unknown(), { description: "Antigravity tool arguments" }),
				path: Type.Optional(Type.String({ description: "File the call touches, if any" })),
				subagent,
			}),
			defaultActive: false,
			exposure,
			execute: (toolCallId, _params, signal) => run(toolCallId, signal),
		});
	};
	registerApprovalTools("hidden");
	let exposed = false;

	/** The approval tools run only for Agy; other models must not see them. */
	const syncTools = (onAgy: boolean): void => {
		if (exposed !== onAgy) {
			exposed = onAgy;
			registerApprovalTools(onAgy ? "direct" : "hidden");
		}
		const active = new Set(pi.getActiveTools());
		const before = active.size;
		for (const name of AGY_TOOL_NAMES) {
			if (onAgy) active.add(name);
			else active.delete(name);
		}
		if (active.size !== before) pi.setActiveTools([...active]);
	};
	const onAgy = (ctx: ExtensionContext): boolean => ctx.model?.provider === AGY_PROVIDER_ID;

	pi.on("session_start", (event, ctx) => {
		session = ctx;
		if (event.reason === "fork") {
			const ids = [...new Set(turnMarks(ctx.sessionManager.getEntries()).map((mark) => mark.acpSessionId))];
			if (ids.length > 0) pi.appendEntry(RETIRED_ENTRY_TYPE, { acpSessionIds: ids });
		}
		handle.syncSession({ piSessionId: ctx.sessionManager.getSessionId(), cwd: ctx.cwd });
		syncTools(onAgy(ctx));
	});

	pi.on("session_tree", (_event, ctx) => {
		handle.syncSession({ piSessionId: ctx.sessionManager.getSessionId(), cwd: ctx.cwd });
	});

	pi.on("model_select", (event) => {
		syncTools(event.model.provider === AGY_PROVIDER_ID);
	});

	// Agy keeps the full conversation, so it is not sent the fixed summary written below.
	pi.on("context", (event, ctx) => {
		if (!onAgy(ctx)) return undefined;
		const messages = event.messages.filter((message) => message.role !== "compactionSummary");
		return messages.length === event.messages.length ? undefined : { messages };
	});

	pi.on("session_before_compact", (event, ctx) => {
		if (!onAgy(ctx)) return undefined;
		return {
			compaction: {
				summary: COMPACTION_SUMMARY,
				firstKeptEntryId: event.preparation.firstKeptEntryId,
				tokensBefore: event.preparation.tokensBefore,
			},
		};
	});

	pi.on("before_agent_start", (event) => {
		promptOptions = event.systemPromptOptions;
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
		session = undefined;
		await handle.dispose();
	});
}
