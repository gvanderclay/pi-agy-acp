/**
 * Pi `Provider` (`agy-native`) that runs the stock Antigravity agent (`agy acp`).
 *
 * Agy keeps its own prompt, tools and history. Each Agy tool call reaches Pi
 * through the `pi-gate` hook (see `hook-gate.ts`) and is shown to Pi as an
 * approval tool call, so Pi's permission system decides whether it runs.
 */

import {
	calculateCost,
	createAssistantMessageEventStream,
	getCurrentSystemPrompt,
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type JsonObject,
	type Message,
	type Model,
	type ModelThinkingLevel,
	type Provider,
	type RefreshModelsContext,
	type SimpleStreamOptions,
	type StreamOptions,
	type ThinkingLevelMap,
	type Tool,
	type ToolResultMessage,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";
import { estimateTokens } from "@earendil-works/pi-coding-agent";
import { join } from "node:path";
import { AgyAcpClient, resolveAgyBinary, type AcpMcpServerHttp, type AcpTransport } from "./acp-client.ts";
import {
	isRecord,
	type AcpChoice,
	type AcpPermissionRequest,
	type AcpPromptBlock,
	type AcpPromptResult,
	type AcpSessionConfig,
	type AcpSessionUpdate,
	type AcpUsage,
} from "./acp-wire.ts";
import { DEFAULT_SYSTEM_PROMPT_SECTIONS, selectMcpTools, type AgyPermissionsMode } from "./config.ts";
import { defaultConversationStore, type ConversationStore } from "./conversation-store.ts";
import { blankAcpTurnError, terminalAcpError, WarningLineFilter } from "./errors.ts";
import {
	GATE_URL_ENV,
	GateServer,
	installHook,
	uninstallHook,
	type GateRequest,
	type HookDecision,
	type HookToolCall,
} from "./hook-gate.ts";
import {
	isOwnMcpToolName,
	MCP_SERVER_NAME,
	MCP_TOKEN_HEADER,
	McpBridgeServer,
	type ParkedMcpCall,
} from "./mcp-server.ts";
import { extractPostAssistantTail, renderTranscriptDelta, renderTranscriptReplay, userMessageToPromptBlocks } from "./prompt.ts";
import { projectPiTools } from "./tools.ts";

export const AGY_PROVIDER_ID = "agy-native";
export const AGY_API = "agy-native";
export const AGY_COMMAND_TOOL = "agy_command";
export const AGY_EDIT_TOOL = "agy_edit";
export const AGY_SEARCH_TOOL = "agy_search_web";
export const AGY_SUBAGENT_TOOL = "agy_invoke_subagent";
export const AGY_MESSAGE_TOOL = "agy_send_message";
export const AGY_OTHER_TOOL = "agy_tool";
export const AGY_TOOL_NAMES = [
	AGY_COMMAND_TOOL,
	AGY_EDIT_TOOL,
	AGY_SEARCH_TOOL,
	AGY_SUBAGENT_TOOL,
	AGY_MESSAGE_TOOL,
	AGY_OTHER_TOOL,
] as const;
/** `responseId` prefix naming the Agy session that produced a reply. */
const SESSION_RESPONSE_PREFIX = "agy:";

const EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"] as const;
const DEFAULT_PARK_BATCH_MS = 10;
/** Below the hook's one-hour timeout, so Agy gets Pi's reason instead of a hook timeout. */
const DEFAULT_HOOK_DEADLINE_MS = 55 * 60_000;

/** An Agy reply on the Pi branch: which Agy session wrote it, and when. */
export interface AgyTurnMark {
	acpSessionId: string;
	timestamp: number;
}

export function agyTurnMark(message: Message): AgyTurnMark | undefined {
	if (message.role !== "assistant" || message.provider !== AGY_PROVIDER_ID) return undefined;
	const id = message.responseId;
	if (!id?.startsWith(SESSION_RESPONSE_PREFIX)) return undefined;
	return { acpSessionId: id.slice(SESSION_RESPONSE_PREFIX.length), timestamp: message.timestamp };
}

/** What the Pi session file says about Agy sessions, beyond the messages in the request. */
export interface SessionView {
	/** The newest Agy reply on the current branch, even if compaction dropped it from the request. */
	branchMark: AgyTurnMark | undefined;
	/** The newest Agy reply anywhere in the session file for this Agy session. */
	latestTimestamp(acpSessionId: string): number | undefined;
	/** Agy sessions this Pi session must not continue (they belong to the session it was forked from). */
	retired: ReadonlySet<string>;
}

function viewFromMessages(messages: readonly Message[]): SessionView {
	const marks = messages.flatMap((message) => agyTurnMark(message) ?? []);
	return {
		branchMark: marks.at(-1),
		latestTimestamp: (acpSessionId) =>
			marks.filter((mark) => mark.acpSessionId === acpSessionId).at(-1)?.timestamp,
		retired: new Set(),
	};
}

export interface SessionSyncContext {
	piSessionId: string;
	cwd: string;
}

export interface AgyNativeProviderOptions {
	cacheDir: string;
	/** Agy's config and conversation folder for the `agy acp` Pi starts, kept apart from `~/.gemini`. */
	geminiDir: string;
	binaryPath?: string;
	parkBatchMs?: number;
	hookDeadlineMs?: number;
	log?: (message: string) => void;
	/** Shown to the user when Agy starts a new session with the kept conversation. */
	notify?: (message: string) => void;
	getSessionView?: () => SessionView | undefined;
	/** Text appended to Agy's own system prompt for the conversation's session. */
	getSystemPromptAppend?: () => string | undefined;
	/** Optional glob patterns of Pi tools to project to Agy over MCP (empty by default). */
	mcpToolPatterns?: readonly string[];
	/** Returns active Pi tools eligible for selective MCP projection. */
	getAvailableTools?: () => readonly Tool[];
	/** Permission engine for Agy's built-in tools: Pi's `pi-gate` ("pi", default) or Agy's native engine ("agy"). */
	permissions?: AgyPermissionsMode;
	/** Pass `--sandbox` to `agy acp` to run terminal commands inside Agy's OS sandbox. */
	sandbox?: boolean;
	/** Handles Agy's `session/request_permission` prompts when `permissions === "agy"`. */
	onPermissionRequest?: (
		request: AcpPermissionRequest,
		signal: AbortSignal | undefined,
	) => boolean | string | Promise<boolean | string>;
	/** Test seams. */
	createTransport?: (env: Record<string, string>, onPermissionRequest: (toolCallId: string | undefined) => boolean) => AcpTransport;
	conversationStore?: ConversationStore;
	mcpServer?: McpBridgeServer;
	installGate?: () => void;
	uninstallGate?: () => void;
	checkAvailable?: () => boolean;
}

/**
 * The parts of Pi's system prompt that tell this session its role: configured
 * named sections (by default Squire's `squire_delegate_task`, optionally also
 * `squire_parent_delegation`, etc.) and `--append-system-prompt`. The rest
 * describes Pi's built-in file/shell harness, which Agy does not use.
 */
export function agyRoleInstructions(
	options: { appendSystemPrompt?: string; sections?: Record<string, string> },
	sectionNames: readonly string[] = DEFAULT_SYSTEM_PROMPT_SECTIONS,
): string | undefined {
	const seen = new Set<string>();
	const parts: string[] = [];
	for (const name of sectionNames) {
		if (seen.has(name)) continue;
		seen.add(name);
		const text = options.sections?.[name]?.trim();
		if (text) parts.push(text);
	}
	const appended = options.appendSystemPrompt?.trim();
	if (appended) parts.push(appended);
	return parts.length > 0 ? parts.join("\n\n") : undefined;
}

export interface ApprovedToolResult {
	text: string;
	isError: boolean;
}

export interface AgyNativeProviderHandle {
	provider: Provider<typeof AGY_API>;
	gate: GateServer;
	syncSession(ctx: SessionSyncContext): void;
	expectLiveRequest(signal: AbortSignal | undefined): void;
	/** Lets the Agy tool call behind a Pi approval tool call run, and returns its output. */
	runApprovedTool(toolCallId: string, signal: AbortSignal | undefined): Promise<ApprovedToolResult>;
	abandonParkedTurn(reason?: string): void;
	dispose(): Promise<void>;
}

const errorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

function zeroUsage(): Usage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function defaultContextWindow(modelId: string): number {
	const lowered = modelId.toLowerCase();
	if (lowered.includes("claude")) return 200_000;
	if (lowered.includes("gpt")) return 128_000;
	return 1_048_576;
}

function buildThinkingLevelMap(supported: ReadonlySet<string>): ThinkingLevelMap | undefined {
	if (supported.size === 0) return undefined;
	return {
		off: null,
		minimal: null,
		low: supported.has("low") ? "low" : null,
		medium: supported.has("medium") ? "medium" : null,
		high: supported.has("high") ? "high" : null,
		xhigh: supported.has("xhigh") ? "xhigh" : null,
		max: supported.has("max") ? "max" : null,
	};
}

function inferSupportedEfforts(baseId: string, variants: readonly AcpChoice[], defaultSelect?: AcpChoice[]): Set<string> {
	const found = new Set<string>();
	const prefix = `${baseId}-`;
	for (const variant of variants) {
		if (!variant.value.startsWith(prefix)) continue;
		const suffix = variant.value.slice(prefix.length).toLowerCase();
		if ((EFFORT_LEVELS as readonly string[]).includes(suffix)) {
			found.add(suffix);
		}
	}
	if (found.size > 0) return found;
	// If the model appears verbatim as a variant with no effort suffix (e.g. `gpt-oss-120b-medium`,
	// `gemini-pro-orchestrator`), it does not support configurable thought_level.
	if (variants.some((v) => v.value === baseId)) return found;
	// Fallback when no flattened variants list was returned: use session thought_level choices.
	for (const choice of defaultSelect ?? []) {
		const lower = choice.value.toLowerCase();
		if ((EFFORT_LEVELS as readonly string[]).includes(lower)) found.add(lower);
	}
	return found;
}

function toPiModel(
	id: string,
	name: string,
	thinkingMap: ThinkingLevelMap | undefined,
): Model<typeof AGY_API> {
	return {
		id,
		name,
		api: AGY_API,
		provider: AGY_PROVIDER_ID,
		baseUrl: `acp://${AGY_PROVIDER_ID}`,
		reasoning: thinkingMap !== undefined,
		...(thinkingMap ? { thinkingLevelMap: thinkingMap } : {}),
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: defaultContextWindow(id),
		maxTokens: 65_536,
	};
}

/** Converts an `agy acp` `session/new` config snapshot into Pi `Model` records. */
export function modelsFromSessionConfig(config: AcpSessionConfig): Model<typeof AGY_API>[] {
	const choices = config.model?.choices ?? [];
	const variants = config.availableModelVariants;
	const defaultThoughtChoices = config.thoughtLevel?.choices;
	return choices.map((choice) => {
		const supported = inferSupportedEfforts(choice.value, variants, defaultThoughtChoices);
		const thinkingMap = buildThinkingLevelMap(supported);
		return toPiModel(choice.value, choice.name, thinkingMap);
	});
}

const STANDARD_THINKING = buildThinkingLevelMap(new Set(["low", "medium", "high"]));
const EXTENDED_THINKING = buildThinkingLevelMap(new Set(["low", "medium", "high", "xhigh"]));

export const BASELINE_MODELS: readonly Model<typeof AGY_API>[] = [
	toPiModel("gemini-4-neon", "Gemini 4 Neon", EXTENDED_THINKING),
	toPiModel("gemini-4-carbon", "Gemini 4 Carbon", STANDARD_THINKING),
	toPiModel("gemini-4-argon", "Gemini 4 Argon", STANDARD_THINKING),
	toPiModel("gemini-4-hydrogen", "Gemini 4 Hydrogen", EXTENDED_THINKING),
	toPiModel("gemini-3.8-flash", "Gemini 3.8 Flash", STANDARD_THINKING),
	toPiModel("claude-opus-5-5", "Claude Opus 5.5", STANDARD_THINKING),
	toPiModel("claude-sonnet-5-5", "Claude Sonnet 5.5", STANDARD_THINKING),
	toPiModel("gpt-oss-120b-medium", "GPT-OSS 120B (Medium)", undefined),
];

/**
 * Maps `agy acp` turn usage to Pi's `Usage` shape.
 *
 * When a single ACP `session/prompt` spans multiple internal tool-call steps,
 * `result.usage` is cumulative across all steps while `latestUsageUpdate.used`
 * reflects the actual final context window occupancy. Scaling `input`/`cacheRead`
 * proportionally to `latestUsageUpdate.used` preserves the exact cache-hit ratio
 * while preventing Pi's context-window meter and auto-compaction from over-counting.
 */
export function normalizeTurnUsage(
	model: Model<Api>,
	raw: AcpUsage | undefined,
	latestUsageUpdate: { used: number; size: number } | null,
	stepCount: number,
): Usage {
	if (!raw) return zeroUsage();

	let input = Math.max(0, raw.inputTokens);
	let cacheRead = Math.max(0, raw.cachedReadTokens);
	let cacheWrite = Math.max(0, raw.cachedWriteTokens);
	const output = Math.max(0, raw.outputTokens);
	const reasoning = raw.thoughtTokens > 0 ? Math.min(raw.thoughtTokens, output || raw.thoughtTokens) : undefined;

	if (stepCount > 1 && latestUsageUpdate && latestUsageUpdate.used > 0) {
		const rawPromptTotal = input + cacheRead + cacheWrite;
		const targetContext = Math.max(0, latestUsageUpdate.used - output);
		if (rawPromptTotal > targetContext * 1.2 && rawPromptTotal > 0) {
			const ratio = targetContext / rawPromptTotal;
			input = Math.round(input * ratio);
			cacheRead = Math.round(cacheRead * ratio);
			cacheWrite = Math.round(cacheWrite * ratio);
		}
	}

	const totalTokens = input + output + cacheRead + cacheWrite;
	const usage: Usage = {
		input,
		output,
		cacheRead,
		cacheWrite,
		...(reasoning !== undefined ? { reasoning } : {}),
		totalTokens,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
	usage.cost = calculateCost(model, usage);
	return usage;
}

/** Builds a balanced Pi `AssistantMessageEventStream` writer. */
function createMessageWriter(stream: AssistantMessageEventStream, message: AssistantMessage) {
	let open: { type: "text" | "thinking"; contentIndex: number; content: string } | null = null;
	let started = false;
	let settled = false;

	const ensureStarted = (): void => {
		if (started) return;
		started = true;
		stream.push({ type: "start", partial: message });
	};

	const closeOpenBlock = (): void => {
		if (!open) return;
		const { type, contentIndex, content } = open;
		stream.push({ type: `${type}_end`, contentIndex, content, partial: message });
		open = null;
	};

	const append = (type: "text" | "thinking", delta: string): void => {
		if (!delta || settled) return;
		ensureStarted();
		if (open?.type !== type) {
			closeOpenBlock();
			open = { type, contentIndex: message.content.length, content: "" };
			message.content.push(type === "text" ? { type, text: "" } : { type, thinking: "" });
			stream.push({ type: `${type}_start`, contentIndex: open.contentIndex, partial: message });
		}
		open.content += delta;
		const block = message.content[open.contentIndex];
		if (block?.type === "text") block.text = open.content;
		else if (block?.type === "thinking") block.thinking = open.content;
		stream.push({ type: `${type}_delta`, contentIndex: open.contentIndex, delta, partial: message });
	};

	const emitToolCall = (id: string, name: string, args: Record<string, unknown>): void => {
		if (settled) return;
		ensureStarted();
		closeOpenBlock();
		const contentIndex = message.content.length;
		const toolCall = { type: "toolCall" as const, id, name, arguments: args as JsonObject };
		message.content.push(toolCall);
		stream.push({ type: "toolcall_start", contentIndex, partial: message });
		stream.push({ type: "toolcall_delta", contentIndex, delta: JSON.stringify(args), partial: message });
		stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: message });
	};

	const settle = (push: () => void): void => {
		if (settled) return;
		settled = true;
		closeOpenBlock();
		push();
		stream.end(message);
	};

	const done = (reason: "stop" | "length" | "toolUse"): void =>
		settle(() => {
			ensureStarted();
			message.stopReason = reason;
			stream.push({ type: "done", reason, message });
		});

	const fail = (reason: "error" | "aborted", error: string): void =>
		settle(() => {
			message.stopReason = reason;
			message.errorMessage = error;
			stream.push({ type: "error", reason, error: message });
		});

	return { ensureStarted, append, emitToolCall, done, fail, get isSettled() { return settled; } };
}

type MessageWriter = ReturnType<typeof createMessageWriter>;




/** An Agy tool call waiting in the `pi-gate` hook for Pi's decision. */
interface ParkedHook {
	id: string;
	call: HookToolCall;
	subagent: boolean;
	answered: boolean;
	answer(decision: HookDecision): void;
}

type PhaseEnd =
	| { kind: "parked"; hooks: ParkedHook[]; mcpCalls: ParkedMcpCall[] }
	| { kind: "done"; result: AcpPromptResult }
	| { kind: "error"; error: Error }
	| { kind: "aborted" };

type ToolUpdate = Extract<AcpSessionUpdate, { type: "tool_call" | "tool_call_update" }>;

interface ActiveBridge {
	sessionId: string;
	model: Model<Api>;
	promptController: AbortController;
	/** Bumped by a steering prompt; Agy resolves the earlier prompt at once. */
	promptGeneration: number;
	writer: MessageWriter | null;
	onRawUpdate?: (params: unknown) => void;
	detachSignal: (() => void) | null;
	bufferedUpdates: AcpSessionUpdate[];
	pendingParks: ParkedHook[];
	awaitingResults: Map<string, ParkedHook>;
	pendingMcpParks: ParkedMcpCall[];
	awaitingMcpResults: Map<string, ParkedMcpCall>;
	parkBatchTimer: NodeJS.Timeout | null;
	phaseResolve: ((end: PhaseEnd) => void) | null;
	terminalOutcome: { kind: "done"; result: AcpPromptResult } | { kind: "error"; error: Error } | null;
	latestUsageUpdate: { used: number; size: number } | null;
	stepCount: number;
	aborted: boolean;
	finishedTools: Map<string, ToolUpdate>;
	toolWaiters: Map<string, Array<(update: ToolUpdate | undefined) => void>>;
	warningFilter: WarningLineFilter;
}

interface BoundSession {
	acpSessionId: string;
	lastModelId: string | undefined;
	lastThoughtLevel: string | undefined;
}

function pickThoughtLevel(
	model: Model<Api>,
	thoughtConfig: AcpSessionConfig["thoughtLevel"],
	requested: ModelThinkingLevel | undefined,
): string | undefined {
	if (!thoughtConfig || thoughtConfig.choices.length === 0) return undefined;
	const available = new Set(thoughtConfig.choices.map((c) => c.value.toLowerCase()));
	const findExact = (level: string): string | undefined =>
		thoughtConfig.choices.find((c) => c.value.toLowerCase() === level.toLowerCase())?.value;

	if (requested) {
		const mapped = model.thinkingLevelMap?.[requested];
		if (typeof mapped === "string") {
			const match = findExact(mapped);
			if (match) return match;
		}
		const direct = findExact(requested);
		if (direct) return direct;
	}

	// If Pi passed `off` (`undefined`) or an unsupported tier, pick the lowest available effort.
	for (const fallback of ["low", "medium", "high"]) {
		if (available.has(fallback)) return findExact(fallback);
	}
	return undefined;
}

/**
 * Resolves the `model` config value `agy acp` expects for a Pi model.
 * Depending on account/CLI version, `agy` advertises either bare IDs with a
 * separate `thought_level` select or effort-suffixed IDs such as
 * `gemini-4-hydrogen-high`.
 */
export function resolveAgyModelValue(
	model: Model<Api>,
	reasoning: ModelThinkingLevel | undefined,
	choices: readonly AcpChoice[],
): { value: string; pinsEffort: boolean } | undefined {
	if (choices.some((c) => c.value === model.id)) {
		return { value: model.id, pinsEffort: false };
	}
	const mapped = reasoning ? model.thinkingLevelMap?.[reasoning] : undefined;
	const candidates = [
		typeof mapped === "string" ? mapped : undefined,
		reasoning,
		"medium",
		"high",
		"low",
	].filter((level, index, all): level is string => typeof level === "string" && all.indexOf(level) === index);
	for (const level of candidates) {
		const suffixed = `${model.id}-${level}`;
		const match = choices.find((c) => c.value.toLowerCase() === suffixed.toLowerCase());
		if (match) return { value: match.value, pinsEffort: true };
	}
	return undefined;
}

async function applySessionModelAndThinking(
	transport: AcpTransport,
	sessionId: string,
	model: Model<Api>,
	reasoning: ModelThinkingLevel | undefined,
	force: boolean,
	lastModelId: string | undefined,
	lastThoughtLevel: string | undefined,
	signal?: AbortSignal,
): Promise<{ modelId: string | undefined; thoughtLevel: string | undefined }> {
	let config = transport.sessionConfig(sessionId);
	let appliedModel = lastModelId;
	let appliedThought = lastThoughtLevel;
	let pinsEffort = false;

	const modelSelect = config?.model;
	if (modelSelect?.configId) {
		const resolved = resolveAgyModelValue(model, reasoning, modelSelect.choices);
		if (!resolved) {
			const offered = modelSelect.choices.map((c) => c.value).join(", ");
			throw new Error(`agy acp does not offer model '${model.id}'. Available: ${offered || "none"}.`);
		}
		pinsEffort = resolved.pinsEffort;
		if (force || modelSelect.currentValue !== resolved.value || appliedModel !== resolved.value) {
			config = await transport.setConfigOption(sessionId, modelSelect.configId, resolved.value, signal);
			appliedModel = resolved.value;
		}
	}

	if (pinsEffort) {
		return { modelId: appliedModel, thoughtLevel: undefined };
	}

	const thoughtSelect = config?.thoughtLevel;
	const wantedThought = pickThoughtLevel(model, thoughtSelect, reasoning);
	if (
		thoughtSelect?.configId &&
		wantedThought !== undefined &&
		(force || thoughtSelect.currentValue !== wantedThought || appliedThought !== wantedThought)
	) {
		await transport.setConfigOption(sessionId, thoughtSelect.configId, wantedThought, signal);
		appliedThought = wantedThought;
	}

	return { modelId: appliedModel, thoughtLevel: appliedThought };
}

function finishFromStopReason(writer: MessageWriter, stopReason: AcpPromptResult["stopReason"]): void {
	switch (stopReason) {
		case "end_turn":
			writer.done("stop");
			return;
		case "max_tokens":
		case "max_turn_requests":
			writer.done("length");
			return;
		case "refusal":
			writer.fail("error", "Antigravity refused to continue this turn.");
			return;
		case "cancelled":
			writer.fail("aborted", "Request was aborted");
			return;
		default: {
			const _exhaustive: never = stopReason;
			void _exhaustive;
		}
	}
}


const stringArg = (args: Record<string, unknown>, ...names: string[]): string | undefined => {
	for (const name of names) {
		const value = args[name];
		if (typeof value === "string" && value.length > 0) return value;
	}
	return undefined;
};

/** The Pi approval tool call that stands in for one Agy tool call. */
export function approvalToolCall(
	call: HookToolCall,
	subagent: boolean,
	cwd: string,
): { name: string; arguments: Record<string, unknown> } {
	const marker = subagent ? { subagent: true } : {};
	const path = stringArg(call.args, "Path", "AbsolutePath", "TargetFile");
	switch (call.toolName) {
		case "run_command": {
			const command = stringArg(call.args, "CommandLine");
			if (command) return { name: AGY_COMMAND_TOOL, arguments: { command, cwd: stringArg(call.args, "Cwd") ?? cwd, ...marker } };
			break;
		}
		case "read_file":
		case "view_file":
			// Pi's own read gets the read-only path rules. If Pi cannot run it, the call fails and Agy is denied.
			if (path) return { name: "read", arguments: { path } };
			break;
		case "edit_file":
		case "write_file":
			if (path) return { name: AGY_EDIT_TOOL, arguments: { path, ...marker } };
			break;
		case "search_web": {
			const query = stringArg(call.args, "query", "Query");
			if (query) return { name: AGY_SEARCH_TOOL, arguments: { query, ...marker } };
			break;
		}
		case "invoke_subagent":
			return { name: AGY_SUBAGENT_TOOL, arguments: { arguments: call.args, ...marker } };
		case "send_message":
			return { name: AGY_MESSAGE_TOOL, arguments: { arguments: call.args, ...marker } };
	}
	// A top-level path lets Pi's path rules see the file the call touches.
	return { name: AGY_OTHER_TOOL, arguments: { tool: call.toolName, arguments: call.args, ...(path ? { path } : {}), ...marker } };
}

function toolResultText(result: ToolResultMessage): string {
	return result.content.map((part) => (part.type === "text" ? part.text : "")).join("\n").trim();
}

function renderToolUpdate(update: ToolUpdate | undefined): ApprovedToolResult {
	if (!update) return { text: "Agy reported no output for this tool call.", isError: false };
	const isError = update.status === "failed";
	const raw = update.rawOutput;
	if (isRecord(raw)) {
		const lines: string[] = [];
		if (typeof raw.output === "string") lines.push(raw.output);
		if (typeof raw.exit_code === "number" && raw.exit_code !== 0) lines.push(`(exit code ${raw.exit_code})`);
		if (lines.length > 0) return { text: lines.join("\n"), isError };
	}
	if (Array.isArray(update.content)) {
		const texts = update.content.flatMap((item: unknown) => {
			if (!isRecord(item)) return [];
			if (item.type === "content" && isRecord(item.content) && typeof item.content.text === "string") return [item.content.text];
			if (item.type === "diff" && typeof item.path === "string") return [`Edited ${item.path}`];
			return [];
		});
		if (texts.length > 0) return { text: texts.join("\n"), isError };
	}
	if (typeof raw === "string" && raw) return { text: raw, isError };
	if (raw !== undefined) return { text: JSON.stringify(raw), isError };
	return { text: isError ? "Agy reported that the tool call failed." : "Done.", isError };
}

const userText = (message: Message): string =>
	message.role === "user"
		? typeof message.content === "string"
			? message.content
			: message.content.map((part) => (part.type === "text" ? part.text : "")).join("")
		: "";

/** Drops what Agy must not be sent: the results of Agy's own approval tool calls. */
function forwardable(messages: readonly Message[], all: readonly Message[]): Message[] {
	const agyCallIds = new Set(
		all.flatMap((message) =>
			message.role === "assistant" && message.provider === AGY_PROVIDER_ID
				? message.content.flatMap((part) => (part.type === "toolCall" ? [part.id] : []))
				: [],
		),
	);
	return messages.filter((message) => {
		if (message.role === "system") return false;
		if (message.role === "assistant") return true;
		if (message.role === "toolResult") return !agyCallIds.has(message.toolCallId);
		return userText(message).trim().length > 0 || (typeof message.content !== "string" && message.content.length > 0);
	});
}

/** Pi's own transcript size, so Pi's context meter and compaction ignore Agy's separate history. */
function piTranscriptTokens(context: TranscriptContext, message: AssistantMessage): number {
	let total = estimateTokens(message);
	for (const entry of context.messages) if (entry.role !== "system") total += estimateTokens(entry);
	return total;
}

export function createAgyNativeProvider(options: AgyNativeProviderOptions): AgyNativeProviderHandle {
	const log = options.log ?? (() => {});
	const parkBatchMs = options.parkBatchMs ?? DEFAULT_PARK_BATCH_MS;
	const hookDeadlineMs = options.hookDeadlineMs ?? DEFAULT_HOOK_DEADLINE_MS;
	const conversationStore = options.conversationStore ?? defaultConversationStore;
	const mcpPatterns = options.mcpToolPatterns ?? [];

	let models: Model<typeof AGY_API>[] = [...BASELINE_MODELS];
	let activePiSessionId: string | undefined;
	let pendingLiveSignal: AbortSignal | undefined;
	let activeCwd = process.cwd();
	let boundSession: BoundSession | null = null;
	let activeBridge: ActiveBridge | null = null;
	let transport: AcpTransport | null = null;
	let mcpServer: McpBridgeServer | null = options.mcpServer ?? null;
	let hookSeq = 0;
	/** Agy tool calls Pi approved; Agy's own permission prompt for them is answered with allow. */
	const approvedToolCalls = new Set<string>();
	/** Agy sessions running a side request; their tool calls are denied. */
	const sideSessions = new Set<string>();

	const checkAvailable = (): boolean =>
		options.checkAvailable ? options.checkAvailable() : resolveAgyBinary(options.binaryPath) !== undefined;

	const deny = (reason: string): HookDecision => ({ decision: "deny", reason });

	const onGateRequest = ({ call, closed }: GateRequest): Promise<HookDecision> => {
		if (call.conversationId !== undefined && sideSessions.has(call.conversationId)) {
			log(`hook ${call.toolName} denied: side request`);
			return Promise.resolve(deny("Tools are not available for this request."));
		}
		const bridge = activeBridge;
		if (!bridge || bridge.aborted) {
			log(`hook ${call.toolName} denied: no Pi turn running`);
			return Promise.resolve(deny("No Pi turn is running, so Pi cannot approve this tool call."));
		}
		if (mcpServer && isOwnMcpToolName(call.toolName)) {
			log(`hook ${call.toolName} -> auto_approve (own MCP tool)`);
			return Promise.resolve({ decision: "auto_approve" });
		}
		return new Promise<HookDecision>((resolvePromise) => {
			const park: ParkedHook = {
				id: call.toolCallId ?? `agy-hook-${++hookSeq}`,
				call,
				subagent: call.conversationId !== undefined && call.conversationId !== bridge.sessionId,
				answered: false,
				answer(decision) {
					if (park.answered) return;
					park.answered = true;
					clearTimeout(timer);
					if (decision.decision === "auto_approve") approvedToolCalls.add(park.id);
					log(`hook ${call.toolName} ${park.id} -> ${decision.decision}`);
					resolvePromise(decision);
				},
			};
			const timer = setTimeout(
				() => park.answer(deny(`Pi did not answer within ${Math.round(hookDeadlineMs / 60_000)} minutes.`)),
				hookDeadlineMs,
			);
			timer.unref();
			void closed.then(() => {
				if (park.answered) return;
				park.answered = true;
				clearTimeout(timer);
				bridge.pendingParks = bridge.pendingParks.filter((pending) => pending !== park);
			});
			bridge.pendingParks.push(park);
			if (!bridge.parkBatchTimer) {
				bridge.parkBatchTimer = setTimeout(() => flushPendingParks(bridge), parkBatchMs);
				bridge.parkBatchTimer.unref();
			}
		});
	};

	const gate = new GateServer(onGateRequest);

	const onMcpPark = (call: ParkedMcpCall): boolean => {
		const bridge = activeBridge;
		if (!bridge || bridge.aborted) {
			log(`mcp ${call.piName} rejected: no Pi turn running`);
			return false;
		}
		log(`mcp park ${call.piName} ${call.callId}`);
		bridge.pendingMcpParks.push(call);
		if (!bridge.parkBatchTimer) {
			bridge.parkBatchTimer = setTimeout(() => flushPendingParks(bridge), parkBatchMs);
			bridge.parkBatchTimer.unref();
		}
		return true;
	};

	const onMcpCancel = (callId: string): void => {
		const bridge = activeBridge;
		if (!bridge) return;
		bridge.pendingMcpParks = bridge.pendingMcpParks.filter((call) => call.callId !== callId);
		bridge.awaitingMcpResults.delete(callId);
	};

	const ensureMcpServers = async (): Promise<AcpMcpServerHttp[]> => {
		if (mcpPatterns.length === 0) return [];
		const selected = selectMcpTools(options.getAvailableTools?.() ?? [], mcpPatterns);
		const projected = projectPiTools(selected);
		if (projected.length === 0 && !mcpServer) return [];
		mcpServer ??= new McpBridgeServer();
		mcpServer.setParkHandler(onMcpPark, onMcpCancel);
		await mcpServer.start();
		await mcpServer.updateTools(projected);
		return [
			{
				type: "http",
				name: MCP_SERVER_NAME,
				url: mcpServer.url,
				headers: [{ name: MCP_TOKEN_HEADER, value: mcpServer.authToken }],
			},
		];
	};

	const ensureTransport = async (): Promise<AcpTransport> => {
		if (transport) return transport;
		try {
			const swept = conversationStore.sweepStaleForks?.(options.geminiDir) ?? 0;
			if (swept > 0) log(`swept ${swept} stale fork/checkpoint artifact(s)`);
		} catch (error) {
			log(`fork sweep skipped: ${errorMessage(error)}`);
		}
		const useAgyPermissions = options.permissions === "agy";
		const hooksPath = join(options.geminiDir, "config", "hooks.json");
		let env: Record<string, string> = {};
		const allowPiGatePermission = (toolCallId: string | undefined) =>
			toolCallId !== undefined && approvedToolCalls.has(toolCallId);
		if (useAgyPermissions) {
			(options.uninstallGate ?? (() => uninstallHook(hooksPath)))();
		} else {
			(options.installGate ?? (() => installHook({
				hooksPath,
				scriptPath: join(options.cacheDir, "pi-gate-hook.mjs"),
			})))();
			const url = await gate.start();
			env = { [GATE_URL_ENV]: url };
		}
		const handleAgyPermission = async (request: AcpPermissionRequest): Promise<boolean | string> => {
			if (request.sessionId !== undefined && sideSessions.has(request.sessionId)) return false;
			const current = activeBridge;
			if (!current || (request.sessionId && request.sessionId !== current.sessionId)) return false;
			if (!options.onPermissionRequest) return false;
			return await options.onPermissionRequest(request, current.promptController.signal);
		};
		transport = options.createTransport
			? options.createTransport(env, allowPiGatePermission)
			: new AgyAcpClient({
					binaryPath: options.binaryPath,
					geminiDir: options.geminiDir,
					sandbox: options.sandbox,
					env,
					onPermissionRequest: useAgyPermissions
						? handleAgyPermission
						: (request) => allowPiGatePermission(request.toolCallId),
					log: (msg) => log(`acp: ${msg}`),
				});
		return transport;
	};

	const takeLiveRequest = (streamOptions: StreamOptions | undefined): boolean => {
		// Pi 1.6 dist/core/sdk.js installs this observer on the agent loop.
		// A nested streamSimple call can share both ctx.signal and sessionId.
		if (
			!pendingLiveSignal ||
			!activePiSessionId ||
			streamOptions?.signal !== pendingLiveSignal ||
			streamOptions.sessionId !== activePiSessionId ||
			!streamOptions.onProviderStreamEvent ||
			streamOptions.disableContinuation
		) {
			return false;
		}
		pendingLiveSignal = undefined;
		return true;
	};

	const releaseToolWaiters = (bridge: ActiveBridge): void => {
		for (const waiters of bridge.toolWaiters.values()) for (const wake of waiters) wake(undefined);
		bridge.toolWaiters.clear();
	};

	const clearBridge = (bridge: ActiveBridge, cancelRemote: boolean, reason: string): void => {
		if (bridge.parkBatchTimer) {
			clearTimeout(bridge.parkBatchTimer);
			bridge.parkBatchTimer = null;
		}
		bridge.detachSignal?.();
		bridge.detachSignal = null;
		bridge.writer = null;
		bridge.onRawUpdate = undefined;
		releaseToolWaiters(bridge);

		if (cancelRemote) {
			bridge.aborted = true;
			for (const park of bridge.pendingParks) park.answer(deny(reason));
			bridge.pendingParks = [];
			for (const park of bridge.awaitingResults.values()) park.answer(deny(reason));
			bridge.awaitingResults.clear();
			for (const mcpCall of bridge.pendingMcpParks) mcpCall.reject(reason);
			bridge.pendingMcpParks = [];
			for (const mcpCall of bridge.awaitingMcpResults.values()) mcpCall.reject(reason);
			bridge.awaitingMcpResults.clear();
			bridge.promptController.abort();
		}

		if (activeBridge === bridge) {
			activeBridge = null;
			transport?.setUpdateListener(bridge.sessionId, undefined);
		}
	};

	const abandonParkedTurn = (reason = "Pi ended the turn before deciding on this tool call."): void => {
		pendingLiveSignal = undefined;
		if (!activeBridge) return;
		const bridge = activeBridge;
		const resolve = bridge.phaseResolve;
		bridge.phaseResolve = null;
		clearBridge(bridge, true, reason);
		resolve?.({ kind: "aborted" });
	};

	const failBridgeRateLimit = (bridge: ActiveBridge, reason: string): void => {
		const outcome: PhaseEnd = { kind: "error", error: new Error(reason) };
		const resolve = bridge.phaseResolve;
		bridge.phaseResolve = null;
		if (!resolve) bridge.terminalOutcome = outcome;
		clearBridge(bridge, true, reason);
		resolve?.(outcome);
	};

	const routeUpdate = (bridge: ActiveBridge, update: AcpSessionUpdate): void => {
		if (update.type === "usage_update") {
			bridge.latestUsageUpdate = { used: update.used, size: update.size };
			return;
		}
		if (update.type === "tool_call" || update.type === "tool_call_update") {
			if (update.status !== "completed" && update.status !== "failed") return;
			bridge.finishedTools.set(update.toolCallId, update);
			const waiters = bridge.toolWaiters.get(update.toolCallId);
			bridge.toolWaiters.delete(update.toolCallId);
			for (const wake of waiters ?? []) wake(update);
			return;
		}
		if (update.type !== "agent_message_chunk" && update.type !== "agent_thought_chunk") return;
		if (!bridge.writer || bridge.writer.isSettled) {
			bridge.bufferedUpdates.push(update);
			return;
		}
		const writer = bridge.writer;
		const kind = update.type === "agent_message_chunk" ? "text" : "thinking";
		const quotaError = bridge.warningFilter.push(kind, update.text, (streamKind, text) => writer.append(streamKind, text));
		if (quotaError) failBridgeRateLimit(bridge, quotaError);
	};

	const flushBufferedUpdates = (bridge: ActiveBridge): void => {
		if (!bridge.writer || bridge.bufferedUpdates.length === 0) return;
		const queued = bridge.bufferedUpdates;
		bridge.bufferedUpdates = [];
		for (const update of queued) routeUpdate(bridge, update);
	};

	const attachSignalToBridge = (bridge: ActiveBridge, signal: AbortSignal | undefined): void => {
		bridge.detachSignal?.();
		bridge.detachSignal = null;
		if (!signal) return;
		const onAbort = () => {
			const resolve = bridge.phaseResolve;
			bridge.phaseResolve = null;
			clearBridge(bridge, true, "Request was aborted");
			resolve?.({ kind: "aborted" });
		};
		if (signal.aborted) {
			onAbort();
			return;
		}
		signal.addEventListener("abort", onAbort, { once: true });
		bridge.detachSignal = () => signal.removeEventListener("abort", onAbort);
	};

	const takePendingParks = (bridge: ActiveBridge): ParkedHook[] => {
		const hooks = bridge.pendingParks.filter((park) => !park.answered);
		bridge.pendingParks = [];
		for (const park of hooks) bridge.awaitingResults.set(park.id, park);
		return hooks;
	};

	const takePendingMcpParks = (bridge: ActiveBridge): ParkedMcpCall[] => {
		const calls = bridge.pendingMcpParks;
		bridge.pendingMcpParks = [];
		for (const call of calls) bridge.awaitingMcpResults.set(call.callId, call);
		return calls;
	};

	const flushPendingParks = (bridge: ActiveBridge): void => {
		if (bridge.parkBatchTimer) {
			clearTimeout(bridge.parkBatchTimer);
			bridge.parkBatchTimer = null;
		}
		if ((bridge.pendingParks.length === 0 && bridge.pendingMcpParks.length === 0) || !bridge.phaseResolve) return;
		const resolve = bridge.phaseResolve;
		bridge.phaseResolve = null;
		resolve({ kind: "parked", hooks: takePendingParks(bridge), mcpCalls: takePendingMcpParks(bridge) });
	};

	const awaitBridgePhase = (bridge: ActiveBridge): Promise<PhaseEnd> => {
		if (bridge.terminalOutcome) {
			const outcome = bridge.terminalOutcome;
			bridge.terminalOutcome = null;
			return Promise.resolve(outcome);
		}
		if (bridge.aborted) return Promise.resolve({ kind: "aborted" });
		if (bridge.pendingParks.some((park) => !park.answered) || bridge.pendingMcpParks.length > 0) {
			return Promise.resolve({ kind: "parked", hooks: takePendingParks(bridge), mcpCalls: takePendingMcpParks(bridge) });
		}
		return new Promise<PhaseEnd>((resolve) => {
			bridge.phaseResolve = resolve;
		});
	};

	const sendPrompt = (bridge: ActiveBridge, prompt: AcpPromptBlock[], steer: boolean): void => {
		const generation = ++bridge.promptGeneration;
		const active = transport;
		if (!active) return;
		void active
			.prompt({ sessionId: bridge.sessionId, prompt, signal: bridge.promptController.signal, steer })
			.then(
				(result): PhaseEnd => {
					log(`raw ACP usage sid=${bridge.sessionId}: ${JSON.stringify(result.usage ?? null)}`);
					return { kind: "done", result };
				},
				(error: unknown): PhaseEnd => ({ kind: "error", error: error instanceof Error ? error : new Error(String(error)) }),
			)
			.then((outcome) => {
				if (bridge.aborted || generation !== bridge.promptGeneration || outcome.kind === "parked" || outcome.kind === "aborted") return;
				releaseToolWaiters(bridge);
				const resolve = bridge.phaseResolve;
				bridge.phaseResolve = null;
				if (resolve) resolve(outcome);
				else bridge.terminalOutcome = outcome;
			});
	};

	const newMessage = (model: Model<Api>): AssistantMessage => ({
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: zeroUsage(),
		stopReason: "pending",
		timestamp: Date.now(),
	});

	const settlePhase = (
		bridge: ActiveBridge,
		phase: PhaseEnd,
		writer: MessageWriter,
		message: AssistantMessage,
		context: TranscriptContext,
	): void => {
		bridge.detachSignal?.();
		bridge.detachSignal = null;
		bridge.writer = null;
		bridge.onRawUpdate = undefined;
		message.usage.totalTokens = piTranscriptTokens(context, message);

		if (phase.kind === "parked") {
			bridge.warningFilter.flush((kind, text) => writer.append(kind, text));
			bridge.stepCount += 1;
			for (const park of phase.hooks) {
				const call = approvalToolCall(park.call, park.subagent, activeCwd);
				writer.emitToolCall(park.id, call.name, call.arguments);
			}
			for (const mcpCall of phase.mcpCalls) {
				writer.emitToolCall(mcpCall.callId, mcpCall.piName, mcpCall.args);
			}
			writer.done("toolUse");
			return;
		}

		clearBridge(bridge, false, "");

		if (phase.kind === "aborted") {
			writer.fail("aborted", "Request was aborted");
			return;
		}
		if (phase.kind === "error") {
			writer.fail("error", errorMessage(phase.error));
			return;
		}

		const { result } = phase;
		const heldQuotaError = bridge.warningFilter.finish((kind, text) => writer.append(kind, text));
		message.usage = normalizeTurnUsage(bridge.model, result.usage, bridge.latestUsageUpdate, bridge.stepCount);
		message.usage.totalTokens = piTranscriptTokens(context, message);
		if (heldQuotaError) {
			writer.fail("error", heldQuotaError);
			return;
		}
		const terminalError = terminalAcpError(message, result) ?? blankAcpTurnError(message, result, bridge.stepCount);
		if (terminalError) {
			writer.fail("error", terminalError);
			return;
		}
		if (result.stopReason === "end_turn") {
			try {
				conversationStore.recordCheckpoint(options.geminiDir, bridge.sessionId, message.timestamp);
			} catch (error) {
				log(`checkpoint skipped sid=${bridge.sessionId}: ${errorMessage(error)}`);
			}
		}
		finishFromStopReason(writer, result.stopReason);
	};

	/** Answers parked hooks from Pi's tool results; returns false if the results do not match. */
	const resumeParkedTurn = async (
		bridge: ActiveBridge,
		writer: MessageWriter,
		message: AssistantMessage,
		context: TranscriptContext,
		model: Model<Api>,
		streamOptions: StreamOptions | undefined,
		onRawUpdate: ((raw: unknown) => void) | undefined,
	): Promise<boolean> => {
		const tail = extractPostAssistantTail(context.messages);
		const resultById = new Map((tail?.toolResults ?? []).map((result) => [result.toolCallId, result]));
		if (
			!tail ||
			bridge.model.id !== model.id ||
			![...bridge.awaitingResults.keys()].every((id) => resultById.has(id)) ||
			![...bridge.awaitingMcpResults.keys()].every((id) => resultById.has(id))
		) {
			return false;
		}
		message.responseId = `${SESSION_RESPONSE_PREFIX}${bridge.sessionId}`;
		writer.ensureStarted();
		bridge.writer = writer;
		bridge.onRawUpdate = onRawUpdate;
		attachSignalToBridge(bridge, streamOptions?.signal);

		const hooks = [...bridge.awaitingResults.values()];
		bridge.awaitingResults.clear();
		for (const park of hooks) {
			const result = resultById.get(park.id)!;
			if (result.isError) park.answer(deny(toolResultText(result) || "Pi blocked this tool call."));
			else park.answer({ decision: "auto_approve" });
		}
		const mcpCalls = [...bridge.awaitingMcpResults.values()];
		bridge.awaitingMcpResults.clear();
		for (const mcpCall of mcpCalls) {
			const result = resultById.get(mcpCall.callId)!;
			mcpCall.resolve({ content: result.content, isError: result.isError });
		}
		if (tail.steeringTexts.length > 0 || tail.steeringImages.length > 0) {
			const steering: AcpPromptBlock[] = [
				...tail.steeringTexts.map((text): AcpPromptBlock => ({ type: "text", text })),
				...tail.steeringImages.map((image): AcpPromptBlock => ({ type: "image", data: image.data, mimeType: image.mimeType })),
			];
			log(`steering prompt sid=${bridge.sessionId}`);
			sendPrompt(bridge, steering, true);
		}
		flushBufferedUpdates(bridge);
		const phase = await awaitBridgePhase(bridge);
		settlePhase(bridge, phase, writer, message, context);
		return true;
	};

	async function runTurn(
		stream: AssistantMessageEventStream,
		model: Model<Api>,
		context: TranscriptContext,
		streamOptions: StreamOptions | undefined,
	): Promise<void> {
		const signal = streamOptions?.signal;
		const message = newMessage(model);
		const writer = createMessageWriter(stream, message);
		const onRawUpdate = streamOptions?.onProviderStreamEvent
			? (raw: unknown) => {
					void Promise.resolve(streamOptions.onProviderStreamEvent?.(raw, model)).catch((err) =>
						log(`onProviderStreamEvent failed: ${errorMessage(err)}`),
					);
				}
			: undefined;

		if (activeBridge) {
			const bridge = activeBridge;
			if (
				(bridge.awaitingResults.size > 0 || bridge.awaitingMcpResults.size > 0) &&
				(await resumeParkedTurn(bridge, writer, message, context, model, streamOptions, onRawUpdate))
			) {
				return;
			}
			abandonParkedTurn("Pi transcript changed before deciding on this tool call.");
		}

		const conversation = context.messages.filter((entry) => entry.role !== "system");

		let active: AcpTransport;
		let mcpServers: AcpMcpServerHttp[];
		try {
			active = await ensureTransport();
			mcpServers = await ensureMcpServers();
		} catch (error) {
			writer.fail("error", errorMessage(error));
			return;
		}

		const view = options.getSessionView?.() ?? viewFromMessages(context.messages);
		const mark = view.branchMark;
		const continuable = mark !== undefined && !view.retired.has(mark.acpSessionId) &&
			view.latestTimestamp(mark.acpSessionId) === mark.timestamp;
		let session: BoundSession | null = null;
		let needApplySettings = false;
		let warm = false;

		try {
			if (continuable) {
				if (boundSession?.acpSessionId === mark.acpSessionId) {
					session = boundSession;
					warm = true;
				} else {
					try {
						await active.resumeSession({
							sessionId: mark.acpSessionId,
							cwd: activeCwd,
							systemPromptAppend: options.getSystemPromptAppend?.(),
							mcpServers,
							signal,
						});
						session = { acpSessionId: mark.acpSessionId, lastModelId: undefined, lastThoughtLevel: undefined };
						// agy resets the model on session/resume.
						needApplySettings = true;
						warm = true;
					} catch (error) {
						if (signal?.aborted) throw error;
						log(`session/resume failed sid=${mark.acpSessionId}: ${errorMessage(error)}`);
					}
				}
			} else if (mark !== undefined) {
				let forkId: string | undefined;
				try {
					forkId = conversationStore.forkConversation(options.geminiDir, mark.acpSessionId, mark.timestamp);
				} catch (error) {
					log(`native fork unavailable sid=${mark.acpSessionId}: ${errorMessage(error)}`);
				}
				if (forkId) {
					if (boundSession) {
						await active.closeSession(boundSession.acpSessionId);
						boundSession = null;
					}
					try {
						await active.resumeSession({
							sessionId: forkId,
							cwd: activeCwd,
							systemPromptAppend: options.getSystemPromptAppend?.(),
							mcpServers,
							signal,
						});
						log(`session/fork-resume source=${mark.acpSessionId} fork=${forkId}`);
						session = { acpSessionId: forkId, lastModelId: undefined, lastThoughtLevel: undefined };
						needApplySettings = true;
						warm = true;
					} catch (error) {
						conversationStore.deleteConversation(options.geminiDir, forkId);
						if (signal?.aborted) throw error;
						log(`session/fork-resume failed fork=${forkId}: ${errorMessage(error)}`);
					}
				}
			}
			if (!session) {
				if (boundSession) await active.closeSession(boundSession.acpSessionId);
				boundSession = null;
				const created = await active.createSession({
					cwd: activeCwd,
					systemPromptAppend: options.getSystemPromptAppend?.(),
					mcpServers,
					signal,
				});
				log(`session/new sid=${created.sessionId}`);
				session = { acpSessionId: created.sessionId, lastModelId: undefined, lastThoughtLevel: undefined };
				needApplySettings = true;
			}
			boundSession = session;
			const applied = await applySessionModelAndThinking(
				active,
				session.acpSessionId,
				model,
				(streamOptions as SimpleStreamOptions | undefined)?.reasoning,
				needApplySettings,
				session.lastModelId,
				session.lastThoughtLevel,
				signal,
			);
			session.lastModelId = applied.modelId;
			session.lastThoughtLevel = applied.thoughtLevel;
		} catch (error) {
			if (signal?.aborted) writer.fail("aborted", "Request was aborted");
			else writer.fail("error", errorMessage(error));
			return;
		}

		const sessionId = session.acpSessionId;
		const unseen = warm && mark ? conversation.filter((entry) => entry.timestamp > mark.timestamp) : conversation;
		const toSend = forwardable(unseen, conversation);
		const replayed = !warm && toSend.some((entry) => entry.role === "assistant");
		const latest = toSend.at(-1);
		const promptBlocks = !warm
			? renderTranscriptReplay(toSend)
			: toSend.length === 1 && latest?.role === "user"
				? userMessageToPromptBlocks(latest)
				: renderTranscriptDelta(toSend);
		if (!promptBlocks || promptBlocks.length === 0) {
			writer.fail("error", "No prompt content found to send to Antigravity.");
			return;
		}
		if (replayed) options.notify?.("Agy started a new session and was sent the conversation kept in Pi.");

		let finalPrompt: AcpPromptBlock[] = promptBlocks;
		try {
			const rewritten = (await streamOptions?.onPayload?.({ sessionId, prompt: promptBlocks }, model)) as
				| { prompt?: AcpPromptBlock[] }
				| undefined;
			if (rewritten?.prompt) finalPrompt = rewritten.prompt;
		} catch (error) {
			writer.fail("error", errorMessage(error));
			return;
		}

		message.responseId = `${SESSION_RESPONSE_PREFIX}${sessionId}`;
		writer.ensureStarted();

		const bridge: ActiveBridge = {
			sessionId,
			model,
			promptController: new AbortController(),
			promptGeneration: 0,
			writer,
			onRawUpdate,
			detachSignal: null,
			bufferedUpdates: [],
			pendingParks: [],
			awaitingResults: new Map(),
			pendingMcpParks: [],
			awaitingMcpResults: new Map(),
			parkBatchTimer: null,
			phaseResolve: null,
			terminalOutcome: null,
			latestUsageUpdate: null,
			stepCount: 1,
			aborted: false,
			finishedTools: new Map(),
			toolWaiters: new Map(),
			warningFilter: new WarningLineFilter(),
		};
		activeBridge = bridge;
		active.setUpdateListener(sessionId, (update, raw) => {
			bridge.onRawUpdate?.(raw);
			routeUpdate(bridge, update);
		});
		attachSignalToBridge(bridge, signal);
		const phasePromise = awaitBridgePhase(bridge);
		log(`prompt sid=${sessionId} warm=${warm} blocks=${finalPrompt.length}`);
		sendPrompt(bridge, finalPrompt, false);
		const phase = await phasePromise;
		settlePhase(bridge, phase, writer, message, context);
	}

	/** Pi requests outside the conversation (summaries, handoffs) run on their own Agy session, closed afterwards. */
	async function runSideRequest(
		stream: AssistantMessageEventStream,
		model: Model<Api>,
		context: TranscriptContext,
		streamOptions: StreamOptions | undefined,
	): Promise<void> {
		const signal = streamOptions?.signal;
		const message = newMessage(model);
		const writer = createMessageWriter(stream, message);
		const replay = renderTranscriptReplay(context.messages);
		if (!replay) {
			writer.fail("error", "No prompt content found to send to Antigravity.");
			return;
		}
		// Stock Agy takes no system prompt, so the request's instructions go first in the prompt.
		const instructions = getCurrentSystemPrompt(context.messages);
		const promptBlocks: AcpPromptBlock[] = instructions
			? [{ type: "text", text: `<instructions>\n${instructions}\n</instructions>` }, ...replay]
			: replay;
		let active: AcpTransport | undefined;
		let sessionId: string | undefined;
		try {
			active = await ensureTransport();
			const created = await active.createSession({ cwd: activeCwd, signal });
			sessionId = created.sessionId;
			sideSessions.add(sessionId);
			log(`side request session/new sid=${sessionId}`);
			await applySessionModelAndThinking(
				active,
				sessionId,
				model,
				(streamOptions as SimpleStreamOptions | undefined)?.reasoning,
				true,
				undefined,
				undefined,
				signal,
			);
			const rewritten = (await streamOptions?.onPayload?.({ sessionId, prompt: promptBlocks }, model)) as
				| { prompt?: AcpPromptBlock[] }
				| undefined;
			writer.ensureStarted();
			let latestUsageUpdate: { used: number; size: number } | null = null;
			const warningFilter = new WarningLineFilter();
			const sideController = new AbortController();
			const onSignalAbort = () => sideController.abort();
			signal?.addEventListener("abort", onSignalAbort, { once: true });
			if (signal?.aborted) sideController.abort();
			let sideQuotaError: string | undefined;
			try {
				active.setUpdateListener(sessionId, (update, raw) => {
					void Promise.resolve(streamOptions?.onProviderStreamEvent?.(raw, model)).catch((err) =>
						log(`onProviderStreamEvent failed: ${errorMessage(err)}`),
					);
					if (update.type === "usage_update") {
						latestUsageUpdate = { used: update.used, size: update.size };
					} else if (update.type === "agent_message_chunk" || update.type === "agent_thought_chunk") {
						const kind = update.type === "agent_message_chunk" ? "text" : "thinking";
						const quotaError = warningFilter.push(kind, update.text, (streamKind, text) => writer.append(streamKind, text));
						if (quotaError && !sideQuotaError) {
							sideQuotaError = quotaError;
							sideController.abort();
						}
					}
				});
				const result = await active.prompt({ sessionId, prompt: rewritten?.prompt ?? promptBlocks, signal: sideController.signal });
				log(`side request raw ACP usage sid=${sessionId}: ${JSON.stringify(result.usage ?? null)}`);
				if (sideQuotaError) {
					writer.fail("error", sideQuotaError);
					return;
				}
				const heldQuotaError = warningFilter.finish((kind, text) => writer.append(kind, text));
				message.usage = normalizeTurnUsage(model, result.usage, latestUsageUpdate, 1);
				if (heldQuotaError) {
					writer.fail("error", heldQuotaError);
					return;
				}
				const terminalError = terminalAcpError(message, result) ?? blankAcpTurnError(message, result, 1);
				if (terminalError) writer.fail("error", terminalError);
				else finishFromStopReason(writer, result.stopReason);
			} finally {
				signal?.removeEventListener("abort", onSignalAbort);
			}
		} catch (error) {
			if (signal?.aborted) writer.fail("aborted", "Request was aborted");
			else writer.fail("error", errorMessage(error));
		} finally {
			if (active && sessionId) {
				active.setUpdateListener(sessionId, undefined);
				sideSessions.delete(sessionId);
				await active.closeSession(sessionId);
			}
		}
	}

	const stream = (
		model: Model<typeof AGY_API>,
		context: TranscriptContext,
		streamOptions?: StreamOptions,
	): AssistantMessageEventStream => {
		const events = createAssistantMessageEventStream();
		const live = takeLiveRequest(streamOptions);
		log(`stream start live=${live} model=${model.id} msgs=${context.messages.length}`);
		(live ? runTurn : runSideRequest)(events, model, context, streamOptions).catch((error: unknown) =>
			log(`stream crashed: ${errorMessage(error)}`),
		);
		return events;
	};

	const runApprovedTool = async (toolCallId: string, signal: AbortSignal | undefined): Promise<ApprovedToolResult> => {
		const bridge = activeBridge;
		const park = bridge?.awaitingResults.get(toolCallId);
		if (!bridge || !park || park.answered) {
			return { text: "Agy is no longer waiting for this tool call.", isError: true };
		}
		park.answer({ decision: "auto_approve" });
		if (park.subagent) return { text: "Approved. Pi cannot see the output of Agy subagent tool calls.", isError: false };
		const finished = bridge.finishedTools.get(toolCallId);
		if (finished) return renderToolUpdate(finished);
		const update = await new Promise<ToolUpdate | undefined>((resolvePromise) => {
			const waiters = bridge.toolWaiters.get(toolCallId) ?? [];
			const onAbort = () => resolvePromise(undefined);
			signal?.addEventListener("abort", onAbort, { once: true });
			waiters.push((value) => {
				signal?.removeEventListener("abort", onAbort);
				resolvePromise(value);
			});
			bridge.toolWaiters.set(toolCallId, waiters);
		});
		if (signal?.aborted) return { text: "Aborted while Agy ran this tool call.", isError: true };
		return renderToolUpdate(update);
	};

	const refreshModels = async (context: RefreshModelsContext): Promise<void> => {
		const storedModels = (context.stored?.models ?? []).filter(
			(m): m is Model<typeof AGY_API> => m.provider === AGY_PROVIDER_ID && m.api === AGY_API,
		);
		if (storedModels.length > 0) {
			await context.publish({
				update: () => {
					models = storedModels;
				},
			});
		}

		if (!context.allowNetwork || context.signal.aborted || !checkAvailable()) return;

		let probeSessionId: string | undefined;
		let active: AcpTransport | undefined;
		try {
			active = await ensureTransport();
			const created = await active.createSession({ cwd: activeCwd, signal: context.signal });
			probeSessionId = created.sessionId;
			const discovered = modelsFromSessionConfig(created.config);
			if (discovered.length > 0) {
				await context.publish({
					persist: { models: discovered, checkedAt: Date.now() },
					update: () => {
						models = discovered;
					},
				});
			}
		} catch (error) {
			log(`model discovery failed: ${errorMessage(error)}`);
		} finally {
			if (probeSessionId) await active?.closeSession(probeSessionId);
		}
	};

	const provider: Provider<typeof AGY_API> = {
		id: AGY_PROVIDER_ID,
		name: "Antigravity",
		baseUrl: `acp://${AGY_PROVIDER_ID}`,
		auth: {
			apiKey: {
				name: "Antigravity CLI (agy acp)",
				check: async () => (checkAvailable() ? { type: "api_key" as const, source: "Antigravity CLI" } : undefined),
				resolve: async () => (checkAvailable() ? { auth: {}, source: "Antigravity CLI" } : undefined),
			},
		},
		getModels: () => models,
		refreshModels,
		stream,
		streamSimple: stream,
	};

	return {
		provider,
		gate,
		syncSession(ctx: SessionSyncContext): void {
			abandonParkedTurn("Pi switched session or branch.");
			activePiSessionId = ctx.piSessionId;
			activeCwd = ctx.cwd;
		},
		expectLiveRequest(signal: AbortSignal | undefined): void {
			pendingLiveSignal = signal;
		},
		runApprovedTool,
		abandonParkedTurn,
		async dispose(): Promise<void> {
			abandonParkedTurn("Pi session shutting down.");
			if (boundSession && transport) await transport.closeSession(boundSession.acpSessionId);
			boundSession = null;
			transport?.dispose();
			transport = null;
			mcpServer?.setParkHandler(undefined);
			await mcpServer?.stop();
			mcpServer = null;
			await gate.stop();
		},
	};
}
