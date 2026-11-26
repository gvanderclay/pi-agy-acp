/**
 * Native Pi 1.6+ `Provider` (`agy-acp`) backed by Antigravity CLI ACP (`agy acp`)
 * with full Pi tool execution over an ephemeral in-memory Streamable HTTP MCP bridge.
 */

import {
	calculateCost,
	createAssistantMessageEventStream,
	getCurrentTools,
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type ImageContent,
	type JsonObject,
	type Model,
	type ModelThinkingLevel,
	type Provider,
	type RefreshModelsContext,
	type SimpleStreamOptions,
	type StreamOptions,
	type TextContent,
	type ThinkingLevelMap,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";
import { AgyAcpClient, resolveAgyBinary, type AcpTransport } from "./acp-client.ts";
import {
	type AcpChoice,
	type AcpPromptBlock,
	type AcpPromptResult,
	type AcpSessionConfig,
	type AcpSessionUpdate,
	type AcpUsage,
} from "./acp-wire.ts";
import { McpBridgeServer, type ParkedToolCall } from "./mcp-server.ts";
import {
	extractPostAssistantTail,
	fingerprintMessages,
	hashSystemPromptAppend,
	messagesAfterFingerprint,
	renderSystemPromptAppend,
	renderTranscriptDelta,
	renderTranscriptReplay,
} from "./prompt.ts";
import { projectPiTools } from "./tools.ts";

export const AGY_ACP_PROVIDER_ID = "agy-acp";
export const AGY_ACP_API = "agy-acp";

const EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"] as const;
const DEFAULT_PARK_BATCH_MS = 10;
const DEFAULT_AUX_IDLE_MS = 60_000;

export interface PersistedAcpBinding {
	piSessionId: string;
	acpSessionId: string;
	syncedFingerprint: string;
	systemPromptHash: string;
	cwd: string;
	modelId: string;
}

export interface SessionSyncContext {
	piSessionId: string;
	cwd: string;
	persistedBinding?: PersistedAcpBinding;
	isFork?: boolean;
}

export interface AgyAcpProviderOptions {
	cacheDir: string;
	binaryPath?: string;
	parkBatchMs?: number;
	/** Idle timeout before disposing the auxiliary `agy acp` process (default 60s). */
	auxIdleMs?: number;
	log?: (message: string) => void;
	onBindingChange?: (binding: PersistedAcpBinding) => void;
	/** Test seams: inject custom MCP server / ACP transports instead of spawning real processes. */
	createMcpServer?: () => McpBridgeServer;
	createMainTransport?: (mcpEndpoint: { url: string; token: string }) => AcpTransport;
	createAuxTransport?: () => AcpTransport;
	checkAvailable?: () => boolean;
}

export interface AgyAcpProviderHandle {
	provider: Provider<typeof AGY_ACP_API>;
	/** Updates active Pi session context (called on `session_start` and `session_tree`). */
	syncSession(ctx: SessionSyncContext): void;
	expectLiveRequest(signal: AbortSignal | undefined): void;
	/** Invalidates live ACP session continuity (called on `session_compact`). */
	invalidateSession(reason?: string): void;
	/** Cancels any parked MCP tool calls if Pi exits the agent loop without resuming. */
	abandonParkedTurn(reason?: string): void;
	/** Disposes child processes and the in-memory MCP server. */
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
): Model<typeof AGY_ACP_API> {
	return {
		id,
		name,
		api: AGY_ACP_API,
		provider: AGY_ACP_PROVIDER_ID,
		baseUrl: `acp://${AGY_ACP_PROVIDER_ID}`,
		reasoning: thinkingMap !== undefined,
		...(thinkingMap ? { thinkingLevelMap: thinkingMap } : {}),
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: defaultContextWindow(id),
		maxTokens: 65_536,
	};
}

/** Converts an `agy acp` `session/new` config snapshot into Pi `Model` records. */
export function modelsFromSessionConfig(config: AcpSessionConfig): Model<typeof AGY_ACP_API>[] {
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

export const BASELINE_MODELS: readonly Model<typeof AGY_ACP_API>[] = [
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

type PhaseEnd =
	| { kind: "parked"; calls: ParkedToolCall[] }
	| { kind: "done"; result: AcpPromptResult }
	| { kind: "error"; error: Error }
	| { kind: "aborted" };

interface ActiveBridge {
	sessionId: string;
	model: Model<Api>;
	cwd: string;
	systemPromptHash: string;
	promptController: AbortController;
	writer: MessageWriter | null;
	onRawUpdate?: (params: unknown) => void;
	detachSignal: (() => void) | null;
	bufferedUpdates: AcpSessionUpdate[];
	pendingParks: ParkedToolCall[];
	awaitingResults: Map<string, ParkedToolCall>;
	parkBatchTimer: NodeJS.Timeout | null;
	phaseResolve: ((end: PhaseEnd) => void) | null;
	terminalOutcome: { kind: "done"; result: AcpPromptResult } | { kind: "error"; error: Error } | null;
	latestUsageUpdate: { used: number; size: number } | null;
	stepCount: number;
	aborted: boolean;
}

interface BoundSession {
	acpSessionId: string;
	cwd: string;
	systemPromptHash: string;
	syncedFingerprint: string;
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

	const modelSelect = config?.model;
	if (modelSelect?.configId && (force || modelSelect.currentValue !== model.id || appliedModel !== model.id)) {
		if (!modelSelect.choices.some((c) => c.value === model.id)) {
			const offered = modelSelect.choices.map((c) => c.value).join(", ");
			throw new Error(`agy acp does not offer model '${model.id}'. Available: ${offered || "none"}.`);
		}
		config = await transport.setConfigOption(sessionId, modelSelect.configId, model.id, signal);
		appliedModel = model.id;
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

export function createAgyAcpProvider(options: AgyAcpProviderOptions): AgyAcpProviderHandle {
	const log = options.log ?? (() => {});
	const parkBatchMs = options.parkBatchMs ?? DEFAULT_PARK_BATCH_MS;
	const auxIdleMs = options.auxIdleMs ?? DEFAULT_AUX_IDLE_MS;

	let models: Model<typeof AGY_ACP_API>[] = [...BASELINE_MODELS];
	let activePiSessionId: string | undefined;
	let pendingLiveSignal: AbortSignal | undefined;
	let activeCwd = process.cwd();
	let persistedBinding: PersistedAcpBinding | undefined;
	let boundSession: BoundSession | null = null;
	let activeBridge: ActiveBridge | null = null;

	let mcpServer: McpBridgeServer | null = null;
	let mcpStarting: Promise<McpBridgeServer> | null = null;
	let mainTransport: AcpTransport | null = null;
	let auxTransport: AcpTransport | null = null;
	let auxInFlight = 0;
	let auxIdleTimer: NodeJS.Timeout | null = null;

	const checkAvailable = (): boolean =>
		options.checkAvailable ? options.checkAvailable() : resolveAgyBinary(options.binaryPath) !== undefined;

	const ensureMcpServer = async (): Promise<McpBridgeServer> => {
		if (mcpServer) return mcpServer;
		mcpStarting ??= (async () => {
			const server = options.createMcpServer ? options.createMcpServer() : new McpBridgeServer();
			await server.start();
			mcpServer = server;
			return server;
		})().catch((error) => {
			mcpStarting = null;
			throw error;
		});
		return mcpStarting;
	};

	const ensureMainTransport = async (): Promise<{ transport: AcpTransport; mcp: McpBridgeServer }> => {
		const mcp = await ensureMcpServer();
		if (!mainTransport) {
			const endpoint = { url: mcp.url, token: mcp.authToken };
			mainTransport = options.createMainTransport
				? options.createMainTransport(endpoint)
				: new AgyAcpClient({
						binaryPath: options.binaryPath,
						cacheDir: options.cacheDir,
						mcpEndpoint: endpoint,
						log: (msg) => log(`main: ${msg}`),
					});
		}
		return { transport: mainTransport, mcp };
	};

	const disposeAuxTransport = (): void => {
		if (auxIdleTimer) {
			clearTimeout(auxIdleTimer);
			auxIdleTimer = null;
		}
		if (auxTransport) {
			auxTransport.dispose();
			auxTransport = null;
		}
	};

	const acquireAuxTransport = (): AcpTransport => {
		if (auxIdleTimer) {
			clearTimeout(auxIdleTimer);
			auxIdleTimer = null;
		}
		auxInFlight += 1;
		if (!auxTransport) {
			auxTransport = options.createAuxTransport
				? options.createAuxTransport()
				: new AgyAcpClient({
						binaryPath: options.binaryPath,
						cacheDir: options.cacheDir,
						mcpEndpoint: undefined,
						ephemeral: true,
						log: (msg) => log(`aux: ${msg}`),
					});
		}
		return auxTransport;
	};

	const releaseAuxTransport = (immediate = false): void => {
		auxInFlight = Math.max(0, auxInFlight - 1);
		if (auxInFlight > 0 || !auxTransport) return;
		if (immediate || auxIdleMs <= 0) {
			disposeAuxTransport();
			return;
		}
		if (auxIdleTimer) clearTimeout(auxIdleTimer);
		auxIdleTimer = setTimeout(() => {
			auxIdleTimer = null;
			if (auxInFlight === 0) disposeAuxTransport();
		}, auxIdleMs);
		auxIdleTimer.unref();
	};

	const takeLiveRequest = (streamOptions: StreamOptions | undefined): boolean => {
		// Pi 1.6.0 dist/core/sdk.js installs this observer on the agent loop.
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

	const clearBridge = (bridge: ActiveBridge, cancelRemote: boolean, reason: string): void => {
		if (bridge.parkBatchTimer) {
			clearTimeout(bridge.parkBatchTimer);
			bridge.parkBatchTimer = null;
		}
		bridge.detachSignal?.();
		bridge.detachSignal = null;
		bridge.writer = null;
		bridge.onRawUpdate = undefined;

		if (cancelRemote) {
			bridge.aborted = true;
			for (const park of bridge.pendingParks) park.reject(reason);
			bridge.pendingParks = [];
			for (const park of bridge.awaitingResults.values()) park.reject(reason);
			bridge.awaitingResults.clear();
			mcpServer?.failAllParks(reason);
			bridge.promptController.abort();
		}

		if (activeBridge === bridge) {
			activeBridge = null;
			mcpServer?.setParkHandler(undefined, undefined);
		}
	};

	const abandonParkedTurn = (reason = "Pi ended the turn before tool results were delivered."): void => {
		pendingLiveSignal = undefined;
		if (!activeBridge) return;
		const bridge = activeBridge;
		const resolve = bridge.phaseResolve;
		bridge.phaseResolve = null;
		clearBridge(bridge, true, reason);
		resolve?.({ kind: "aborted" });
	};

	const routeUpdate = (bridge: ActiveBridge, update: AcpSessionUpdate): void => {
		if (update.type === "usage_update") {
			bridge.latestUsageUpdate = { used: update.used, size: update.size };
			return;
		}
		if (update.type !== "agent_message_chunk" && update.type !== "agent_thought_chunk") return;
		if (!bridge.writer || bridge.writer.isSettled) {
			bridge.bufferedUpdates.push(update);
			return;
		}
		if (update.type === "agent_message_chunk") bridge.writer.append("text", update.text);
		else bridge.writer.append("thinking", update.text);
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

	const flushPendingParks = (bridge: ActiveBridge): void => {
		if (bridge.parkBatchTimer) {
			clearTimeout(bridge.parkBatchTimer);
			bridge.parkBatchTimer = null;
		}
		if (bridge.pendingParks.length === 0 || !bridge.phaseResolve) return;
		const calls = bridge.pendingParks;
		bridge.pendingParks = [];
		for (const call of calls) bridge.awaitingResults.set(call.callId, call);
		const resolve = bridge.phaseResolve;
		bridge.phaseResolve = null;
		resolve({ kind: "parked", calls });
	};

	const awaitBridgePhase = (bridge: ActiveBridge): Promise<PhaseEnd> => {
		if (bridge.aborted) return Promise.resolve({ kind: "aborted" });
		if (bridge.terminalOutcome) {
			const outcome = bridge.terminalOutcome;
			bridge.terminalOutcome = null;
			return Promise.resolve(outcome);
		}
		if (bridge.pendingParks.length > 0) {
			const calls = bridge.pendingParks;
			bridge.pendingParks = [];
			for (const call of calls) bridge.awaitingResults.set(call.callId, call);
			return Promise.resolve({ kind: "parked", calls });
		}
		return new Promise<PhaseEnd>((resolve) => {
			bridge.phaseResolve = resolve;
		});
	};

	const settleMainPhase = (
		bridge: ActiveBridge,
		phase: PhaseEnd,
		writer: MessageWriter,
		message: AssistantMessage,
		context: TranscriptContext,
		model: Model<Api>,
	): void => {
		bridge.detachSignal?.();
		bridge.detachSignal = null;
		bridge.writer = null;
		bridge.onRawUpdate = undefined;

		if (phase.kind === "parked") {
			bridge.stepCount += 1;
			for (const call of phase.calls) {
				writer.emitToolCall(call.callId, call.piName, call.args);
			}
			writer.done("toolUse");
			return;
		}

		// Terminal outcome for this ACP prompt turn.
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
		message.usage = normalizeTurnUsage(model, result.usage, bridge.latestUsageUpdate, bridge.stepCount);
		finishFromStopReason(writer, result.stopReason);

		if (result.stopReason === "end_turn" || result.stopReason === "max_tokens" || result.stopReason === "max_turn_requests") {
			const syncedFingerprint = fingerprintMessages([...context.messages, message]);
			if (boundSession && boundSession.acpSessionId === bridge.sessionId) {
				boundSession.syncedFingerprint = syncedFingerprint;
			}
			if (activePiSessionId) {
				const binding: PersistedAcpBinding = {
					piSessionId: activePiSessionId,
					acpSessionId: bridge.sessionId,
					syncedFingerprint,
					systemPromptHash: bridge.systemPromptHash,
					cwd: bridge.cwd,
					modelId: model.id,
				};
				persistedBinding = binding;
				options.onBindingChange?.(binding);
			}
		}
	};

	async function runAuxiliaryTurn(
		stream: AssistantMessageEventStream,
		model: Model<Api>,
		context: TranscriptContext,
		streamOptions: StreamOptions | undefined,
	): Promise<void> {
		const signal = streamOptions?.signal;
		const message: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: zeroUsage(),
			stopReason: "pending",
			timestamp: Date.now(),
		};
		const writer = createMessageWriter(stream, message);
		const promptBlocks = renderTranscriptReplay(context.messages);
		if (!promptBlocks) {
			writer.fail("error", "No prompt content to send to Antigravity ACP.");
			return;
		}
		const systemPrompt = renderSystemPromptAppend(context.messages, log);
		const transport = acquireAuxTransport();
		let sessionId: string | undefined;
		let latestUsageUpdate: { used: number; size: number } | null = null;

		try {
			const created = await transport.createSession({
				cwd: activeCwd,
				systemPromptAppend: systemPrompt || undefined,
				signal,
			});
			sessionId = created.sessionId;
			const reasoning = (streamOptions as SimpleStreamOptions | undefined)?.reasoning;
			await applySessionModelAndThinking(transport, sessionId, model, reasoning, true, undefined, undefined, signal);

			writer.ensureStarted();
			const defaultPayload = { sessionId, prompt: promptBlocks };
			const rewritten = (await streamOptions?.onPayload?.(defaultPayload, model)) as
				| { prompt?: AcpPromptBlock[] }
				| undefined;
			const finalPrompt = rewritten?.prompt ?? promptBlocks;

			log(`aux full transcript replay: standalone request sid=${sessionId}`);
			const result = await transport.prompt({
				sessionId,
				prompt: finalPrompt,
				signal,
				onRawUpdate: streamOptions?.onProviderStreamEvent
					? (raw) => {
							void Promise.resolve(streamOptions.onProviderStreamEvent?.(raw, model)).catch((err) =>
								log(`onProviderStreamEvent failed: ${errorMessage(err)}`),
							);
						}
					: undefined,
				onUpdate: (update) => {
					if (update.type === "usage_update") latestUsageUpdate = { used: update.used, size: update.size };
					else if (update.type === "agent_message_chunk") writer.append("text", update.text);
					else if (update.type === "agent_thought_chunk") writer.append("thinking", update.text);
				},
			});
			log(`aux raw ACP usage sid=${sessionId}: ${JSON.stringify(result.usage ?? null)}`);
			message.usage = normalizeTurnUsage(model, result.usage, latestUsageUpdate, 1);
			finishFromStopReason(writer, result.stopReason);
		} catch (error) {
			if (signal?.aborted) writer.fail("aborted", "Request was aborted");
			else writer.fail("error", errorMessage(error));
		} finally {
			if (sessionId) await transport.closeSession(sessionId);
			releaseAuxTransport(false);
		}
	}

	async function runMainTurn(
		stream: AssistantMessageEventStream,
		model: Model<Api>,
		context: TranscriptContext,
		streamOptions: StreamOptions | undefined,
	): Promise<void> {
		const signal = streamOptions?.signal;
		const message: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: zeroUsage(),
			stopReason: "pending",
			timestamp: Date.now(),
		};
		const writer = createMessageWriter(stream, message);

		const onRawUpdate = streamOptions?.onProviderStreamEvent
			? (raw: unknown) => {
					void Promise.resolve(streamOptions.onProviderStreamEvent?.(raw, model)).catch((err) =>
						log(`onProviderStreamEvent failed: ${errorMessage(err)}`),
					);
				}
			: undefined;

		// Case 1: Resume an active parked ACP turn if matching tool results arrived.
		if (activeBridge && activeBridge.awaitingResults.size > 0) {
			const bridge = activeBridge;
			const tail = extractPostAssistantTail(context.messages);
			const resultById = new Map((tail?.toolResults ?? []).map((r) => [r.toolCallId, r]));
			const allMatched =
				tail !== null &&
				bridge.model.id === model.id &&
				bridge.cwd === activeCwd &&
				[...bridge.awaitingResults.keys()].every((id) => resultById.has(id));

			if (allMatched && tail) {
				try {
					const mcp = await ensureMcpServer();
					await mcp.updateTools(projectPiTools(getCurrentTools(context.messages)));
				} catch (error) {
					log(`tool update before unpark failed: ${errorMessage(error)}`);
				}

				writer.ensureStarted();
				bridge.writer = writer;
				bridge.onRawUpdate = onRawUpdate;
				attachSignalToBridge(bridge, signal);
				flushBufferedUpdates(bridge);

				const deliveries = [...bridge.awaitingResults.values()];
				bridge.awaitingResults.clear();

				const steeringText =
					tail.steeringTexts.length > 0
						? `\n\n[User steering message received during tool execution]:\n${tail.steeringTexts.join("\n")}`
						: "";
				const steeringImages: ImageContent[] = tail.steeringImages;

				const phasePromise = awaitBridgePhase(bridge);

				for (let i = 0; i < deliveries.length; i++) {
					const park = deliveries[i];
					const toolRes = resultById.get(park.callId)!;
					const content: (TextContent | ImageContent)[] = [...toolRes.content];
					if (i === deliveries.length - 1) {
						if (steeringText) content.push({ type: "text", text: steeringText });
						if (steeringImages.length > 0) content.push(...steeringImages);
					}
					park.resolve({ content, isError: toolRes.isError });
				}

				const phase = await phasePromise;
				settleMainPhase(bridge, phase, writer, message, context, model);
				return;
			}

			// Transcript diverged (e.g. threshold compaction ran mid-loop) — abort parked turn cleanly.
			abandonParkedTurn("Pi transcript changed before tool results were delivered.");
		}

		// Case 2: Start a fresh ACP prompt turn (reusing/resuming warm session if continuity matches).
		const systemPrompt = renderSystemPromptAppend(context.messages, log);
		const systemPromptHash = hashSystemPromptAppend(context.messages);
		const projectedTools = projectPiTools(getCurrentTools(context.messages));

		let transport: AcpTransport;
		let mcp: McpBridgeServer;
		try {
			({ transport, mcp } = await ensureMainTransport());
			await mcp.updateTools(projectedTools);
		} catch (error) {
			if (signal?.aborted) writer.fail("aborted", "Request was aborted");
			else writer.fail("error", errorMessage(error));
			return;
		}

		const binding = boundSession ?? (persistedBinding?.piSessionId === activePiSessionId ? persistedBinding : undefined);
		let delta = binding?.cwd === activeCwd ? messagesAfterFingerprint(context.messages, binding.syncedFingerprint) : null;
		let session = delta !== null ? boundSession : null;
		let replayReason = !binding
			? "cold start (no ACP session)"
			: binding.cwd !== activeCwd ? "working directory changed" : "diverged prefix";
		let needApplySettings = false;
		let sessionId: string;

		try {
			// An append changed during a parked turn is applied here, after that ACP prompt finishes.
			if (delta !== null && binding && (!session || session.systemPromptHash !== systemPromptHash)) {
				try {
					await transport.resumeSession({
						sessionId: binding.acpSessionId,
						cwd: activeCwd,
						systemPromptAppend: systemPrompt,
						signal,
					});
					session = {
						acpSessionId: binding.acpSessionId,
						cwd: activeCwd,
						systemPromptHash,
						syncedFingerprint: binding.syncedFingerprint,
						lastModelId: undefined,
						lastThoughtLevel: undefined,
					};
					// agy resets the model on session/resume.
					needApplySettings = true;
				} catch (error) {
					if (session) {
						log(`main session/resume failed sid=${session.acpSessionId}; keeping previous system append: ${errorMessage(error)}`);
					} else {
						log(`main session/resume failed sid=${binding.acpSessionId}: ${errorMessage(error)}`);
						replayReason = "session/resume failed";
					}
				}
			}

			if (!session) {
				delta = null;
				persistedBinding = undefined;
				if (boundSession) {
					await transport.closeSession(boundSession.acpSessionId);
					boundSession = null;
				}
				log(`main createSession cwd=${activeCwd} sysLen=${systemPrompt.length} tools=${projectedTools.map(t=>t.mcpName).join(",")}`);
				const created = await transport.createSession({
					cwd: activeCwd,
					systemPromptAppend: systemPrompt || undefined,
					signal,
				});
				log(`main createSession ok sid=${created.sessionId}`);
				session = {
					acpSessionId: created.sessionId,
					cwd: activeCwd,
					systemPromptHash,
					syncedFingerprint: "",
					lastModelId: undefined,
					lastThoughtLevel: undefined,
				};
				needApplySettings = true;
			}

			boundSession = session;
			sessionId = session.acpSessionId;
			const reasoning = (streamOptions as SimpleStreamOptions | undefined)?.reasoning;
			const applied = await applySessionModelAndThinking(
				transport,
				sessionId,
				model,
				reasoning,
				needApplySettings,
				session.lastModelId,
				session.lastThoughtLevel,
				signal,
			);
			log(`main applied model=${applied.modelId} thought=${applied.thoughtLevel}`);
			session.lastModelId = applied.modelId;
			session.lastThoughtLevel = applied.thoughtLevel;
		} catch (error) {
			if (signal?.aborted) writer.fail("aborted", "Request was aborted");
			else writer.fail("error", errorMessage(error));
			return;
		}

		const promptBlocks: AcpPromptBlock[] | null = delta === null
			? renderTranscriptReplay(context.messages)
			: delta.length === 0
				? [{ type: "text", text: "Continue the conversation." }]
				: renderTranscriptDelta(delta);

		if (!promptBlocks || promptBlocks.length === 0) {
			writer.fail("error", "No prompt content found to send to Antigravity ACP.");
			return;
		}

		let finalPrompt: AcpPromptBlock[] = promptBlocks;
		try {
			const defaultPayload = { sessionId, prompt: promptBlocks };
			const rewritten = (await streamOptions?.onPayload?.(defaultPayload, model)) as
				| { prompt?: AcpPromptBlock[] }
				| undefined;
			if (rewritten?.prompt) finalPrompt = rewritten.prompt;
		} catch (error) {
			writer.fail("error", errorMessage(error));
			return;
		}

		writer.ensureStarted();

		const bridge: ActiveBridge = {
			sessionId,
			model,
			cwd: activeCwd,
			systemPromptHash: session.systemPromptHash,
			promptController: new AbortController(),
			writer,
			onRawUpdate,
			detachSignal: null,
			bufferedUpdates: [],
			pendingParks: [],
			awaitingResults: new Map(),
			parkBatchTimer: null,
			phaseResolve: null,
			terminalOutcome: null,
			latestUsageUpdate: null,
			stepCount: 1,
			aborted: false,
		};
		activeBridge = bridge;
		attachSignalToBridge(bridge, signal);

		mcp.setParkHandler(
			(park) => {
				if (activeBridge !== bridge || bridge.aborted) return false;
				bridge.pendingParks.push(park);
				if (!bridge.parkBatchTimer) {
					if (parkBatchMs <= 0) {
						queueMicrotask(() => flushPendingParks(bridge));
					} else {
						bridge.parkBatchTimer = setTimeout(() => flushPendingParks(bridge), parkBatchMs);
						bridge.parkBatchTimer.unref();
					}
				}
				return true;
			},
			(cancelledId) => {
				bridge.awaitingResults.delete(cancelledId);
			},
		);

		const phasePromise = awaitBridgePhase(bridge);
		if (delta === null) log(`main full transcript replay: ${replayReason} sid=${sessionId}`);
		log(`main prompt sending sid=${sessionId} delta=${delta !== null} blocks=${finalPrompt.length}`);

		void transport
			.prompt({
				sessionId,
				prompt: finalPrompt,
				signal: bridge.promptController.signal,
				onRawUpdate: (raw) => bridge.onRawUpdate?.(raw),
				onUpdate: (update) => routeUpdate(bridge, update),
			})
			.then(
				(result) => {
					log(`main raw ACP usage sid=${sessionId}: ${JSON.stringify(result.usage ?? null)}`);
					if (bridge.aborted) return;
					const resolve = bridge.phaseResolve;
					bridge.phaseResolve = null;
					if (resolve) resolve({ kind: "done", result });
					else bridge.terminalOutcome = { kind: "done", result };
				},
				(error: unknown) => {
					if (bridge.aborted) return;
					const err = error instanceof Error ? error : new Error(String(error));
					const resolve = bridge.phaseResolve;
					bridge.phaseResolve = null;
					if (resolve) resolve({ kind: "error", error: err });
					else bridge.terminalOutcome = { kind: "error", error: err };
				},
			);

		const phase = await phasePromise;
		log(`main phase ended kind=${phase.kind}`);
		settleMainPhase(bridge, phase, writer, message, context, model);
	}

	const stream = (
		model: Model<typeof AGY_ACP_API>,
		context: TranscriptContext,
		streamOptions?: StreamOptions,
	): AssistantMessageEventStream => {
		const events = createAssistantMessageEventStream();
		const aux = !takeLiveRequest(streamOptions);
		log(`stream start aux=${aux} model=${model.id} optSid=${streamOptions?.sessionId ?? "none"} activeSid=${activePiSessionId ?? "none"} msgs=${context.messages.length}`);
		const runner = aux
			? runAuxiliaryTurn(events, model, context, streamOptions)
			: runMainTurn(events, model, context, streamOptions);
		runner.catch((error: unknown) => log(`stream crashed: ${errorMessage(error)}`));
		return events;
	};

	const refreshModels = async (context: RefreshModelsContext): Promise<void> => {
		const storedModels = (context.stored?.models ?? []).filter(
			(m): m is Model<typeof AGY_ACP_API> => m.provider === AGY_ACP_PROVIDER_ID && m.api === AGY_ACP_API,
		);
		if (storedModels.length > 0) {
			await context.publish({
				update: () => {
					models = storedModels;
				},
			});
		}

		if (!context.allowNetwork || context.signal.aborted || !checkAvailable()) return;

		const transport = acquireAuxTransport();
		let probeSessionId: string | undefined;
		try {
			const created = await transport.createSession({ cwd: activeCwd, signal: context.signal });
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
			if (probeSessionId) await transport.closeSession(probeSessionId);
			releaseAuxTransport(true);
		}
	};

	const provider: Provider<typeof AGY_ACP_API> = {
		id: AGY_ACP_PROVIDER_ID,
		name: "Antigravity ACP",
		baseUrl: `acp://${AGY_ACP_PROVIDER_ID}`,
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
		syncSession(ctx: SessionSyncContext): void {
			abandonParkedTurn("Pi switched session or branch.");
			activePiSessionId = ctx.piSessionId;
			activeCwd = ctx.cwd;
			if (ctx.isFork) {
				boundSession = null;
				persistedBinding = undefined;
				return;
			}
			// If we already have a live session matching the persisted binding, keep it;
			// otherwise clear the in-memory binding and let the next turn resume from `persistedBinding`.
			if (
				boundSession &&
				ctx.persistedBinding &&
				boundSession.acpSessionId === ctx.persistedBinding.acpSessionId &&
				boundSession.cwd === ctx.cwd
			) {
				persistedBinding = ctx.persistedBinding;
				return;
			}
			boundSession = null;
			persistedBinding = ctx.persistedBinding;
		},
		expectLiveRequest(signal: AbortSignal | undefined): void {
			pendingLiveSignal = signal;
		},
		invalidateSession(reason?: string): void {
			abandonParkedTurn(reason ? `Session invalidated: ${reason}` : "Session invalidated.");
			if (boundSession && mainTransport) {
				void mainTransport.closeSession(boundSession.acpSessionId);
			}
			boundSession = null;
			persistedBinding = undefined;
		},
		abandonParkedTurn,
		async dispose(): Promise<void> {
			abandonParkedTurn("Pi session shutting down.");
			if (boundSession && mainTransport) {
				await mainTransport.closeSession(boundSession.acpSessionId);
			}
			boundSession = null;
			mainTransport?.dispose();
			mainTransport = null;
			disposeAuxTransport();
			if (mcpServer) {
				await mcpServer.stop();
				mcpServer = null;
				mcpStarting = null;
			}
		},
	};
}
