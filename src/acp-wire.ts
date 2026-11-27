/**
 * ACP (Agent Client Protocol) v1 JSON-RPC 2.0 wire types and boundary parsers.
 */

export const ACP_PROTOCOL_VERSION = 1;
export const ACP_AUTH_REQUIRED = -32000;
export const RPC_METHOD_NOT_FOUND = -32601;
export const RPC_INVALID_PARAMS = -32602;
export const RPC_INTERNAL_ERROR = -32603;

export function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

const optionalString = (value: unknown): string | undefined => (typeof value === "string" ? value : undefined);
const optionalNumber = (value: unknown): number | undefined =>
	typeof value === "number" && Number.isFinite(value) ? value : undefined;

export interface AcpPermissionOption {
	optionId: string;
	name: string;
	kind: string;
}

export interface AcpPermissionRequest {
	sessionId: string | undefined;
	toolCallId: string | undefined;
	title: string | undefined;
	kind: string | undefined;
	rawInput: unknown;
	options: AcpPermissionOption[];
}

export function parsePermissionRequest(params: unknown): AcpPermissionRequest {
	const record = isRecord(params) ? params : {};
	const toolCall = isRecord(record.toolCall) ? record.toolCall : {};
	const rawOptions = Array.isArray(record.options) ? record.options : [];
	const options = rawOptions.flatMap((option): AcpPermissionOption[] => {
		if (!isRecord(option) || typeof option.optionId !== "string") return [];
		return [{
			optionId: option.optionId,
			name: optionalString(option.name) ?? option.optionId,
			kind: optionalString(option.kind) ?? "",
		}];
	});
	return {
		sessionId: optionalString(record.sessionId),
		toolCallId: optionalString(toolCall.toolCallId),
		title: optionalString(toolCall.title),
		kind: optionalString(toolCall.kind),
		rawInput: toolCall.rawInput,
		options,
	};
}

/** Answers `session/request_permission` with an explicit optionId, or the first option of the wanted kind, else cancels. */
export function permissionOutcome(params: unknown, decision: boolean | string): {
	outcome: { outcome: "selected"; optionId: string } | { outcome: "cancelled" };
} {
	const options = isRecord(params) && Array.isArray(params.options) ? params.options : [];
	if (typeof decision === "string") {
		const exact = options.find((candidate: unknown) => isRecord(candidate) && candidate.optionId === decision);
		return isRecord(exact)
			? { outcome: { outcome: "selected", optionId: decision } }
			: { outcome: { outcome: "cancelled" } };
	}
	const wanted = decision ? "allow_once" : "reject_once";
	const option: unknown = options.find((candidate: unknown) =>
		isRecord(candidate) && candidate.kind === wanted && typeof candidate.optionId === "string");
	return isRecord(option) && typeof option.optionId === "string"
		? { outcome: { outcome: "selected", optionId: option.optionId } }
		: { outcome: { outcome: "cancelled" } };
}

function oneOf<T extends string>(allowed: readonly T[], value: unknown): T | undefined {
	return allowed.includes(value as T) ? (value as T) : undefined;
}

// --- JSON-RPC 2.0 envelopes -------------------------------------------------

export type RpcId = number | string;

export interface RpcError {
	code: number;
	message: string;
}

export type RpcIncoming =
	| { kind: "response"; id: RpcId; outcome: { ok: true; result: unknown } | { ok: false; error: RpcError } }
	| { kind: "request"; id: RpcId; method: string; params: unknown }
	| { kind: "notification"; method: string; params: unknown };

const isRpcId = (value: unknown): value is RpcId => typeof value === "number" || typeof value === "string";

export function parseRpcMessage(value: unknown): RpcIncoming | null {
	if (!isRecord(value)) return null;
	const { id, method } = value;
	if (typeof method === "string") {
		return isRpcId(id)
			? { kind: "request", id, method, params: value.params }
			: { kind: "notification", method, params: value.params };
	}
	if (!isRpcId(id)) return null;
	if (isRecord(value.error)) {
		return {
			kind: "response",
			id,
			outcome: {
				ok: false,
				error: {
					code: typeof value.error.code === "number" ? value.error.code : RPC_INTERNAL_ERROR,
					message: optionalString(value.error.message) ?? "Unknown ACP error",
				},
			},
		};
	}
	return "result" in value ? { kind: "response", id, outcome: { ok: true, result: value.result } } : null;
}

// --- Prompt blocks ----------------------------------------------------------

export type AcpPromptBlock =
	| { type: "text"; text: string }
	| { type: "image"; data: string; mimeType: string };

// --- session/update ---------------------------------------------------------

export type AcpSessionUpdate =
	| { type: "agent_message_chunk"; text: string }
	| { type: "agent_thought_chunk"; text: string }
	| { type: "usage_update"; used: number; size: number }
	| {
			type: "tool_call" | "tool_call_update";
			toolCallId: string;
			title: string | undefined;
			status: string | undefined;
			rawOutput: unknown;
			content: unknown;
	  };

const updateOf = (params: unknown): Record<string, unknown> | undefined =>
	isRecord(params) && isRecord(params.update) ? params.update : undefined;

function contentBlockText(block: unknown): string {
	return isRecord(block) && block.type === "text" ? (optionalString(block.text) ?? "") : "";
}

export function parseSessionUpdate(params: unknown): AcpSessionUpdate | null {
	const update = updateOf(params);
	switch (update?.sessionUpdate) {
		case "agent_message_chunk":
		case "agent_thought_chunk":
			return { type: update.sessionUpdate, text: contentBlockText(update.content) };
		case "usage_update": {
			const used = optionalNumber(update.used);
			const size = optionalNumber(update.size);
			if (used === undefined || size === undefined) return null;
			return { type: "usage_update", used, size };
		}
		case "tool_call":
		case "tool_call_update":
			if (typeof update.toolCallId !== "string") return null;
			return {
				type: update.sessionUpdate,
				toolCallId: update.toolCallId,
				title: optionalString(update.title),
				status: optionalString(update.status),
				rawOutput: update.rawOutput,
				content: update.content,
			};
		default:
			return null;
	}
}

// --- initialize -------------------------------------------------------------

export interface AcpAuthMethod {
	id: string;
	name: string;
	description: string | undefined;
}

export interface AcpAgentInfo {
	protocolVersion: number;
	name: string;
	version: string;
	authMethods: AcpAuthMethod[];
	supportsResume: boolean;
	supportsClose: boolean;
	supportsSystemPromptAppend: boolean;
}

function parseAuthMethods(value: unknown): AcpAuthMethod[] {
	if (!Array.isArray(value)) return [];
	return value.flatMap((method): AcpAuthMethod[] =>
		isRecord(method) && typeof method.id === "string"
			? [{ id: method.id, name: optionalString(method.name) ?? method.id, description: optionalString(method.description) }]
			: [],
	);
}

export function parseInitializeResult(result: unknown): AcpAgentInfo {
	const record = isRecord(result) ? result : {};
	const info = isRecord(record.agentInfo) ? record.agentInfo : {};
	const caps = isRecord(record.agentCapabilities) ? record.agentCapabilities : {};
	const sessionCaps = isRecord(caps.sessionCapabilities) ? caps.sessionCapabilities : {};
	const meta = isRecord(caps._meta) ? caps._meta : {};
	const agyMeta = isRecord(meta.antigravity) ? meta.antigravity : {};
	const sysPromptMeta = isRecord(agyMeta.systemPrompt) ? agyMeta.systemPrompt : {};
	return {
		protocolVersion: typeof record.protocolVersion === "number" ? record.protocolVersion : ACP_PROTOCOL_VERSION,
		name: optionalString(info.name) ?? "agy-acp",
		version: optionalString(info.version) ?? "0.0.0",
		authMethods: parseAuthMethods(record.authMethods),
		supportsResume: "resume" in sessionCaps,
		supportsClose: "close" in sessionCaps,
		supportsSystemPromptAppend: sysPromptMeta.append === true,
	};
}

// --- Session configuration (models, thought_level, flattened variants) ------

export interface AcpChoice {
	value: string;
	name: string;
	description: string | undefined;
}

export interface AcpSelect {
	configId: string | null;
	currentValue: string | undefined;
	choices: AcpChoice[];
}

export interface AcpSessionConfig {
	model: AcpSelect | undefined;
	thoughtLevel: AcpSelect | undefined;
	/** Raw flattened `models.availableModels` entries (e.g. `gemini-4-neon-xhigh`) for per-model thinking discovery. */
	availableModelVariants: AcpChoice[];
}

const emptySessionConfig = (): AcpSessionConfig => ({
	model: undefined,
	thoughtLevel: undefined,
	availableModelVariants: [],
});

function parseChoice(record: Record<string, unknown>, value: string): AcpChoice {
	return { value, name: optionalString(record.name) ?? value, description: optionalString(record.description) };
}

function parseChoices(value: unknown): AcpChoice[] {
	if (!Array.isArray(value)) return [];
	return value.flatMap((item): AcpChoice[] => {
		if (!isRecord(item)) return [];
		if (Array.isArray(item.options)) return parseChoices(item.options);
		if (typeof item.value !== "string") return [];
		return [parseChoice(item, item.value)];
	});
}

function findSelect(configOptions: unknown, category: string): AcpSelect | undefined {
	if (!Array.isArray(configOptions)) return undefined;
	const option: unknown = configOptions.find(
		(candidate) => isRecord(candidate) && candidate.category === category && candidate.type === "select",
	);
	if (!isRecord(option) || typeof option.id !== "string") return undefined;
	return {
		configId: option.id,
		currentValue: optionalString(option.currentValue),
		choices: parseChoices(option.options),
	};
}

function parseLegacyModels(value: unknown): AcpSelect | undefined {
	if (!isRecord(value) || !Array.isArray(value.availableModels)) return undefined;
	const choices = value.availableModels.flatMap((model): AcpChoice[] =>
		isRecord(model) && typeof model.modelId === "string" ? [parseChoice(model, model.modelId)] : [],
	);
	return { configId: null, currentValue: optionalString(value.currentModelId), choices };
}

export function parseSessionConfig(result: unknown, previous: AcpSessionConfig = emptySessionConfig()): AcpSessionConfig {
	if (!isRecord(result)) return previous;
	const hasConfigOptions = Array.isArray(result.configOptions);
	const legacy = parseLegacyModels(result.models);
	const keptModel = !hasConfigOptions || previous.model?.configId === null ? previous.model : undefined;
	return {
		model: findSelect(result.configOptions, "model") ?? legacy ?? keptModel,
		thoughtLevel: hasConfigOptions ? findSelect(result.configOptions, "thought_level") : previous.thoughtLevel,
		availableModelVariants: legacy ? legacy.choices : previous.availableModelVariants,
	};
}

export function parseConfigOptionUpdate(params: unknown, previous: AcpSessionConfig): AcpSessionConfig | null {
	const update = updateOf(params);
	return update?.sessionUpdate === "config_option_update" ? parseSessionConfig(update, previous) : null;
}

export function parseNewSessionResult(result: unknown): { sessionId: string; config: AcpSessionConfig } {
	if (!isRecord(result) || typeof result.sessionId !== "string" || result.sessionId.length === 0) {
		throw new Error("session/new did not return a sessionId");
	}
	return { sessionId: result.sessionId, config: parseSessionConfig(result) };
}

// --- session/prompt result & usage ------------------------------------------

const STOP_REASONS = ["end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"] as const;
export type AcpStopReason = (typeof STOP_REASONS)[number];

export interface AcpUsage {
	inputTokens: number;
	outputTokens: number;
	cachedReadTokens: number;
	cachedWriteTokens: number;
	thoughtTokens: number;
	totalTokens: number;
}

export interface AcpPromptResult {
	stopReason: AcpStopReason;
	usage: AcpUsage | undefined;
}

function parseUsage(value: unknown): AcpUsage | undefined {
	if (!isRecord(value)) return undefined;
	const inputTokens = optionalNumber(value.inputTokens);
	const outputTokens = optionalNumber(value.outputTokens);
	const totalTokens = optionalNumber(value.totalTokens);
	if (inputTokens === undefined && outputTokens === undefined && totalTokens === undefined) return undefined;
	const cachedReadTokens = optionalNumber(value.cachedReadTokens) ?? optionalNumber(value.cachedTokens) ?? 0;
	const cachedWriteTokens = optionalNumber(value.cachedWriteTokens) ?? 0;
	const thoughtTokens = optionalNumber(value.thoughtTokens) ?? 0;
	const input = inputTokens ?? 0;
	const output = outputTokens ?? 0;
	return {
		inputTokens: input,
		outputTokens: output,
		cachedReadTokens,
		cachedWriteTokens,
		thoughtTokens,
		totalTokens: totalTokens ?? input + output + cachedReadTokens + cachedWriteTokens,
	};
}

export function parsePromptResult(result: unknown): AcpPromptResult {
	const record = isRecord(result) ? result : {};
	return {
		stopReason: oneOf(STOP_REASONS, record.stopReason) ?? "end_turn",
		usage: parseUsage(record.usage),
	};
}
