import type { AssistantMessage } from "@earendil-works/pi-ai";
import type { AcpPromptResult } from "./acp-wire.ts";

// agy <1.5.14 streams its own retry attempts as a line of agent text; anchored so prose mentioning an API error stays text.
const RETRY_NOTICE = /(?:^|\n)[^\S\n]*(?:⚠\uFE0F?[^\S\n]*)?API error \(attempt \d+\):[^\S\n]*([^\n]+)/;
// agy >=1.5.14 retries silently and ends a turn whose retries all failed with one "⚠ <error>" line.
const FINAL_NOTICE = /^\s*⚠\uFE0F?[^\S\n]*([^\n]+?)\s*$/;
// A trailing warning line, withheld until the turn either continues or ends on it.
const TRAILING_WARNING = /(?<=^|\n)[^\S\n]*⚠[^\n]*\s*$/;
const QUOTA_EXHAUSTED = /\bRESOURCE_EXHAUSTED\b|\bcode 429\b|\b[Qq]uota reached\b/;
const NOTICE_MARK = /^[^\S\n]*(?:⚠\uFE0F?[^\S\n]*)?/;
const NOTICE_HEAD = "API error (attempt ";
const NOTICE_ATTEMPT = /^\d*$|^\d+\)(?::.*)?$/;

function quotaNotice(pattern: RegExp, text: string): string | undefined {
	const detail = pattern.exec(text)?.[1]?.trim();
	return detail && QUOTA_EXHAUSTED.test(detail) ? detail : undefined;
}

function mayBecomeRetryNotice(line: string): boolean {
	const rest = line.replace(NOTICE_MARK, "");
	if (rest.length < NOTICE_HEAD.length) return NOTICE_HEAD.startsWith(rest);
	return rest.startsWith(NOTICE_HEAD) && NOTICE_ATTEMPT.test(rest.slice(NOTICE_HEAD.length));
}

export type StreamKind = "text" | "thinking";

/**
 * Buffers unfinished trailing lines that may become an Agy retry or quota
 * notice, suppressing quota noise and surfacing a rate-limit error immediately.
 */
export class WarningLineFilter {
	private held: { kind: StreamKind; text: string } | undefined;

	push(kind: StreamKind, chunk: string, emit: (kind: StreamKind, text: string) => void): string | undefined {
		if (!chunk) return undefined;
		if (this.held && this.held.kind !== kind) this.flush(emit);
		const text = (this.held?.text ?? "") + chunk;
		this.held = undefined;

		const exhausted = quotaNotice(RETRY_NOTICE, text);
		if (exhausted) return `Antigravity rate limited: ${exhausted}`;

		const lineStart = text.lastIndexOf("\n") + 1;
		const retryStart = mayBecomeRetryNotice(text.slice(lineStart)) ? lineStart : text.length;
		const warningStart = TRAILING_WARNING.exec(text)?.index ?? text.length;
		const visible = text.slice(0, Math.min(retryStart, warningStart));
		if (visible.length < text.length) this.held = { kind, text: text.slice(visible.length) };
		if (visible) emit(kind, visible);
		return undefined;
	}

	flush(emit: (kind: StreamKind, text: string) => void): void {
		const held = this.held;
		this.held = undefined;
		if (held) emit(held.kind, held.text);
	}

	finish(emit: (kind: StreamKind, text: string) => void): string | undefined {
		const held = this.held;
		this.held = undefined;
		if (!held) return undefined;
		const exhausted = quotaNotice(FINAL_NOTICE, held.text);
		if (exhausted) return `Antigravity rate limited: ${exhausted}`;
		emit(held.kind, held.text);
		return undefined;
	}

	reset(): void {
		this.held = undefined;
	}
}

/** agy can resolve an error-only turn as end_turn with a warning and zero usage. */
export function terminalAcpError(message: AssistantMessage, result: AcpPromptResult): string | undefined {
	if (result.stopReason !== "end_turn" || !result.usage ||
		result.usage.outputTokens !== 0 || result.usage.thoughtTokens !== 0 ||
		message.content.some((part) => part.type !== "text")) return undefined;
	const text = message.content.map((part) => part.type === "text" ? part.text : "").join("").trim();
	const lines = text.split(/\r?\n/).filter((line) => line.trim());
	const warning = /^⚠\uFE0F?\s*(?:(?:API error \(attempt \d+\):\s*)?(?:RESOURCE_EXHAUSTED|UNAVAILABLE|INTERNAL|DEADLINE_EXCEEDED|UNAUTHENTICATED|PERMISSION_DENIED|INVALID_ARGUMENT) \(code \d+\): .+|.*[Qq]uota reached.*)$/;
	return lines.length > 0 && lines.every((line) => warning.test(line.trim())) ? text : undefined;
}

/**
 * Detects a silent blank `end_turn` (no visible text or tool calls, and no
 * generated output tokens) so Pi/Squire treats it as a failure instead of a
 * completed turn.
 */
export function blankAcpTurnError(
	message: AssistantMessage,
	result: AcpPromptResult,
	stepCount: number,
): string | undefined {
	if (result.stopReason !== "end_turn" || stepCount > 1) return undefined;
	const hasVisible = message.content.some(
		(part) => (part.type === "text" && part.text.trim().length > 0) || part.type === "toolCall",
	);
	if (hasVisible) return undefined;
	if (result.usage && result.usage.outputTokens > 0) return undefined;
	return "Antigravity ended the turn without visible output.";
}
