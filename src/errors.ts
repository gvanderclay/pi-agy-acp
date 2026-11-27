import type { AssistantMessage } from "@earendil-works/pi-ai";
import type { AcpPromptResult } from "./acp-wire.ts";

/** agy can resolve an error-only turn as end_turn with a warning and zero usage. */
export function terminalAcpError(message: AssistantMessage, result: AcpPromptResult): string | undefined {
	if (result.stopReason !== "end_turn" || !result.usage ||
		result.usage.outputTokens !== 0 || result.usage.thoughtTokens !== 0 ||
		message.content.some((part) => part.type !== "text")) return undefined;
	const text = message.content.map((part) => part.type === "text" ? part.text : "").join("").trim();
	const lines = text.split(/\r?\n/).filter((line) => line.trim());
	const warning = /^⚠\uFE0F?\s*(?:API error \(attempt \d+\):\s*)?(?:RESOURCE_EXHAUSTED|UNAVAILABLE|INTERNAL|DEADLINE_EXCEEDED|UNAUTHENTICATED|PERMISSION_DENIED|INVALID_ARGUMENT) \(code \d+\): .+$/;
	return lines.length > 0 && lines.every((line) => warning.test(line.trim())) ? text : undefined;
}
