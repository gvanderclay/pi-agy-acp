/**
 * Prompt rendering between Pi messages and ACP `session/prompt` content blocks.
 */

import type { ImageContent, Message, TextContent, ToolResultMessage, UserMessage } from "@earendil-works/pi-ai";
import type { AcpPromptBlock } from "./acp-wire.ts";

/** Converts a Pi `UserMessage` into ACP `session/prompt` content blocks (text + images). */
export function userMessageToPromptBlocks(message: UserMessage): AcpPromptBlock[] {
	if (typeof message.content === "string") {
		return message.content.trim() ? [{ type: "text", text: message.content }] : [];
	}
	const blocks: AcpPromptBlock[] = [];
	for (const part of message.content) {
		if (part.type === "text") {
			if (part.text.length > 0) blocks.push({ type: "text", text: part.text });
		} else if (part.type === "image") {
			blocks.push({ type: "image", data: part.data, mimeType: part.mimeType });
		}
	}
	return blocks;
}

function renderContentParts(parts: readonly (TextContent | ImageContent)[]): string {
	const out: string[] = [];
	let images = 0;
	for (const part of parts) {
		if (part.type === "text" && part.text) out.push(part.text);
		else if (part.type === "image") images++;
	}
	if (images > 0) out.push(`[${images} image attachment(s)]`);
	return out.join("\n");
}

/**
 * Renders the kept Pi conversation into ACP prompt blocks for a fresh Agy session.
 * Preserves images attached to the final user message.
 */
export function renderTranscriptReplay(messages: readonly Message[]): AcpPromptBlock[] | null {
	const nonSystem = messages.filter((m) => m.role !== "system");
	const lastMsg = nonSystem.at(-1);
	const latestUser = lastMsg?.role === "user" ? lastMsg : undefined;
	const prior = latestUser ? nonSystem.slice(0, -1) : nonSystem;
	return renderTranscriptPrompt(prior, latestUser, "conversation_history");
}

/** Sends the conversation messages a warm Agy session has not seen. */
export function renderTranscriptDelta(messages: readonly Message[]): AcpPromptBlock[] | null {
	const latestUserIndex = messages.findLastIndex((message) => message.role === "user");
	const latestUser = messages[latestUserIndex];
	const prior = messages.filter((_, index) => index !== latestUserIndex);
	return renderTranscriptPrompt(prior, latestUser?.role === "user" ? latestUser : undefined, "conversation_context");
}

function renderTranscriptPrompt(
	prior: readonly Message[],
	latestUser: UserMessage | undefined,
	contextTag: "conversation_history" | "conversation_context",
): AcpPromptBlock[] | null {
	const historyLines: string[] = [];

	for (const msg of prior) {
		if (msg.role === "user") {
			const text = typeof msg.content === "string" ? msg.content : renderContentParts(msg.content);
			if (text.trim()) historyLines.push(`[User]:\n${text}`);
		} else if (msg.role === "assistant") {
			const chunks: string[] = [];
			for (const block of msg.content) {
				if (block.type === "text" && block.text.trim()) {
					chunks.push(block.text);
				} else if (block.type === "toolCall") {
					chunks.push(`[Tool call ${block.name}(${JSON.stringify(block.arguments)})]`);
				}
			}
			if (chunks.length > 0) historyLines.push(`[Assistant]:\n${chunks.join("\n")}`);
		} else if (msg.role === "toolResult") {
			const status = msg.isError ? "error" : "ok";
			const body = renderContentParts(msg.content);
			historyLines.push(`[Tool result ${msg.toolName} (${status})]:\n${body}`);
		}
	}

	const blocks: AcpPromptBlock[] = [];
	if (historyLines.length > 0) {
		blocks.push({
			type: "text",
			text: `<${contextTag}>\n${historyLines.join("\n\n")}\n</${contextTag}>`,
		});
	}

	if (latestUser) blocks.push(...userMessageToPromptBlocks(latestUser));

	return blocks.length > 0 ? blocks : null;
}

export interface PostAssistantTail {
	toolResults: ToolResultMessage[];
	steeringTexts: string[];
	steeringImages: ImageContent[];
}

/**
 * Extracts all `toolResult` messages and any interleaved/trailing user steering
 * messages appearing after the last assistant message in the transcript.
 */
export function extractPostAssistantTail(messages: readonly Message[]): PostAssistantTail | null {
	const lastAssistantIdx = messages.findLastIndex((m) => m.role === "assistant");
	if (lastAssistantIdx < 0) return null;

	const toolResults: ToolResultMessage[] = [];
	const steeringTexts: string[] = [];
	const steeringImages: ImageContent[] = [];

	for (let i = lastAssistantIdx + 1; i < messages.length; i++) {
		const msg = messages[i];
		if (msg.role === "toolResult") {
			toolResults.push(msg);
		} else if (msg.role === "user") {
			if (typeof msg.content === "string") {
				if (msg.content.trim()) steeringTexts.push(msg.content);
			} else {
				for (const part of msg.content) {
					if (part.type === "text" && part.text.trim()) steeringTexts.push(part.text);
					else if (part.type === "image") steeringImages.push(part);
				}
			}
		}
	}

	return { toolResults, steeringTexts, steeringImages };
}
