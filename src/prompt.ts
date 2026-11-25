/**
 * Transcript fingerprinting and prompt rendering between Pi's `TranscriptContext`
 * and ACP `session/prompt` content blocks.
 */

import { createHash } from "node:crypto";
import {
	getCurrentSystemMessage,
	getSystemMessageText,
	type ImageContent,
	type Message,
	type SystemMessage,
	type TextContent,
	type ToolResultMessage,
	type UserMessage,
} from "@earendil-works/pi-ai";
import type { AcpPromptBlock } from "./acp-wire.ts";

function hashText(text: string): string {
	return createHash("sha256").update(text).digest("hex").slice(0, 16);
}

function hashImage(img: ImageContent): string {
	const digest = createHash("sha256").update(img.data).digest("hex").slice(0, 12);
	return `[image:${img.mimeType}:${digest}]`;
}

function summarizeUserContent(content: UserMessage["content"]): string {
	if (typeof content === "string") return content;
	return content
		.map((block) => (block.type === "text" ? block.text : hashImage(block)))
		.join("\n");
}

function summarizeMessageForFingerprint(message: Message): string {
	if (message.role === "user") {
		return `u:${summarizeUserContent(message.content)}`;
	}
	if (message.role === "assistant") {
		const parts = message.content.map((block) => {
			if (block.type === "text") return `t:${block.text}`;
			if (block.type === "toolCall") return `c:${block.id}:${block.name}:${JSON.stringify(block.arguments)}`;
			if (block.type === "thinking") return "";
			return "";
		});
		return `a:${parts.filter(Boolean).join("|")}`;
	}
	if (message.role === "toolResult") {
		const parts = message.content.map((block) => (block.type === "text" ? block.text : hashImage(block)));
		return `r:${message.toolCallId}:${message.toolName}:${message.isError ? "err" : "ok"}:${parts.join("|")}`;
	}
	return "";
}

/**
 * Deterministic fingerprint of the non-system conversation messages.
 * Ignores thinking blocks and system messages so only semantic conversation
 * turns affect continuity checks.
 */
export function fingerprintMessages(messages: readonly Message[]): string {
	const hash = createHash("sha256");
	let count = 0;
	for (const message of messages) {
		if (message.role === "system") continue;
		count++;
		hash.update(summarizeMessageForFingerprint(message));
		hash.update("\0");
	}
	return `m${count}-${hash.digest("hex").slice(0, 16)}`;
}

/** Returns the unsent tail, or null if the ACP session's prefix no longer matches. */
export function messagesAfterFingerprint(messages: readonly Message[], fingerprint: string): Message[] | null {
	const match = /^m(\d+)-[0-9a-f]{16}$/.exec(fingerprint);
	if (!match) return null;
	const count = Number(match[1]);
	const conversation = messages.filter((message) => message.role !== "system");
	if (count > conversation.length || fingerprintMessages(conversation.slice(0, count)) !== fingerprint) return null;
	return conversation.slice(count);
}

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
 * Renders full Pi conversation history into ACP prompt blocks for cold starts,
 * `/fork`, `/tree` rewinds, compaction, or cross-provider switches.
 * Preserves images attached to the final user message.
 */
export function renderTranscriptReplay(messages: readonly Message[]): AcpPromptBlock[] | null {
	const nonSystem = messages.filter((m) => m.role !== "system");
	const lastMsg = nonSystem.at(-1);
	const latestUser = lastMsg?.role === "user" ? lastMsg : undefined;
	const prior = latestUser ? nonSystem.slice(0, -1) : nonSystem;
	return renderTranscriptPrompt(prior, latestUser, "conversation_history");
}

/** Sends only conversation messages missing from a warm ACP session. */
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

const PI_DEFAULT_PREAMBLE_PREFIXES = [
	"You are a coding assistant running inside pi, a coding agent harness.",
	"You are an expert coding assistant operating inside pi, a coding agent harness.",
] as const;

const PI_OWNED_SECTIONS = ["tools", "rules", "docs"] as const;
const PI_DOCS_ANCHORS = ["Pi documentation", "pi-coding-agent"] as const;

let warnedPreambleDrift = false;

/** Reset one-time drift warning latch (for tests). */
export function _resetSystemPromptWarnings(): void {
	warnedPreambleDrift = false;
}

function isDefaultPiPreamble(text: string): boolean {
	return PI_DEFAULT_PREAMBLE_PREFIXES.some((prefix) => text.startsWith(prefix));
}

function isPiDocsSection(text: string): boolean {
	return PI_DOCS_ANCHORS.some((anchor) => text.includes(anchor));
}

function systemContentText(content: string | TextContent[]): string {
	if (typeof content === "string") return content;
	return content.map((part) => part.text).join("");
}

/**
 * Renders Pi's effective system prompt for ACP `_meta.antigravity.systemPrompt.append`.
 *
 * When Pi is using its stock structured system prompt, omits the redundant `preamble`
 * section (since `pi-host/agent.md` already supplies the base agent preamble) and Pi's
 * harness `docs` section, while preserving `tools`, `rules`, `addendum`,
 * `project_context` (`AGENTS.md`), `skills`, `cwd`, and all extension sections verbatim.
 *
 * Custom `--system-prompt`, forced prompts, and auxiliary prompts pass through untouched.
 */
export function renderSystemPromptAppend(messages: readonly Message[], log?: (message: string) => void): string {
	const sys = getCurrentSystemMessage(messages);
	return sys ? renderSystemAppend(sys, log) : "";
}

export function hashSystemPromptAppend(messages: readonly Message[]): string {
	const sys = getCurrentSystemMessage(messages);
	if (!sys) return hashText("");

	// ACP learns tool changes independently via MCP notifications/tools/list_changed.
	const { tools: _tools, ...sections } = sys.sections ?? {};
	return hashText(renderSystemAppend({ ...sys, sections: Object.keys(sections).length > 0 ? sections : undefined }));
}

function renderSystemAppend(sys: SystemMessage, log?: (message: string) => void): string {
	const sections = sys.sections;
	if (sections) {
		const preamble = sections.preamble;
		if (typeof preamble === "string" && isDefaultPiPreamble(preamble)) {
			const parts: string[] = [];
			const baseContent = systemContentText(sys.content);
			if (baseContent.length > 0) parts.push(baseContent);

			for (const [name, value] of Object.entries(sections)) {
				if (value === null || value.length === 0) continue;
				if (name === "preamble") continue;
				if (name === "docs" && isPiDocsSection(value)) continue;
				parts.push(value);
			}
			return parts.join("\n\n");
		}

		// Drift detection: if all built-in Pi sections exist alongside an unrecognized
		// preamble, Pi may have reworded its default preamble upstream.
		if (
			!warnedPreambleDrift &&
			typeof preamble === "string" &&
			PI_OWNED_SECTIONS.every((name) => typeof sections[name] === "string")
		) {
			warnedPreambleDrift = true;
			log?.(
				"Pi system prompt has built-in sections (tools/rules/docs) with an unrecognized preamble; passing through untouched.",
			);
		}
		return getSystemMessageText(sys);
	}

	const rendered = getSystemMessageText(sys);
	if (isDefaultPiPreamble(rendered)) {
		const splitIdx = rendered.indexOf("\n\n");
		return splitIdx >= 0 ? rendered.slice(splitIdx + 2) : "";
	}
	return rendered;
}
