import assert from "node:assert/strict";
import { test } from "node:test";
import type { AssistantMessage, Message, ToolResultMessage } from "@earendil-works/pi-ai";
import { extractPostAssistantTail, renderTranscriptDelta, renderTranscriptReplay } from "../src/prompt.ts";

const makeAssistant = (text: string, thinking?: string): AssistantMessage => ({
	role: "assistant",
	content: [
		...(thinking ? [{ type: "thinking" as const, thinking }] : []),
		{ type: "text" as const, text },
	],
	api: "agy-native",
	provider: "agy-native",
	model: "gemini-3.8-flash",
	usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
	stopReason: "stop",
	timestamp: 1,
});

test("renderTranscriptDelta forwards a lone user prompt as text and image blocks", () => {
	const messages: Message[] = [
		{
			role: "user",
			content: [
				{ type: "text", text: "look at this" },
				{ type: "image", data: "AAAA", mimeType: "image/png" },
			],
			timestamp: 1,
		},
	];
	assert.deepEqual(renderTranscriptDelta(messages), [
		{ type: "text", text: "look at this" },
		{ type: "image", data: "AAAA", mimeType: "image/png" },
	]);
});

test("renderTranscriptReplay wraps prior turns and preserves latest user images", () => {
	const messages: Message[] = [
		{ role: "user", content: "first", timestamp: 1 },
		makeAssistant("answer 1"),
		{
			role: "user",
			content: [
				{ type: "text", text: "second" },
				{ type: "image", data: "BBBB", mimeType: "image/png" },
			],
			timestamp: 2,
		},
	];
	const blocks = renderTranscriptReplay(messages)!;
	assert.equal(blocks.length, 3);
	assert.equal(blocks[0].type, "text");
	assert.match((blocks[0] as { type: "text"; text: string }).text, /<conversation_history>/);
	assert.deepEqual(blocks[1], { type: "text", text: "second" });
	assert.deepEqual(blocks[2], { type: "image", data: "BBBB", mimeType: "image/png" });
});

test("extractPostAssistantTail collects toolResults and trailing user steering", () => {
	const tr: ToolResultMessage = {
		role: "toolResult",
		toolCallId: "c1",
		toolName: "codemode",
		content: [{ type: "text", text: "42" }],
		isError: false,
		timestamp: 2,
	};
	const messages: Message[] = [
		{ role: "user", content: "start", timestamp: 1 },
		makeAssistant("calling"),
		tr,
		{ role: "user", content: "actually hurry up", timestamp: 3 },
	];
	const tail = extractPostAssistantTail(messages)!;
	assert.equal(tail.toolResults.length, 1);
	assert.equal(tail.toolResults[0].toolCallId, "c1");
	assert.deepEqual(tail.steeringTexts, ["actually hurry up"]);
});
