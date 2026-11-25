import assert from "node:assert/strict";
import { test } from "node:test";
import type { AssistantMessage, Message, ToolResultMessage, UserMessage } from "@earendil-works/pi-ai";
import {
	_resetSystemPromptWarnings,
	extractLatestUserPrompt,
	extractPostAssistantTail,
	fingerprintMessages,
	prefixBeforeLatestUser,
	renderSystemPromptAppend,
	renderTranscriptReplay,
} from "../src/prompt.ts";

const makeAssistant = (text: string, thinking?: string): AssistantMessage => ({
	role: "assistant",
	content: [
		...(thinking ? [{ type: "thinking" as const, thinking }] : []),
		{ type: "text" as const, text },
	],
	api: "agy-acp",
	provider: "agy-acp",
	model: "gemini-3.8-flash",
	usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
	stopReason: "stop",
	timestamp: 1,
});

test("fingerprintMessages ignores thinking blocks and system messages", () => {
	const u1: UserMessage = { role: "user", content: "hello", timestamp: 1 };
	const a1 = makeAssistant("world", "secret thoughts 1");
	const a2 = makeAssistant("world", "completely different thoughts");
	assert.equal(fingerprintMessages([u1, a1]), fingerprintMessages([u1, a2]));
	assert.equal(
		fingerprintMessages([{ role: "system", content: "sys", timestamp: 0 }, u1, a1]),
		fingerprintMessages([u1, a1]),
	);
});

test("extractLatestUserPrompt forwards text and image blocks", () => {
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
	assert.deepEqual(extractLatestUserPrompt(messages), [
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
	assert.equal(prefixBeforeLatestUser(messages).length, 2);
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

test("renderSystemPromptAppend omits stock Pi preamble and docs while keeping project context and extension sections", () => {
	const messages: Message[] = [
		{
			role: "system",
			content: "",
			sections: {
				preamble:
					"You are a coding assistant running inside pi, a coding agent harness. Help users inspect files, run commands, and modify code.",
				tools: "<tools>\n- codemode: Execute code\n</tools>",
				rules: "<rules>\n- Be concise\n</rules>",
				docs: "<docs>\nPi documentation: /path/to/pi-coding-agent/README.md\n</docs>",
				project_context: "<project_context>\nAGENTS.md content\n</project_context>",
				skills: "<skills>\n- research\n</skills>",
				cwd: "<cwd>\n/work\n</cwd>",
				custom_ext: "<custom_ext>\nextension instructions\n</custom_ext>",
			},
			timestamp: 0,
		},
	];
	const rendered = renderSystemPromptAppend(messages);
	assert.doesNotMatch(rendered, /coding agent harness/);
	assert.doesNotMatch(rendered, /Pi documentation/);
	assert.match(rendered, /<tools>/);
	assert.match(rendered, /<rules>/);
	assert.match(rendered, /AGENTS\.md content/);
	assert.match(rendered, /<skills>/);
	assert.match(rendered, /<cwd>\n\/work\n<\/cwd>/);
	assert.match(rendered, /extension instructions/);
});

test("renderSystemPromptAppend preserves custom --system-prompt and forced prompts untouched", () => {
	const customMessages: Message[] = [
		{
			role: "system",
			content: "",
			sections: {
				preamble: "You are a specialized SQL reviewer.",
				cwd: "<cwd>\n/work\n</cwd>",
			},
			timestamp: 0,
		},
	];
	assert.equal(
		renderSystemPromptAppend(customMessages),
		"You are a specialized SQL reviewer.\n\n<cwd>\n/work\n</cwd>",
	);

	const forcedMessages: Message[] = [
		{ role: "system", content: "Summarize this conversation.", timestamp: 0 },
	];
	assert.equal(renderSystemPromptAppend(forcedMessages), "Summarize this conversation.");
});

test("renderSystemPromptAppend warns once on unrecognized preamble alongside Pi built-in sections", () => {
	_resetSystemPromptWarnings();
	const logs: string[] = [];
	const messages: Message[] = [
		{
			role: "system",
			content: "",
			sections: {
				preamble: "Future upstream Pi preamble wording.",
				tools: "<tools>\n- read\n</tools>",
				rules: "<rules>\n- Be concise\n</rules>",
				docs: "<docs>\nPi documentation\n</docs>",
			},
			timestamp: 0,
		},
	];
	const first = renderSystemPromptAppend(messages, (m) => logs.push(m));
	const second = renderSystemPromptAppend(messages, (m) => logs.push(m));
	assert.match(first, /Future upstream Pi preamble wording/);
	assert.equal(first, second);
	assert.equal(logs.length, 1);
	assert.match(logs[0], /unrecognized preamble/);
});
