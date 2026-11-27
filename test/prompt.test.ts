import assert from "node:assert/strict";
import { test } from "node:test";
import type { AssistantMessage, Message, ToolResultMessage, UserMessage } from "@earendil-works/pi-ai";
import {
	_resetSystemPromptWarnings,
	extractPostAssistantTail,
	fingerprintMessages,
	hashSystemPromptAppend,
	messagesAfterFingerprint,
	renderSystemPromptAppend,
	renderTranscriptDelta,
	renderTranscriptReplay,
	type HostPromptSources,
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

test("a matched conversation prefix sends an unsent tool result and preserves the next user's attachments", () => {
	const oldUser: UserMessage = { role: "user", content: "old request", timestamp: 1 };
	const oldAssistant = makeAssistant("old reply");
	const seen = fingerprintMessages([oldUser, oldAssistant]);
	const messages: Message[] = [
		{ role: "system", content: "sys", timestamp: 0 },
		oldUser,
		oldAssistant,
		{
			role: "toolResult",
			toolCallId: "call-1",
			toolName: "lookup",
			content: [{ type: "text", text: "new result" }],
			isError: false,
			timestamp: 2,
		},
		{
			role: "user",
			content: [
				{ type: "text", text: "next request" },
				{ type: "image", mimeType: "image/png", data: "AAAA" },
			],
			timestamp: 3,
		},
	];
	const delta = messagesAfterFingerprint(messages, seen);
	assert.ok(delta);
	const blocks = renderTranscriptDelta(delta);
	assert.ok(blocks);
	assert.deepEqual(blocks, [
		{ type: "text", text: "<conversation_context>\n[Tool result lookup (ok)]:\nnew result\n</conversation_context>" },
		{ type: "text", text: "next request" },
		{ type: "image", mimeType: "image/png", data: "AAAA" },
	]);
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
	assert.doesNotMatch(rendered, /<tools>/);
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

test("host append uses explicit rule sources and preserves project, role, skills and custom sections", () => {
	const sources: HostPromptSources = {
		selectedTools: ["codemode"], toolGuidelines: { codemode: ["Use the tool's namespace."] },
		promptGuidelines: ["Keep the task's acceptance checks."], sections: {},
	};
	const messages: Message[] = [{ role: "system", timestamp: 0, content: "", sections: {
		preamble: "You are a coding assistant running inside pi, a coding agent harness.",
		tools: "<tools>\nGeneric descriptions\n</tools>",
		rules: "<rules>\n- Be concise in your responses\n</rules>",
		docs: "<docs>\nPi documentation: /pi/README.md\n</docs>",
		addendum: "<addendum>\nExplorer role\n</addendum>",
		project_context: "<project_context>\nKeep references to pi-coding-agent in this project.\n</project_context>",
		skills: "<skills>\nResearch skill\n</skills>",
		permission_rules: "<permission_rules>\nDo not write outside the workspace.\n</permission_rules>",
	} }];
	const rendered = renderSystemPromptAppend(messages, undefined, sources);
	assert.doesNotMatch(rendered, /coding agent harness|Generic descriptions|Be concise|Pi documentation/);
	for (const text of ["Use the tool's namespace.", "acceptance checks", "Explorer role", "pi-coding-agent", "Research skill", "Do not write outside"]) assert.ok(rendered.includes(text));
	const hash = hashSystemPromptAppend(messages, sources);
	sources.promptGuidelines.push("New extension requirement.");
	assert.notEqual(hashSystemPromptAppend(messages, sources), hash);
	sources.sections.tools = "Explicit tool instructions";
	const customToolsHash = hashSystemPromptAppend(messages, sources);
	assert.notEqual(hashSystemPromptAppend([...messages, { role: "system", content: "", timestamp: 1, sections: { tools: "Revised explicit tool instructions" } }], sources), customToolsHash);
	sources.sections.docs = "Pi documentation deliberately supplied by an extension";
	assert.match(renderSystemPromptAppend(messages, undefined, sources), /Pi documentation/);
});

test("flattened stock prompts fail locally instead of deleting or forwarding unknown instructions", () => {
	const content = "You are a coding assistant running inside pi, a coding agent harness.\n\n<docs>\nPi documentation\n</docs>\n\nImportant delegate instructions";
	assert.throws(() => renderSystemPromptAppend([{ role: "system", content, timestamp: 0 }]), /Flattened stock Pi system prompt/);
	const custom = "Review this literal example: <docs>Pi documentation</docs>. Keep pi-coding-agent names intact.";
	assert.equal(renderSystemPromptAppend([{ role: "system", content: custom, timestamp: 0 }]), custom);
});

test("append continuity includes custom prompt text and non-tool sections", () => {
	const system: Message = {
		role: "system",
		content: "Custom instructions with a literal <tools>example</tools>.",
		sections: { tools: "Pi tool documentation", project_context: "Project A", extra: "Extension A" },
		timestamp: 0,
	};
	const hash = hashSystemPromptAppend([system]);
	assert.equal(hashSystemPromptAppend([system, {
		role: "system", content: "", sections: { tools: null }, timestamp: 1,
	}]), hash);
	assert.notEqual(hashSystemPromptAppend([{ ...system, content: "Custom instructions with a literal <tools>edit</tools>." }]), hash);
	assert.notEqual(hashSystemPromptAppend([system, {
		role: "system", content: "", sections: { project_context: "Project B" }, timestamp: 1,
	}]), hash);
	assert.notEqual(hashSystemPromptAppend([system, {
		role: "system", content: "", sections: { extra: "Extension B" }, timestamp: 1,
	}]), hash);
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
