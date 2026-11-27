import assert from "node:assert/strict";
import { test } from "node:test";
import {
	parseInitializeResult,
	parseNewSessionResult,
	parsePromptResult,
	parseRpcMessage,
	parseSessionUpdate,
	permissionOutcome,
} from "../src/acp-wire.ts";

test("permission answers pick a one-time option and never an always option", () => {
	const request = { toolCall: { toolCallId: "tc-1" }, options: [
		{ kind: "allow_always", optionId: "always" }, { kind: "allow_once", optionId: "once" },
		{ kind: "reject_always", optionId: "never" }, { kind: "reject_once", optionId: "no" },
	] };
	assert.deepEqual(permissionOutcome(request, true), { outcome: { outcome: "selected", optionId: "once" } });
	assert.deepEqual(permissionOutcome(request, false), { outcome: { outcome: "selected", optionId: "no" } });
	assert.deepEqual(permissionOutcome(request, "always"), { outcome: { outcome: "selected", optionId: "always" } });
	assert.deepEqual(permissionOutcome(request, "unknown"), { outcome: { outcome: "cancelled" } });
	assert.deepEqual(permissionOutcome({ options: [{ kind: "allow_always", optionId: "always" }] }, true), { outcome: { outcome: "cancelled" } });
	assert.deepEqual(permissionOutcome(null, true), { outcome: { outcome: "cancelled" } });
});

test("parseRpcMessage classifies request, notification, success response, and error response", () => {
	assert.deepEqual(parseRpcMessage({ jsonrpc: "2.0", id: 1, method: "ping", params: { ok: true } }), {
		kind: "request",
		id: 1,
		method: "ping",
		params: { ok: true },
	});
	assert.deepEqual(parseRpcMessage({ jsonrpc: "2.0", method: "session/update", params: { x: 1 } }), {
		kind: "notification",
		method: "session/update",
		params: { x: 1 },
	});
	assert.deepEqual(parseRpcMessage({ jsonrpc: "2.0", id: 2, result: { ok: 1 } }), {
		kind: "response",
		id: 2,
		outcome: { ok: true, result: { ok: 1 } },
	});
	assert.deepEqual(parseRpcMessage({ jsonrpc: "2.0", id: 3, error: { code: -32602, message: "bad" } }), {
		kind: "response",
		id: 3,
		outcome: { ok: false, error: { code: -32602, message: "bad" } },
	});
});

test("parseInitializeResult extracts capabilities and systemPrompt append support", () => {
	const info = parseInitializeResult({
		protocolVersion: 1,
		agentInfo: { name: "Antigravity ACP Server", version: "1.5.17" },
		agentCapabilities: {
			sessionCapabilities: { resume: {}, close: {} },
			_meta: { antigravity: { systemPrompt: { append: true } } },
		},
		authMethods: [{ id: "oauth-personal", name: "Google" }],
	});
	assert.equal(info.name, "Antigravity ACP Server");
	assert.equal(info.version, "1.5.17");
	assert.equal(info.supportsResume, true);
	assert.equal(info.supportsClose, true);
	assert.equal(info.supportsSystemPromptAppend, true);
	assert.equal(info.authMethods.length, 1);
});

test("parseNewSessionResult parses configOptions and flattened model variants", () => {
	const parsed = parseNewSessionResult({
		sessionId: "sess-1",
		models: {
			currentModelId: "gemini-4-neon-high",
			availableModels: [
				{ modelId: "gemini-4-neon-low", name: "Neon Low" },
				{ modelId: "gemini-4-neon-xhigh", name: "Neon Extra" },
				{ modelId: "gpt-oss-120b-medium", name: "GPT-OSS 120B" },
			],
		},
		configOptions: [
			{
				id: "model",
				category: "model",
				type: "select",
				currentValue: "gemini-4-neon",
				options: [
					{ value: "gemini-4-neon", name: "Gemini 4 Neon" },
					{ value: "gpt-oss-120b-medium", name: "GPT-OSS 120B" },
				],
			},
			{
				id: "thought_level",
				category: "thought_level",
				type: "select",
				currentValue: "high",
				options: [
					{ value: "low", name: "Low" },
					{ value: "high", name: "High" },
				],
			},
		],
	});
	assert.equal(parsed.sessionId, "sess-1");
	assert.equal(parsed.config.model?.configId, "model");
	assert.equal(parsed.config.thoughtLevel?.configId, "thought_level");
	assert.equal(parsed.config.availableModelVariants.length, 3);
});

test("parseSessionUpdate and parsePromptResult extract chunks, usage_update, and turn usage", () => {
	assert.deepEqual(
		parseSessionUpdate({
			sessionId: "s1",
			update: { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "hi" } },
		}),
		{ type: "agent_message_chunk", text: "hi" },
	);
	assert.deepEqual(
		parseSessionUpdate({
			sessionId: "s1",
			update: { sessionUpdate: "usage_update", used: 1234, size: 1048576 },
		}),
		{ type: "usage_update", used: 1234, size: 1048576 },
	);

	const result = parsePromptResult({
		stopReason: "end_turn",
		usage: {
			inputTokens: 100,
			outputTokens: 50,
			cachedReadTokens: 800,
			cachedWriteTokens: 0,
			thoughtTokens: 20,
			totalTokens: 950,
		},
	});
	assert.equal(result.stopReason, "end_turn");
	assert.deepEqual(result.usage, {
		inputTokens: 100,
		outputTokens: 50,
		cachedReadTokens: 800,
		cachedWriteTokens: 0,
		thoughtTokens: 20,
		totalTokens: 950,
	});
});
