import assert from "node:assert/strict";
import { test } from "node:test";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { terminalAcpError, WarningLineFilter } from "../src/errors.ts";
import type { AcpPromptResult } from "../src/acp-wire.ts";

const warning = "⚠ RESOURCE_EXHAUSTED (code 429): Individual quota reached. Resets in 1h11m59s.";
const quotaOnly = "⚠ Individual quota reached. Resets in 124h22m23s.";
const result: AcpPromptResult = { stopReason: "end_turn", usage: { inputTokens: 0, outputTokens: 0, thoughtTokens: 0, cachedReadTokens: 0, cachedWriteTokens: 0, totalTokens: 0 } };
const message = (text: string) => ({ content: [{ type: "text", text }] }) as AssistantMessage;

test("only recognized error-only zero-generation turns are classified as terminal failures", () => {
	assert.equal(terminalAcpError(message(warning), result), warning);
	assert.equal(terminalAcpError(message(quotaOnly), result), quotaOnly);
	assert.equal(terminalAcpError(message(`⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): Failed.\n${warning}`), result)?.split("\n").length, 2);
	for (const text of ["", "⚠ Check the file before editing.", `The server reported:\n${warning}`, `\`\`\`\n${warning}\n\`\`\``]) {
		assert.equal(terminalAcpError(message(text), result), undefined);
	}
	assert.equal(terminalAcpError(message(warning), { ...result, usage: undefined }), undefined);
	assert.equal(terminalAcpError(message(warning), { ...result, usage: { ...result.usage!, outputTokens: 5 } }), undefined);
	assert.equal(terminalAcpError(message(warning), { ...result, stopReason: "cancelled" }), undefined);
});

test("WarningLineFilter withholds chunk-split retry notices and fails immediately on quota exhaustion", () => {
	const filter = new WarningLineFilter();
	const emitted: string[] = [];
	const emit = (_kind: "text" | "thinking", text: string) => emitted.push(text);

	assert.equal(filter.push("text", "Hello.\n⚠ API error ", emit), undefined);
	assert.equal(emitted.join(""), "Hello.\n");
	const err = filter.push("text", "(attempt 2): RESOURCE_EXHAUSTED (code 429): Quota reached\n", emit);
	assert.match(err ?? "", /Antigravity rate limited:.*RESOURCE_EXHAUSTED/);
	assert.equal(emitted.join(""), "Hello.\n");
});

test("WarningLineFilter suppresses final quota warning lines while letting normal warning prose pass", () => {
	const quotaFilter = new WarningLineFilter();
	const quotaEmitted: string[] = [];
	const emitQuota = (_kind: "text" | "thinking", text: string) => quotaEmitted.push(text);
	assert.equal(quotaFilter.push("text", "⚠ Individual quota ", emitQuota), undefined);
	assert.equal(quotaFilter.push("text", "reached. Resets in 2h.", emitQuota), undefined);
	assert.equal(quotaEmitted.join(""), "");
	assert.match(quotaFilter.finish(emitQuota) ?? "", /Individual quota reached/);
	assert.equal(quotaEmitted.join(""), "");

	const proseFilter = new WarningLineFilter();
	const proseEmitted: string[] = [];
	const emitProse = (_kind: "text" | "thinking", text: string) => proseEmitted.push(text);
	assert.equal(proseFilter.push("text", "⚠ Check the file before editing.", emitProse), undefined);
	assert.equal(proseFilter.finish(emitProse), undefined);
	assert.equal(proseEmitted.join(""), "⚠ Check the file before editing.");
});
