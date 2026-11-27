import assert from "node:assert/strict";
import { test } from "node:test";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { terminalAcpError } from "../src/errors.ts";
import type { AcpPromptResult } from "../src/acp-wire.ts";

const warning = "⚠ RESOURCE_EXHAUSTED (code 429): Individual quota reached. Resets in 1h11m59s.";
const result: AcpPromptResult = { stopReason: "end_turn", usage: { inputTokens: 0, outputTokens: 0, thoughtTokens: 0, cachedReadTokens: 0, cachedWriteTokens: 0, totalTokens: 0 } };
const message = (text: string) => ({ content: [{ type: "text", text }] }) as AssistantMessage;

test("only recognized error-only zero-generation turns are classified as terminal failures", () => {
	assert.equal(terminalAcpError(message(warning), result), warning);
	assert.equal(terminalAcpError(message(`⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): Failed.\n${warning}`), result)?.split("\n").length, 2);
	for (const text of ["", "⚠ Check the file before editing.", `The server reported:\n${warning}`, `\`\`\`\n${warning}\n\`\`\``]) {
		assert.equal(terminalAcpError(message(text), result), undefined);
	}
	assert.equal(terminalAcpError(message(warning), { ...result, usage: undefined }), undefined);
	assert.equal(terminalAcpError(message(warning), { ...result, usage: { ...result.usage!, outputTokens: 5 } }), undefined);
	assert.equal(terminalAcpError(message(warning), { ...result, stopReason: "cancelled" }), undefined);
});
