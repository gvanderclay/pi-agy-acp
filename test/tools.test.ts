import assert from "node:assert/strict";
import { test } from "node:test";
import type { Tool } from "@earendil-works/pi-ai";
import { fingerprintProjectedTools, normalizeInputSchema, projectPiTools } from "../src/tools.ts";

test("projectPiTools sanitizes names, resolves collisions, and preserves piName", () => {
	const tools: Tool[] = [
		{ name: "codemode", description: "Run JS", parameters: { type: "object", properties: { code: { type: "string" } } } as never },
		{ name: "web.search", description: "Search", parameters: { type: "object", properties: {} } as never },
		{ name: "web/search", description: "Search 2", parameters: { type: "object", properties: {} } as never },
	];

	const projected = projectPiTools(tools);
	assert.equal(projected.length, 3);
	assert.deepEqual(
		projected.map((t) => ({ mcp: t.mcpName, pi: t.piName })),
		[
			{ mcp: "codemode", pi: "codemode" },
			{ mcp: "web_search", pi: "web.search" },
			{ mcp: "web_search_2", pi: "web/search" },
		],
	);
});

test("normalizeInputSchema unwraps nullable oneOf/anyOf and merges plain allOf", () => {
	const nullable = normalizeInputSchema({
		anyOf: [{ type: "null" }, { type: "object", properties: { a: { type: "string" } }, required: ["a"] }],
	});
	assert.deepEqual(nullable, {
		type: "object",
		properties: { a: { type: "string" } },
		required: ["a"],
	});

	const merged = normalizeInputSchema({
		allOf: [
			{ type: "object", properties: { a: { type: "string" } }, required: ["a"] },
			{ type: "object", properties: { b: { type: "number" } }, required: ["b"] },
		],
	});
	assert.deepEqual(merged, {
		type: "object",
		properties: { a: { type: "string" }, b: { type: "number" } },
		required: ["a", "b"],
	});

	assert.deepEqual(normalizeInputSchema(undefined), { type: "object", properties: {} });
});

test("fingerprintProjectedTools changes when tool definitions change", () => {
	const a = projectPiTools([{ name: "codemode", description: "v1", parameters: { type: "object", properties: {} } as never }]);
	const b = projectPiTools([{ name: "codemode", description: "v2", parameters: { type: "object", properties: {} } as never }]);
	assert.notEqual(fingerprintProjectedTools(a), fingerprintProjectedTools(b));
	assert.equal(fingerprintProjectedTools(a), fingerprintProjectedTools([...a]));
});
