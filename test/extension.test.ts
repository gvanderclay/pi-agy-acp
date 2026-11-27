import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { test } from "node:test";
import { promisify } from "node:util";
import { AGY_TOOL_NAMES } from "../src/provider.ts";

const root = resolve(import.meta.dirname, "..");

test("a session on another model is never sent the Agy approval tools, even when its tool list names them", async (t) => {
	const dir = mkdtempSync(join(tmpdir(), "pi-agy-extension-"));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	const out = join(dir, "tools.jsonl");
	const run = promisify(execFile)(
		join(root, "node_modules/.bin/pi"),
		[
			"-p", "--no-session", "--no-extensions", "--no-skills", "--no-context-files",
			"-e", join(root, "src/index.ts"),
			"-e", join(root, "test/fixtures/fake-model.ts"),
			"--model", "fake/model",
			"--tools", ["read", "late_tool", ...AGY_TOOL_NAMES].join(","),
			"hi",
		],
		{ cwd: dir, env: { ...process.env, PI_CODING_AGENT_DIR: join(dir, "agent"), FAKE_MODEL_TOOLS_OUT: out, PI_OFFLINE: "1" } },
	);
	// Print mode reads a piped stdin before it starts.
	run.child.stdin?.end();
	await run;
	const requests = readFileSync(out, "utf8").trim().split("\n").map((line) => JSON.parse(line) as string[]);
	assert.deepEqual(requests, [["late_tool", "read"]]);
});

const MCP_CALLING_AGY = `#!/usr/bin/env node
import { writeFileSync } from "node:fs";

let buffer = "";
let mcpServer;
const logPath = process.env.FAKE_AGY_MCP_OUT;

function send(message) {
	process.stdout.write(JSON.stringify(message) + "\\n");
}

async function handle(message) {
	if (message.method === "initialize") {
		send({
			jsonrpc: "2.0",
			id: message.id,
			result: {
				protocolVersion: 1,
				agentCapabilities: {},
				agentInfo: { name: "mock-agy", version: "1.5.18" },
				authMethods: [],
			},
		});
		return;
	}
	if (message.method === "session/new") {
		mcpServer = message.params?.mcpServers?.[0];
		send({
			jsonrpc: "2.0",
			id: message.id,
			result: {
				sessionId: "acp-deferred-1",
				configOptions: [
					{
						id: "model",
						name: "Model",
						type: "select",
						currentValue: "gemini-4-neon",
						options: [{ value: "gemini-4-neon", name: "Gemini 4 Neon" }],
					},
				],
			},
		});
		return;
	}
	if (message.method === "session/prompt") {
		const headers = { "content-type": "application/json" };
		for (const header of mcpServer?.headers ?? []) headers[header.name] = header.value;
		const listRes = await fetch(mcpServer.url, {
			method: "POST",
			headers,
			body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/list" }),
		});
		const listJson = await listRes.json();
		const callRes = await fetch(mcpServer.url, {
			method: "POST",
			headers,
			body: JSON.stringify({
				jsonrpc: "2.0",
				id: 2,
				method: "tools/call",
				params: { name: "deferred_lookup", arguments: { key: "alpha" } },
			}),
		});
		const callJson = await callRes.json();
		writeFileSync(logPath, JSON.stringify({ listJson, callJson }));
		const text = callJson?.result?.content?.[0]?.text ?? "missing";
		send({
			jsonrpc: "2.0",
			method: "session/update",
			params: {
				sessionId: message.params.sessionId,
				update: { sessionUpdate: "agent_message_chunk", content: { type: "text", text } },
			},
		});
		send({
			jsonrpc: "2.0",
			id: message.id,
			result: { stopReason: "end_turn" },
		});
		return;
	}
	if (message.method === "session/close") {
		send({ jsonrpc: "2.0", id: message.id, result: {} });
	}
}

process.stdin.on("data", (chunk) => {
	buffer += chunk;
	let end;
	while ((end = buffer.indexOf("\\n")) >= 0) {
		const line = buffer.slice(0, end);
		buffer = buffer.slice(end + 1);
		if (line.trim()) void handle(JSON.parse(line));
	}
});
`;

test("a deferred Pi tool matching mcpTools executes end-to-end under codemode-only mode", async (t) => {
	const dir = mkdtempSync(join(tmpdir(), "pi-agy-deferred-mcp-"));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	const agentDir = join(dir, "agent");
	mkdirSync(agentDir, { recursive: true });
	writeFileSync(
		join(agentDir, "settings.json"),
		JSON.stringify({ defaultTools: ["+codemode"], codemode: { mode: "only" } }),
	);
	writeFileSync(join(agentDir, "agy-native.json"), JSON.stringify({ mcpTools: ["deferred_lookup"] }));

	const deferredExt = join(dir, "deferred-tool.mjs");
	writeFileSync(
		deferredExt,
		`export default function (pi) {
	pi.registerTool({
		name: "deferred_lookup",
		label: "Deferred lookup",
		description: "Look up a value by key",
		exposure: "deferred",
		parameters: { type: "object", properties: { key: { type: "string" } }, required: ["key"] },
		execute: async (_id, params) => ({
			content: [{ type: "text", text: "looked up: " + params.key }],
			details: {},
		}),
	});
}
`,
	);

	const fakeAgy = join(dir, "agy");
	writeFileSync(fakeAgy, MCP_CALLING_AGY);
	chmodSync(fakeAgy, 0o755);
	const mcpOut = join(dir, "mcp-out.json");

	const run = promisify(execFile)(
		join(root, "node_modules/.bin/pi"),
		[
			"-p", "--no-session", "--no-extensions", "--no-skills", "--no-context-files",
			"-e", deferredExt,
			"-e", join(root, "src/index.ts"),
			"--model", "agy-native/gemini-4-neon",
			"look up alpha",
		],
		{
			cwd: dir,
			env: {
				...process.env,
				PI_CODING_AGENT_DIR: agentDir,
				PI_AGY_ACP_BIN: fakeAgy,
				FAKE_AGY_MCP_OUT: mcpOut,
				PI_OFFLINE: "1",
			},
		},
	);
	run.child.stdin?.end();
	await run;

	const recorded = JSON.parse(readFileSync(mcpOut, "utf8")) as {
		listJson: { result: { tools: Array<{ name: string }> } };
		callJson: { result: { content: Array<{ type: string; text: string }>; isError?: boolean } };
	};
	assert.deepEqual(
		recorded.listJson.result.tools.map((tool) => tool.name),
		["deferred_lookup"],
	);
	assert.equal(recorded.callJson.result.isError, undefined);
	assert.deepEqual(recorded.callJson.result.content, [{ type: "text", text: "looked up: alpha" }]);
});
