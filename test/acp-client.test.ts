import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { AgyAcpClient } from "../src/acp-client.ts";

// Answers like agy 1.5.18 with no sign-in for its --gemini-dir: initialize succeeds and
// session/new fails with ACP's auth_required error.
const SIGNED_OUT_AGY = `#!/usr/bin/env node
let buffer = "";
process.stdin.on("data", (chunk) => {
	buffer += chunk;
	let end;
	while ((end = buffer.indexOf("\\n")) >= 0) {
		const message = JSON.parse(buffer.slice(0, end));
		buffer = buffer.slice(end + 1);
		const reply = message.method === "initialize"
			? { result: { protocolVersion: 1, agentCapabilities: {}, agentInfo: { name: "Antigravity ACP Server", version: "1.5.18" }, authMethods: [{ id: "oauth-personal", name: "Log in with Google" }] } }
			: { error: { code: -32000, message: "Authentication required" } };
		process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: message.id, ...reply }) + "\\n");
	}
});
`;

test("a signed-out Agy names the command that signs in Pi's Agy", async (t) => {
	const dir = mkdtempSync(join(tmpdir(), "pi-agy-acp-signin-"));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	const binaryPath = join(dir, "agy");
	writeFileSync(binaryPath, SIGNED_OUT_AGY);
	chmodSync(binaryPath, 0o755);
	const geminiDir = join(dir, "agy-native");
	const client = new AgyAcpClient({ binaryPath, geminiDir });
	t.after(() => client.dispose());

	await assert.rejects(client.createSession({ cwd: dir }), (error: Error) => {
		assert.match(error.message, /not signed in/);
		assert.ok(error.message.includes(`${binaryPath} acp --gemini-dir ${geminiDir} --login`), error.message);
		return true;
	});
});

test("a final JSON-RPC error emitted right before process exit is delivered before close", async (t) => {
	const dir = mkdtempSync(join(tmpdir(), "pi-agy-acp-exit-"));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	const binaryPath = join(dir, "agy");
	const exitAfterReply = `#!/usr/bin/env node
let buffer = "";
process.stdin.on("data", (chunk) => {
	buffer += chunk;
	let end;
	while ((end = buffer.indexOf("\\n")) >= 0) {
		const message = JSON.parse(buffer.slice(0, end));
		buffer = buffer.slice(end + 1);
		if (message.method === "initialize") {
			process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: message.id, result: { protocolVersion: 1, agentCapabilities: {}, agentInfo: { name: "mock", version: "1.5.18" } } }) + "\\n");
		} else {
			process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: message.id, error: { code: -32001, message: "Fatal backend error before exit" } }) + "\\n", () => process.exit(1));
		}
	}
});
`;
	writeFileSync(binaryPath, exitAfterReply);
	chmodSync(binaryPath, 0o755);
	const client = new AgyAcpClient({ binaryPath, geminiDir: join(dir, "agy-native") });
	t.after(() => client.dispose());

	await assert.rejects(client.createSession({ cwd: dir }), /Fatal backend error before exit/);
});

test("sandbox flag is forwarded and async onPermissionRequest selects an explicit optionId", async (t) => {
	const dir = mkdtempSync(join(tmpdir(), "pi-agy-acp-perm-"));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	const binaryPath = join(dir, "agy");
	const argsOut = join(dir, "args.json");
	const permAgy = `#!/usr/bin/env node
import { writeFileSync } from "node:fs";
writeFileSync(${JSON.stringify(argsOut)}, JSON.stringify(process.argv.slice(2)));
let buffer = "";
let promptId;
process.stdin.on("data", (chunk) => {
	buffer += chunk;
	let end;
	while ((end = buffer.indexOf("\\n")) >= 0) {
		const message = JSON.parse(buffer.slice(0, end));
		buffer = buffer.slice(end + 1);
		if (message.method === "initialize") {
			process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: message.id, result: { protocolVersion: 1, agentCapabilities: {}, agentInfo: { name: "mock", version: "1.5.18" } } }) + "\\n");
		} else if (message.method === "session/new") {
			process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: message.id, result: { sessionId: "s-1", configOptions: [] } }) + "\\n");
		} else if (message.method === "session/prompt") {
			promptId = message.id;
			process.stdout.write(JSON.stringify({
				jsonrpc: "2.0",
				id: 99,
				method: "session/request_permission",
				params: {
					sessionId: "s-1",
					toolCall: { toolCallId: "tc-1", title: "Run git push", kind: "execute" },
					options: [
						{ optionId: "allow_once", name: "Allow once", kind: "allow_once" },
						{ optionId: "allow_always", name: "Always allow", kind: "allow_always" },
						{ optionId: "reject_once", name: "Reject", kind: "reject_once" },
					],
				},
			}) + "\\n");
		} else if (message.id === 99) {
			const selected = message.result?.outcome?.optionId ?? "none";
			process.stdout.write(JSON.stringify({
				jsonrpc: "2.0",
				id: promptId,
				result: { stopReason: selected === "allow_always" ? "end_turn" : "refusal" },
			}) + "\\n");
		}
	}
});
`;
	writeFileSync(binaryPath, permAgy);
	chmodSync(binaryPath, 0o755);

	const seen: Array<{ title: string | undefined; optionIds: string[] }> = [];
	const client = new AgyAcpClient({
		binaryPath,
		geminiDir: join(dir, "agy-native"),
		sandbox: true,
		onPermissionRequest: async (request) => {
			seen.push({ title: request.title, optionIds: request.options.map((o) => o.optionId) });
			await new Promise((resolve) => setTimeout(resolve, 10));
			return "allow_always";
		},
	});
	t.after(() => client.dispose());

	const { sessionId } = await client.createSession({ cwd: dir });
	const result = await client.prompt({ sessionId, prompt: [{ type: "text", text: "push" }] });
	assert.equal(result.stopReason, "end_turn");
	assert.deepEqual(seen, [{ title: "Run git push", optionIds: ["allow_once", "allow_always", "reject_once"] }]);
	const argv = JSON.parse(readFileSync(argsOut, "utf8")) as string[];
	assert.ok(argv.includes("--sandbox"), `expected --sandbox in ${JSON.stringify(argv)}`);
});
