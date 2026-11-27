import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
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
