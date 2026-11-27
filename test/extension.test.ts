import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
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
