import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { GATE_URL_ENV, GateServer, HOOK_ENTRY_NAME, installHook } from "../src/hook-gate.ts";

function tempDir(t: { after: (fn: () => void) => void }): string {
	const dir = mkdtempSync(join(tmpdir(), "pi-agy-gate-"));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	return dir;
}

/** Runs the installed hook command the way Agy does: hook input on stdin, decision on stdout. */
function runHook(hooksPath: string, input: unknown, env: Record<string, string>): Promise<unknown> {
	const config = JSON.parse(readFileSync(hooksPath, "utf8"));
	const command: string = config[HOOK_ENTRY_NAME].PreToolUse[0].hooks[0].command;
	return new Promise((resolvePromise, rejectPromise) => {
		const { [GATE_URL_ENV]: _drop, ...base } = process.env;
		const child = execFile("/bin/sh", ["-c", command], { env: { ...base, ...env } }, (error, stdout) => {
			if (error) rejectPromise(error);
			else resolvePromise(JSON.parse(stdout));
		});
		child.stdin?.end(JSON.stringify(input));
	});
}

test("installHook adds the pi-gate entry, keeps other entries and leaves an installed entry untouched", (t) => {
	const dir = tempDir(t);
	const hooksPath = join(dir, "config", "hooks.json");
	const scriptPath = join(dir, "cache dir", "gate.mjs");
	installHook({ hooksPath, scriptPath });
	writeFileSync(hooksPath, JSON.stringify({ "workmux-status": { Stop: [] }, ...JSON.parse(readFileSync(hooksPath, "utf8")) }));
	installHook({ hooksPath, scriptPath });
	const config = JSON.parse(readFileSync(hooksPath, "utf8"));
	assert.deepEqual(Object.keys(config).sort(), ["pi-gate", "workmux-status"]);
	assert.deepEqual(config["workmux-status"], { Stop: [] });
	assert.equal(config[HOOK_ENTRY_NAME].PreToolUse[0].matcher, "*");
	assert.equal(config[HOOK_ENTRY_NAME].PreToolUse[0].hooks[0].timeout, 3600);
	const written = statSync(hooksPath).mtimeMs;
	installHook({ hooksPath, scriptPath });
	assert.equal(statSync(hooksPath).mtimeMs, written);
});

test("the hook allows calls from an Agy that Pi did not start", async (t) => {
	const dir = tempDir(t);
	const hooksPath = join(dir, "hooks.json");
	installHook({ hooksPath, scriptPath: join(dir, "gate.mjs") });
	assert.deepEqual(await runHook(hooksPath, { toolCall: { name: "run_command", args: {} } }, {}), { decision: "allow" });
});

test("the hook returns Pi's decision, and denies when Pi cannot be reached", async (t) => {
	const dir = tempDir(t);
	const hooksPath = join(dir, "hooks.json");
	installHook({ hooksPath, scriptPath: join(dir, "gate.mjs") });
	const seen: unknown[] = [];
	const gate = new GateServer(async ({ call }) => {
		seen.push(call);
		return { decision: "deny", reason: "no secrets" };
	});
	const url = await gate.start();
	try {
		const input = { toolCall: { id: "tc-1", name: "run_command", args: { CommandLine: "printenv KEY" } }, conversationId: "c-1" };
		assert.deepEqual(await runHook(hooksPath, input, { [GATE_URL_ENV]: url }), { decision: "deny", reason: "no secrets" });
		assert.deepEqual(seen, [{ toolName: "run_command", args: { CommandLine: "printenv KEY" }, toolCallId: "tc-1", conversationId: "c-1" }]);
		const wrongToken = url.replace(/\/[^/]+$/, "/wrong");
		const refused = await runHook(hooksPath, input, { [GATE_URL_ENV]: wrongToken }) as { decision: string };
		assert.equal(refused.decision, "deny");
	} finally {
		await gate.stop();
	}
	const unreachable = await runHook(hooksPath, { toolCall: { name: "read_file", args: {} } }, { [GATE_URL_ENV]: url }) as { decision: string };
	assert.equal(unreachable.decision, "deny");
});
