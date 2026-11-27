import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { test, type TestContext } from "node:test";
import { RpcClient, type JsonAgentSessionEvent } from "@earendil-works/pi-coding-agent";

const LIVE = process.env.PI_AGY_ACP_LIVE === "1";
const root = resolve(import.meta.dirname, "..");
const MODEL = "gemini-3.8-flash";
const TURN_MS = 120_000;

/** Real Pi in RPC mode with only this extension, on a fresh agent folder; the end of its debug log is printed afterwards. */
async function startPi(t: TestContext) {
	const dir = mkdtempSync(join(tmpdir(), "pi-agy-native-live-rpc-"));
	const agentDir = join(dir, "agent");
	const client = new RpcClient({
		cliPath: join(root, "node_modules/@earendil-works/pi-coding-agent/dist/cli.js"),
		cwd: dir,
		env: { PI_CODING_AGENT_DIR: agentDir, PI_AGY_ACP_DEBUG: "1" },
		provider: "agy-native",
		model: MODEL,
		args: ["--no-session", "--no-extensions", "--no-skills", "--no-context-files", "-e", join(root, "src/index.ts")],
	});
	const events: JsonAgentSessionEvent[] = [];
	client.onEvent((event) => events.push(event));
	t.after(async () => {
		await client.stop();
		const log = join(agentDir, "cache/agy-native/debug.log");
		if (existsSync(log)) process.stderr.write(`--- ${t.name}\n${readFileSync(log, "utf8").split("\n").slice(-40).join("\n")}\n`);
		// agy may still be writing to its gemini dir for a moment after SIGTERM.
		rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 300 });
	});
	await client.start();

	const waitFor = (match: (event: JsonAgentSessionEvent) => boolean, from = events.length) =>
		new Promise<JsonAgentSessionEvent>((resolvePromise, rejectPromise) => {
			const found = events.slice(from).find(match);
			if (found) return resolvePromise(found);
			const timer = setTimeout(() => {
				stop();
				rejectPromise(new Error(`timed out; events: ${events.map((e) => e.type).join(" ")}`));
			}, TURN_MS);
			const stop = client.onEvent((event) => {
				if (!match(event)) return;
				clearTimeout(timer);
				stop();
				resolvePromise(event);
			});
		});
	const settled = (from = events.length) => waitFor((event) => event.type === "agent_settled", from);
	const lastAssistant = async () => {
		const messages = await client.getMessages();
		const last = messages.filter((m) => m.role === "assistant").at(-1);
		assert.ok(last && last.role === "assistant");
		const text = last.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		return { message: last, text };
	};
	return { client, events, waitFor, settled, lastAssistant, dir };
}

test("live pi: Esc while Agy runs a command, then a new message gets a clean reply", { skip: !LIVE, timeout: 4 * TURN_MS }, async (t) => {
	const pi = await startPi(t);
	const start = pi.events.length;
	await pi.client.prompt("Run the shell command: sleep 30; echo slept-7731 . Then reply with the word done.");
	await pi.waitFor((e) => e.type === "tool_execution_start" && e.toolName === "agy_command", start);
	await new Promise((r) => setTimeout(r, 2_000));
	await pi.client.abort();
	await pi.settled(start);

	const next = pi.events.length;
	await pi.client.prompt("Reply with only the word PELICAN.");
	await pi.settled(next);
	const { message, text } = await pi.lastAssistant();
	assert.equal(message.stopReason, "stop", message.errorMessage);
	assert.match(text, /PELICAN/);
	assert.doesNotMatch(text, /context canceled|cancelled by the client|slept-7731/i);
});

test("live pi: a message typed while Agy runs a command steers the same turn", { skip: !LIVE, timeout: 3 * TURN_MS }, async (t) => {
	const pi = await startPi(t);
	const start = pi.events.length;
	await pi.client.prompt("Run the shell command: sleep 8; echo step-4417 . Then reply with one short sentence.");
	await pi.waitFor((e) => e.type === "tool_execution_start" && e.toolName === "agy_command", start);
	await pi.client.steer("Also end your reply with the word HERON.");
	await pi.settled(start);
	const { message, text } = await pi.lastAssistant();
	assert.equal(message.stopReason, "stop", message.errorMessage);
	assert.match(text, /HERON/);
});

test("live pi: Agy's question ends the turn, and the next message answers it", { skip: !LIVE, timeout: 3 * TURN_MS }, async (t) => {
	const pi = await startPi(t);
	const start = pi.events.length;
	await pi.client.prompt("Use your ask_question tool to ask me whether I prefer red or blue. Then tell me which one I picked.");
	await pi.settled(start);
	assert.ok(pi.events.slice(start).some((e) => e.type === "tool_execution_start" && e.toolName === "agy_tool"));
	const asked = await pi.lastAssistant();
	assert.equal(asked.message.stopReason, "stop", asked.message.errorMessage);
	assert.match(asked.text, /red/i);

	const next = pi.events.length;
	await pi.client.prompt("Blue.");
	await pi.settled(next);
	const { message, text } = await pi.lastAssistant();
	assert.equal(message.stopReason, "stop", message.errorMessage);
	assert.match(text, /blue/i);
});
