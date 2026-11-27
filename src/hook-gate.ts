/**
 * The Agy `PreToolUse` hook that routes every Agy tool call through Pi.
 *
 * Agy runs hooks from `<gemini dir>/config/hooks.json` under `agy acp`
 * (https://antigravity.google/docs/hooks.md). The `pi-gate` entry runs a small
 * script that posts the hook input to the address in `PI_AGY_GATE_URL`, which
 * the provider sets only for the `agy acp` it starts. Agy started outside Pi
 * gets `allow` and keeps its own permission flow.
 */

import { randomBytes } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { dirname } from "node:path";
import { isRecord } from "./acp-wire.ts";

export const GATE_URL_ENV = "PI_AGY_GATE_URL";
export const HOOK_ENTRY_NAME = "pi-gate";
/** Agy waits this many seconds for the hook; the provider answers well before. */
export const HOOK_TIMEOUT_SECONDS = 3600;

export const HOOK_SCRIPT = `// Installed by pi-agy-acp. Posts Agy's PreToolUse input to the Pi session
// that started this agy acp process, and prints Pi's decision.
import { request } from "node:http";

const url = process.env.${GATE_URL_ENV};
let input = "";
for await (const chunk of process.stdin) input += chunk;
const print = (answer) => process.stdout.write(JSON.stringify(answer));

if (!url) {
	print({ decision: "allow" });
} else {
	try {
		const body = await new Promise((resolve, reject) => {
			const req = request(url, { method: "POST", headers: { "content-type": "application/json" } }, (res) => {
				let out = "";
				res.setEncoding("utf8");
				res.on("data", (chunk) => (out += chunk));
				res.on("end", () => (res.statusCode === 200 ? resolve(out) : reject(new Error("gate answered " + res.statusCode))));
				res.on("error", reject);
			});
			req.on("error", reject);
			req.end(input);
		});
		const answer = JSON.parse(body);
		if (typeof answer?.decision !== "string") throw new Error("gate answer has no decision");
		print(answer);
	} catch (error) {
		print({ decision: "deny", reason: "Pi permission gate failed: " + (error instanceof Error ? error.message : String(error)) });
	}
}
`;

export interface HookToolCall {
	toolName: string;
	args: Record<string, unknown>;
	toolCallId: string | undefined;
	conversationId: string | undefined;
}

export type HookDecision = { decision: "auto_approve" } | { decision: "deny"; reason: string };

export function parseHookInput(value: unknown): HookToolCall | null {
	if (!isRecord(value) || !isRecord(value.toolCall) || typeof value.toolCall.name !== "string") return null;
	const { toolCall } = value;
	return {
		toolName: toolCall.name as string,
		args: isRecord(toolCall.args) ? toolCall.args : {},
		toolCallId: typeof toolCall.id === "string" ? toolCall.id : undefined,
		conversationId: typeof value.conversationId === "string" ? value.conversationId : undefined,
	};
}

function hookCommand(nodePath: string, scriptPath: string): string {
	const quote = (text: string) => `'${text.replaceAll("'", `'\\''`)}'`;
	return `${quote(nodePath)} ${quote(scriptPath)}`;
}

export interface HookInstallOptions {
	hooksPath: string;
	scriptPath: string;
	nodePath?: string;
}

/** Writes the hook script and the `pi-gate` entry, keeping every other entry in hooks.json. */
export function installHook(options: HookInstallOptions): void {
	const { hooksPath } = options;
	mkdirSync(dirname(options.scriptPath), { recursive: true, mode: 0o700 });
	let currentScript: string | undefined;
	try {
		currentScript = readFileSync(options.scriptPath, "utf8");
	} catch {
		currentScript = undefined;
	}
	if (currentScript !== HOOK_SCRIPT) writeFileSync(options.scriptPath, HOOK_SCRIPT, { mode: 0o600 });

	let raw: string | undefined;
	try {
		raw = readFileSync(hooksPath, "utf8");
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
	}
	const config: unknown = raw === undefined || raw.trim() === "" ? {} : JSON.parse(raw);
	if (!isRecord(config)) throw new Error(`${hooksPath} does not hold a JSON object`);
	const entry = {
		PreToolUse: [{
			matcher: "*",
			hooks: [{
				type: "command",
				command: hookCommand(options.nodePath ?? process.execPath, options.scriptPath),
				timeout: HOOK_TIMEOUT_SECONDS,
			}],
		}],
	};
	if (JSON.stringify(config[HOOK_ENTRY_NAME]) === JSON.stringify(entry)) return;
	mkdirSync(dirname(hooksPath), { recursive: true });
	writeFileSync(hooksPath, `${JSON.stringify({ ...config, [HOOK_ENTRY_NAME]: entry }, null, 2)}\n`);
}

export interface GateRequest {
	call: HookToolCall;
	/** Settles when Agy stops waiting (hook killed or connection closed) before an answer. */
	closed: Promise<void>;
}

export type GateHandler = (request: GateRequest) => Promise<HookDecision>;

/** Local HTTP endpoint the hook script posts to. */
export class GateServer {
	private server: Server | null = null;
	private readonly token = randomBytes(18).toString("base64url");
	private handler: GateHandler;
	url = "";

	constructor(handler: GateHandler) {
		this.handler = handler;
	}

	async start(): Promise<string> {
		if (this.server) return this.url;
		const server = createServer((req, res) => this.handle(req, res));
		await new Promise<void>((resolvePromise, rejectPromise) => {
			server.once("error", rejectPromise);
			server.listen(0, "127.0.0.1", () => resolvePromise());
		});
		const address = server.address();
		if (!address || typeof address === "string") throw new Error("gate server has no port");
		this.server = server;
		this.url = `http://127.0.0.1:${address.port}/${this.token}`;
		return this.url;
	}

	async stop(): Promise<void> {
		const server = this.server;
		this.server = null;
		if (!server) return;
		server.closeAllConnections();
		await new Promise<void>((resolvePromise) => server.close(() => resolvePromise()));
	}

	private handle(req: IncomingMessage, res: ServerResponse): void {
		if (req.method !== "POST" || req.url !== `/${this.token}`) {
			res.writeHead(403).end();
			return;
		}
		let body = "";
		req.setEncoding("utf8");
		req.on("data", (chunk: string) => (body += chunk));
		req.on("end", () => {
			let input: unknown;
			try {
				input = JSON.parse(body);
			} catch {
				input = undefined;
			}
			const call = parseHookInput(input);
			if (!call) {
				res.writeHead(200, { "content-type": "application/json" });
				res.end(JSON.stringify({ decision: "deny", reason: "Pi could not read this hook input." }));
				return;
			}
			const closed = new Promise<void>((resolvePromise) => res.once("close", () => resolvePromise()));
			this.handler({ call, closed }).then(
				(decision) => {
					if (res.writableEnded || res.destroyed) return;
					res.writeHead(200, { "content-type": "application/json" });
					res.end(JSON.stringify(decision));
				},
				(error: unknown) => {
					if (res.writableEnded || res.destroyed) return;
					res.writeHead(200, { "content-type": "application/json" });
					res.end(JSON.stringify({ decision: "deny", reason: `Pi permission gate failed: ${String(error)}` }));
				},
			);
		});
	}
}
