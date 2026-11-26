/**
 * JSON-RPC 2.0 client managing an `agy acp` subprocess with an isolated
 * per-process `--gemini-dir` (sharing the SQLite conversation store across
 * restarts while isolating per-process `mcp_config.json` and `agent.md`).
 */

import { spawn, type ChildProcess, type ChildProcessWithoutNullStreams } from "node:child_process";
import {
	accessSync,
	constants as fsConstants,
	existsSync,
	mkdirSync,
	readdirSync,
	rmSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import { delimiter, join, resolve } from "node:path";
import { createInterface } from "node:readline";
import {
	ACP_PROTOCOL_VERSION,
	RPC_METHOD_NOT_FOUND,
	isRecord,
	parseConfigOptionUpdate,
	parseInitializeResult,
	parseNewSessionResult,
	parsePromptResult,
	parseRpcMessage,
	parseSessionConfig,
	parseSessionUpdate,
	type AcpAgentInfo,
	type AcpPromptBlock,
	type AcpPromptResult,
	type AcpSessionConfig,
	type AcpSessionUpdate,
	type RpcId,
} from "./acp-wire.ts";
import { MCP_SERVER_NAME, MCP_TOKEN_HEADER } from "./mcp-server.ts";

const AGENT_NAME = "pi-host";
const APP_DATA_DIR_NAME = "antigravity-cli";
const CLIENT_NAME = "pi-agy-acp";
const CLIENT_VERSION = "0.1.0";
const KILL_GRACE_MS = 3_000;
const CANCEL_GRACE_MS = 3_000;
const DEFAULT_HANDSHAKE_TIMEOUT_MS = 30_000;
const PARENT_SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"] as const;
const IS_WINDOWS = process.platform === "win32";

const PI_HOST_AGENT_MD = `---
name: ${AGENT_NAME}
description: Minimal Pi host bridge agent delegating all tool execution to Pi.
mainAgent: true
excludeDefaultComponents: true
inheritMcp: true
---
You are a coding assistant. Follow the provided system instructions and use the available tools directly when needed.
`;

export interface AcpMcpEndpoint {
	url: string;
	token: string;
}

export interface AcpSessionStartOptions {
	cwd: string;
	systemPromptAppend?: string;
	signal?: AbortSignal;
}

export interface AcpSessionResumeOptions extends AcpSessionStartOptions {
	sessionId: string;
}

export interface AcpPromptCall {
	sessionId: string;
	prompt: AcpPromptBlock[];
	onUpdate: (update: AcpSessionUpdate) => void;
	onRawUpdate?: (params: unknown) => void;
	signal?: AbortSignal;
}

/** Transport interface used by `provider.ts` (implemented by `AgyAcpClient` and test doubles). */
export interface AcpTransport {
	ensureReady(signal?: AbortSignal): Promise<AcpAgentInfo>;
	createSession(options: AcpSessionStartOptions): Promise<{ sessionId: string; config: AcpSessionConfig }>;
	resumeSession(options: AcpSessionResumeOptions): Promise<{ sessionId: string; config: AcpSessionConfig }>;
	closeSession(sessionId: string): Promise<void>;
	sessionConfig(sessionId: string): AcpSessionConfig | undefined;
	setConfigOption(sessionId: string, configId: string, value: string, signal?: AbortSignal): Promise<AcpSessionConfig>;
	prompt(request: AcpPromptCall): Promise<AcpPromptResult>;
	cancel(sessionId: string): void;
	dispose(): void;
}

export class AcpRpcError extends Error {
	readonly code: number;

	constructor(message: string, code: number) {
		super(message);
		this.name = "AcpRpcError";
		this.code = code;
	}
}

export interface AgyAcpClientOptions {
	binaryPath?: string;
	cacheDir: string;
	/** When set, exposes Pi's in-memory MCP server with `timeoutSeconds: -1`. When omitted, configures zero MCP servers. */
	mcpEndpoint?: AcpMcpEndpoint;
	/** When true, uses an isolated per-process app data directory instead of symlinking the shared SQLite store. */
	ephemeral?: boolean;
	handshakeTimeoutMs?: number;
	log?: (message: string) => void;
}

function isExecutable(filePath: string): boolean {
	try {
		accessSync(filePath, fsConstants.X_OK);
		return true;
	} catch {
		return false;
	}
}

/** Resolves the `agy` CLI binary path from explicit override, env var, PATH, or standard Homebrew locations. */
export function resolveAgyBinary(override?: string): string | undefined {
	const explicit = override?.trim() || process.env.PI_AGY_ACP_BIN?.trim();
	if (explicit) {
		return isExecutable(explicit) ? explicit : undefined;
	}

	const names = IS_WINDOWS ? ["agy.exe", "agy.cmd", "agy"] : ["agy"];
	const pathDirs = (process.env.PATH ?? "").split(delimiter).filter(Boolean);
	const searchDirs = [...pathDirs, "/opt/homebrew/bin", "/usr/local/bin"];
	const seen = new Set<string>();

	for (const dir of searchDirs) {
		if (seen.has(dir)) continue;
		seen.add(dir);
		for (const name of names) {
			const candidate = join(dir, name);
			if (isExecutable(candidate)) return candidate;
		}
	}
	return undefined;
}

function isPidAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch (error) {
		return (error as NodeJS.ErrnoException).code === "EPERM";
	}
}

let instanceSeq = 0;

function sweepDeadInstances(instancesRoot: string): void {
	let entries: string[];
	try {
		entries = readdirSync(instancesRoot);
	} catch {
		return;
	}
	for (const name of entries) {
		if (!name.startsWith("pid-")) continue;
		const pid = Number(name.slice("pid-".length).split("-", 1)[0]);
		if (!Number.isInteger(pid) || pid <= 0 || pid === process.pid) continue;
		if (isPidAlive(pid)) continue;
		try {
			rmSync(join(instancesRoot, name), { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
		} catch {
			// Best effort.
		}
	}
}

function createIsolatedGeminiDir(
	cacheDir: string,
	mcpEndpoint: AcpMcpEndpoint | undefined,
	ephemeral = false,
): string {
	const absCacheDir = resolve(cacheDir);
	const sharedAppDataDir = join(absCacheDir, APP_DATA_DIR_NAME);
	const instancesRoot = join(absCacheDir, "instances");

	if (!ephemeral) mkdirSync(sharedAppDataDir, { recursive: true, mode: 0o700 });
	mkdirSync(instancesRoot, { recursive: true, mode: 0o700 });
	sweepDeadInstances(instancesRoot);

	const instanceDir = join(instancesRoot, `pid-${process.pid}-${++instanceSeq}`);
	const agentDir = join(instanceDir, "config", "agents", AGENT_NAME);
	mkdirSync(agentDir, { recursive: true, mode: 0o700 });

	// Main sessions share the SQLite conversation store across restarts while
	// keeping `config/` isolated per process. Ephemeral aux transports get their
	// own isolated app data directory so throwaway calls never contend on SQLite.
	const instanceAppData = join(instanceDir, APP_DATA_DIR_NAME);
	if (ephemeral) {
		mkdirSync(instanceAppData, { recursive: true, mode: 0o700 });
	} else if (!existsSync(instanceAppData)) {
		symlinkSync(sharedAppDataDir, instanceAppData, IS_WINDOWS ? "junction" : "dir");
	}

	writeFileSync(join(agentDir, "agent.md"), PI_HOST_AGENT_MD, { mode: 0o600 });

	const mcpConfig = mcpEndpoint
		? {
				mcpServers: {
					[MCP_SERVER_NAME]: {
						serverUrl: mcpEndpoint.url,
						headers: { [MCP_TOKEN_HEADER]: mcpEndpoint.token },
						timeoutSeconds: -1,
					},
				},
			}
		: { mcpServers: {} };

	writeFileSync(join(instanceDir, "config", "mcp_config.json"), `${JSON.stringify(mcpConfig, null, 2)}\n`, {
		mode: 0o600,
	});

	return instanceDir;
}

const abortError = (): Error => new Error("Request was aborted");

function abortable<T>(promise: Promise<T>, signal: AbortSignal | undefined): Promise<T> {
	if (!signal) return promise;
	if (signal.aborted) return Promise.reject(abortError());
	return new Promise<T>((resolvePromise, rejectPromise) => {
		const onAbort = () => rejectPromise(abortError());
		signal.addEventListener("abort", onAbort, { once: true });
		promise.then(resolvePromise, rejectPromise).finally(() => signal.removeEventListener("abort", onAbort));
	});
}

function settleWithin(promise: Promise<unknown>, ms: number): Promise<void> {
	return new Promise((resolvePromise) => {
		const timer = setTimeout(resolvePromise, ms);
		timer.unref();
		const done = () => {
			clearTimeout(timer);
			resolvePromise();
		};
		promise.then(done, done);
	});
}

function signalTree(proc: ChildProcess, signal: NodeJS.Signals): void {
	if (proc.exitCode !== null || proc.signalCode !== null) return;
	try {
		if (proc.pid !== undefined && !IS_WINDOWS) process.kill(-proc.pid, signal);
		else proc.kill(signal);
	} catch {
		// Already gone.
	}
}

const running = new Set<ChildProcess>();

const killRunning = (): void => {
	for (const proc of running) signalTree(proc, "SIGTERM");
};

function onParentSignal(signal: NodeJS.Signals): void {
	killRunning();
	if (process.listenerCount(signal) === 1) {
		setParentHooks(false);
		process.kill(process.pid, signal);
	}
}

function setParentHooks(on: boolean): void {
	const method = on ? "on" : "off";
	process[method]("exit", killRunning);
	for (const signal of PARENT_SIGNALS) process[method](signal, onParentSignal);
}

function trackProcess(proc: ChildProcess): void {
	if (running.size === 0) setParentHooks(true);
	running.add(proc);
}

function terminateProcess(proc: ChildProcess): void {
	if (running.delete(proc) && running.size === 0) setParentHooks(false);
	signalTree(proc, "SIGTERM");
	setTimeout(() => signalTree(proc, "SIGKILL"), KILL_GRACE_MS).unref();
}

interface PendingRequest {
	resolve: (result: unknown) => void;
	reject: (error: Error) => void;
}

interface Connection {
	proc: ChildProcessWithoutNullStreams;
	instanceDir: string;
	sessions: Map<string, AcpSessionConfig>;
}

export class AgyAcpClient implements AcpTransport {
	private readonly options: AgyAcpClientOptions;
	private readonly handshakeTimeoutMs: number;
	private readonly log: (message: string) => void;
	private conn: Connection | null = null;
	private ready: Promise<AcpAgentInfo> | null = null;
	private disposed = false;
	private nextId = 1;
	private readonly pending = new Map<RpcId, PendingRequest>();
	private activeTurn: AcpPromptCall | null = null;
	private draining: Promise<void> = Promise.resolve();

	constructor(options: AgyAcpClientOptions) {
		this.options = options;
		this.handshakeTimeoutMs = options.handshakeTimeoutMs ?? DEFAULT_HANDSHAKE_TIMEOUT_MS;
		this.log = options.log ?? (() => {});
	}

	ensureReady(signal?: AbortSignal): Promise<AcpAgentInfo> {
		if (this.disposed) return Promise.reject(new Error("AgyAcpClient was disposed"));
		this.ready ??= this.handshake();
		return abortable(this.ready, signal);
	}

	async createSession(options: AcpSessionStartOptions): Promise<{ sessionId: string; config: AcpSessionConfig }> {
		await this.ensureReady(options.signal);
		const conn = this.conn;
		if (!conn) throw new Error("agy acp is not running");

		const params: Record<string, unknown> = {
			cwd: options.cwd,
			mcpServers: [],
		};
		if (options.systemPromptAppend && options.systemPromptAppend.trim().length > 0) {
			params._meta = {
				antigravity: {
					systemPrompt: { append: options.systemPromptAppend },
				},
			};
		}

		const raw = await abortable(this.request("session/new", params, this.handshakeTimeoutMs), options.signal);
		const parsed = parseNewSessionResult(raw);
		conn.sessions.set(parsed.sessionId, parsed.config);
		return parsed;
	}

	async resumeSession(options: AcpSessionResumeOptions): Promise<{ sessionId: string; config: AcpSessionConfig }> {
		await this.ensureReady(options.signal);
		const conn = this.conn;
		if (!conn) throw new Error("agy acp is not running");

		const params: Record<string, unknown> = {
			sessionId: options.sessionId,
			cwd: options.cwd,
			mcpServers: [],
		};
		if (options.systemPromptAppend && options.systemPromptAppend.trim().length > 0) {
			params._meta = {
				antigravity: {
					systemPrompt: { append: options.systemPromptAppend },
				},
			};
		}

		const raw = await abortable(this.request("session/resume", params, this.handshakeTimeoutMs), options.signal);
		const config = parseSessionConfig(raw);
		conn.sessions.set(options.sessionId, config);
		return { sessionId: options.sessionId, config };
	}

	async closeSession(sessionId: string): Promise<void> {
		const conn = this.conn;
		if (!conn) return;
		conn.sessions.delete(sessionId);
		try {
			await this.request("session/close", { sessionId }, 5_000);
		} catch {
			// Best effort: session may already be closed or server shutting down.
		}
	}

	sessionConfig(sessionId: string): AcpSessionConfig | undefined {
		return this.conn?.sessions.get(sessionId);
	}

	async setConfigOption(
		sessionId: string,
		configId: string,
		value: string,
		signal?: AbortSignal,
	): Promise<AcpSessionConfig> {
		const conn = this.conn;
		const existing = conn?.sessions.get(sessionId);
		const raw = await abortable(
			this.request("session/set_config_option", { sessionId, configId, value }, this.handshakeTimeoutMs),
			signal,
		);
		const nextConfig = parseSessionConfig(raw, existing);
		conn?.sessions.set(sessionId, nextConfig);
		return nextConfig;
	}

	async prompt(request: AcpPromptCall): Promise<AcpPromptResult> {
		if (this.activeTurn) {
			throw new Error("Another agy acp prompt is already in flight");
		}
		this.activeTurn = request;
		try {
			await this.draining;
			if (request.signal?.aborted) {
				return { stopReason: "cancelled", usage: undefined };
			}
			const response = this.request("session/prompt", {
				sessionId: request.sessionId,
				prompt: request.prompt,
			});
			const finished = response.then((raw) => parsePromptResult(raw));
			const signal = request.signal;
			if (!signal) return await finished;
			return await new Promise<AcpPromptResult>((resolvePromise, rejectPromise) => {
				const onAbort = () => {
					if (this.activeTurn === request) this.activeTurn = null;
					this.cancel(request.sessionId);
					this.draining = settleWithin(response, CANCEL_GRACE_MS);
					resolvePromise({ stopReason: "cancelled", usage: undefined });
				};
				signal.addEventListener("abort", onAbort, { once: true });
				finished.then(resolvePromise, rejectPromise).finally(() => signal.removeEventListener("abort", onAbort));
			});
		} finally {
			if (this.activeTurn === request) this.activeTurn = null;
		}
	}

	cancel(sessionId: string): void {
		this.notify("session/cancel", { sessionId });
	}

	dispose(): void {
		this.disposed = true;
		if (this.conn) this.close(this.conn, new Error("AgyAcpClient was disposed"));
	}

	private async handshake(): Promise<AcpAgentInfo> {
		const conn = this.connect();
		try {
			const raw = await this.request(
				"initialize",
				{
					protocolVersion: ACP_PROTOCOL_VERSION,
					clientCapabilities: {
						fs: { readTextFile: false, writeTextFile: false },
						terminal: false,
					},
					clientInfo: { name: CLIENT_NAME, version: CLIENT_VERSION },
				},
				this.handshakeTimeoutMs,
			);
			const info = parseInitializeResult(raw);
			if (info.protocolVersion !== ACP_PROTOCOL_VERSION) {
				throw new Error(
					`agy acp speaks ACP v${info.protocolVersion}; ${CLIENT_NAME} supports v${ACP_PROTOCOL_VERSION}.`,
				);
			}
			return info;
		} catch (error) {
			this.close(conn, error instanceof Error ? error : new Error(String(error)));
			throw error;
		}
	}

	private connect(): Connection {
		const binary = resolveAgyBinary(this.options.binaryPath);
		if (!binary) {
			throw new Error(
				"Antigravity CLI ('agy') was not found on PATH. Install it or set PI_AGY_ACP_BIN to its path.",
			);
		}
		const instanceDir = createIsolatedGeminiDir(
			this.options.cacheDir,
			this.options.mcpEndpoint,
			this.options.ephemeral,
		);
		const args = [
			"acp",
			"--stdio",
			"--agent",
			AGENT_NAME,
			"--include-default-customizations=false",
			"--dangerously-skip-permissions",
			"--disable-telemetry",
			"--gemini-dir",
			instanceDir,
			"--app-data-dir",
			APP_DATA_DIR_NAME,
		];

		const proc = spawn(binary, args, {
			env: { ...process.env },
			detached: !IS_WINDOWS,
			stdio: ["pipe", "pipe", "pipe"],
		});

		const conn: Connection = { proc, instanceDir, sessions: new Map() };
		this.conn = conn;
		trackProcess(proc);

		proc.stdin.on("error", (error) => this.log(`stdin error: ${error.message}`));
		proc.stderr.on("data", (chunk: Buffer) => this.log(`stderr: ${chunk.toString().trimEnd()}`));
		createInterface({ input: proc.stdout }).on("line", (line) => {
			if (this.conn !== conn) return;
			try {
				this.handleLine(line);
			} catch (error) {
				this.log(`line handler failed: ${String(error)}`);
			}
		});
		proc.on("error", (error: NodeJS.ErrnoException) => {
			this.close(conn, new Error(`Failed to start agy acp ('${binary}'): ${error.message}`));
		});
		proc.on("exit", (code, signal) => {
			this.close(conn, new Error(`agy acp exited unexpectedly (code ${code}, signal ${signal})`));
		});
		return conn;
	}

	private close(conn: Connection, reason: Error): void {
		if (this.conn !== conn) return;
		this.log(`closing connection: ${reason.message}`);
		this.conn = null;
		this.ready = null;
		this.draining = Promise.resolve();
		for (const entry of [...this.pending.values()]) entry.reject(reason);
		this.pending.clear();
		terminateProcess(conn.proc);
		// Remove only the per-instance config/symlink dir; shared SQLite store stays intact.
		try {
			rmSync(conn.instanceDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
		} catch {
			// Best effort (Windows or busy file handles).
		}
	}

	private write(message: object): boolean {
		const stdin = this.conn?.proc.stdin;
		if (!stdin?.writable) return false;
		stdin.write(`${JSON.stringify(message)}\n`);
		return true;
	}

	private notify(method: string, params: unknown): void {
		this.write({ jsonrpc: "2.0", method, params });
	}

	private request(method: string, params: unknown, timeoutMs?: number): Promise<unknown> {
		const id = this.nextId++;
		return new Promise<unknown>((resolvePromise, rejectPromise) => {
			let timer: NodeJS.Timeout | undefined;
			const settle = () => {
				this.pending.delete(id);
				if (timer) clearTimeout(timer);
			};
			const entry: PendingRequest = {
				resolve: (result) => {
					settle();
					resolvePromise(result);
				},
				reject: (error) => {
					settle();
					rejectPromise(error);
				},
			};
			this.pending.set(id, entry);
			if (timeoutMs !== undefined) {
				timer = setTimeout(() => entry.reject(new Error(`${method} timed out after ${timeoutMs}ms`)), timeoutMs);
				timer.unref();
			}
			if (!this.write({ jsonrpc: "2.0", id, method, params })) {
				entry.reject(new Error(`Cannot send ${method}: agy acp is not running`));
			}
		});
	}

	private handleLine(line: string): void {
		if (!line.trim()) return;
		let parsed: unknown;
		try {
			parsed = JSON.parse(line);
		} catch {
			this.log("discarded malformed NDJSON line");
			return;
		}
		const message = parseRpcMessage(parsed);
		if (!message) return;

		if (message.kind === "request") {
			this.write({
				jsonrpc: "2.0",
				id: message.id,
				error: {
					code: RPC_METHOD_NOT_FOUND,
					message: `pi-agy-acp does not handle server request '${message.method}'`,
				},
			});
			return;
		}

		if (message.kind === "notification") {
			if (message.method === "session/update") this.dispatchUpdate(message.params);
			return;
		}

		const entry = this.pending.get(message.id);
		if (!entry) return;
		if (message.outcome.ok) entry.resolve(message.outcome.result);
		else entry.reject(new AcpRpcError(message.outcome.error.message, message.outcome.error.code));
	}

	private dispatchUpdate(params: unknown): void {
		if (!isRecord(params) || typeof params.sessionId !== "string") return;
		const { sessionId } = params;
		const conn = this.conn;
		if (conn) {
			const existing = conn.sessions.get(sessionId);
			const nextConfig = existing && parseConfigOptionUpdate(params, existing);
			if (existing && nextConfig) conn.sessions.set(sessionId, nextConfig);
		}
		const turn = this.activeTurn;
		if (!turn || sessionId !== turn.sessionId) return;
		turn.onRawUpdate?.(params);
		const update = parseSessionUpdate(params);
		if (update) turn.onUpdate(update);
	}
}
