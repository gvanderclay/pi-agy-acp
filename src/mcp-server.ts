/**
 * Ephemeral in-memory Streamable HTTP MCP server (`node:http`) that projects
 * configured Pi tools into `agy acp` with `antigravity/alwaysLoad: true`,
 * pushes live `notifications/tools/list_changed` updates over `GET /mcp` SSE,
 * and parks `tools/call` HTTP requests while Pi executes the real Pi tool.
 */

import { randomBytes, randomUUID, timingSafeEqual } from "node:crypto";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import { isRecord } from "./acp-wire.ts";
import { fingerprintProjectedTools, type ProjectedTool } from "./tools.ts";

export const MCP_SERVER_NAME = "pi-tools";
export const MCP_TOKEN_HEADER = "x-pi-agy-token";
const MCP_PATH = "/mcp";
const MAX_BODY_BYTES = 4 * 1024 * 1024;
const MCP_PROTOCOL_VERSION = "2025-11-25";

export interface McpToolResultPayload {
	content: (TextContent | ImageContent)[];
	isError: boolean;
}

export interface ParkedMcpCall {
	callId: string;
	mcpName: string;
	piName: string;
	args: Record<string, unknown>;
	resolve: (result: McpToolResultPayload) => void;
	reject: (message: string) => void;
}

export type ParkMcpCallHandler = (call: ParkedMcpCall) => boolean;
export type ParkMcpCancelHandler = (callId: string) => void;

interface OpenPark {
	callId: string;
	res: ServerResponse;
	rpcId: number | string;
	settled: boolean;
}

/** True when a `PreToolUse` hook tool name targets our own projected MCP server. */
export function isOwnMcpToolName(toolName: string): boolean {
	return toolName.startsWith(`mcp_${MCP_SERVER_NAME}_`);
}

function constantTimeEquals(a: string, b: string): boolean {
	const bufA = Buffer.from(a);
	const bufB = Buffer.from(b);
	if (bufA.length !== bufB.length) return false;
	return timingSafeEqual(bufA, bufB);
}

function normalizeArgs(value: unknown): Record<string, unknown> {
	return isRecord(value) ? value : {};
}

function toMcpContent(parts: readonly (TextContent | ImageContent)[]): Array<Record<string, unknown>> {
	const out: Array<Record<string, unknown>> = [];
	for (const part of parts) {
		if (part.type === "text") {
			out.push({ type: "text", text: part.text });
		} else if (part.type === "image") {
			out.push({ type: "image", data: part.data, mimeType: part.mimeType });
		}
	}
	if (out.length === 0) out.push({ type: "text", text: "" });
	return out;
}

export class McpBridgeServer {
	private readonly token = randomBytes(32).toString("hex");
	private readonly sessionId = randomUUID();
	private readonly server: Server;
	private port: number | null = null;
	private tools: ProjectedTool[] = [];
	private byMcpName = new Map<string, ProjectedTool>();
	private toolsFingerprint = fingerprintProjectedTools([]);
	private readonly sseStreams = new Set<ServerResponse>();
	private readonly toolsListWaiters = new Set<() => void>();
	private readonly openParks = new Map<string, OpenPark>();
	private parkHandler: ParkMcpCallHandler | undefined;
	private parkCancelHandler: ParkMcpCancelHandler | undefined;
	private callSeq = 0;

	constructor() {
		this.server = createServer((req, res) => {
			void this.handleRequest(req, res);
		});
		this.server.keepAliveTimeout = 0;
		this.server.headersTimeout = 0;
		this.server.requestTimeout = 0;
	}

	get url(): string {
		if (this.port === null) throw new Error("McpBridgeServer is not listening");
		return `http://127.0.0.1:${this.port}${MCP_PATH}`;
	}

	get authToken(): string {
		return this.token;
	}

	get openParkCount(): number {
		return this.openParks.size;
	}

	async start(): Promise<string> {
		if (this.port !== null) return this.url;
		await new Promise<void>((resolve, reject) => {
			const onError = (error: Error) => {
				this.server.off("listening", onListening);
				reject(error);
			};
			const onListening = () => {
				this.server.off("error", onError);
				const addr = this.server.address();
				if (!addr || typeof addr === "string") {
					reject(new Error("McpBridgeServer bound to unexpected address"));
					return;
				}
				this.port = addr.port;
				resolve();
			};
			this.server.once("error", onError);
			this.server.once("listening", onListening);
			this.server.listen(0, "127.0.0.1");
		});
		return this.url;
	}

	setParkHandler(handler: ParkMcpCallHandler | undefined, onCancelled?: ParkMcpCancelHandler): void {
		this.parkHandler = handler;
		this.parkCancelHandler = onCancelled;
	}

	async updateTools(nextTools: ProjectedTool[], waitTimeoutMs = 1_500): Promise<boolean> {
		const nextFingerprint = fingerprintProjectedTools(nextTools);
		if (nextFingerprint === this.toolsFingerprint) return false;

		this.tools = [...nextTools];
		this.byMcpName = new Map(this.tools.map((tool) => [tool.mcpName, tool]));
		this.toolsFingerprint = nextFingerprint;

		if (this.sseStreams.size === 0) return true;

		const waitPromise = new Promise<void>((resolve) => {
			let settled = false;
			const finish = () => {
				if (settled) return;
				settled = true;
				clearTimeout(timer);
				this.toolsListWaiters.delete(finish);
				resolve();
			};
			const timer = setTimeout(finish, waitTimeoutMs);
			timer.unref();
			this.toolsListWaiters.add(finish);
		});

		const frame = `event: message\ndata: ${JSON.stringify({
			jsonrpc: "2.0",
			method: "notifications/tools/list_changed",
		})}\n\n`;
		for (const stream of [...this.sseStreams]) {
			if (!stream.writableEnded) {
				try {
					stream.write(frame);
				} catch {
					this.sseStreams.delete(stream);
				}
			}
		}

		await waitPromise;
		return true;
	}

	failAllParks(message: string): void {
		for (const park of [...this.openParks.values()]) {
			this.finishPark(park, {
				content: [{ type: "text", text: message }],
				isError: true,
			});
		}
	}

	async stop(): Promise<void> {
		this.failAllParks("Pi ACP MCP bridge stopped.");
		for (const stream of [...this.sseStreams]) {
			try {
				stream.end();
			} catch {
				// Best effort.
			}
		}
		this.sseStreams.clear();
		for (const waiter of [...this.toolsListWaiters]) waiter();
		this.toolsListWaiters.clear();
		if (this.port === null) return;
		this.port = null;
		this.server.closeAllConnections();
		await new Promise<void>((resolve) => this.server.close(() => resolve()));
	}

	private isAuthorized(req: IncomingMessage): boolean {
		const header = req.headers[MCP_TOKEN_HEADER];
		if (typeof header !== "string") return false;
		return constantTimeEquals(header, this.token);
	}

	private async handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
		const pathname = new URL(req.url ?? "/", "http://127.0.0.1").pathname;

		// `agy` probes OAuth discovery paths without custom headers; 404 is expected.
		if (pathname.startsWith("/.well-known/")) {
			res.writeHead(404).end();
			return;
		}

		if (!this.isAuthorized(req)) {
			res.writeHead(401, { "content-type": "application/json" });
			res.end(JSON.stringify({ error: "Unauthorized" }));
			return;
		}

		if (pathname !== MCP_PATH) {
			res.writeHead(404).end();
			return;
		}

		if (req.method === "GET") {
			req.setTimeout(0);
			res.setTimeout(0);
			res.writeHead(200, {
				"content-type": "text/event-stream",
				"cache-control": "no-cache",
				connection: "keep-alive",
				"mcp-session-id": this.sessionId,
			});
			res.flushHeaders();
			this.sseStreams.add(res);
			res.on("close", () => this.sseStreams.delete(res));
			return;
		}

		if (req.method === "DELETE") {
			res.writeHead(200, { "content-length": "0", "mcp-session-id": this.sessionId }).end();
			return;
		}

		if (req.method !== "POST") {
			res.writeHead(405).end();
			return;
		}

		let raw: string;
		try {
			raw = await this.readBody(req);
		} catch (error) {
			res.writeHead(413, { "content-type": "application/json" });
			res.end(JSON.stringify({ error: error instanceof Error ? error.message : String(error) }));
			return;
		}

		let parsed: unknown;
		try {
			parsed = JSON.parse(raw);
		} catch {
			res.writeHead(400, { "content-type": "application/json" });
			res.end(JSON.stringify({ error: "Invalid JSON" }));
			return;
		}

		if (!isRecord(parsed) || typeof parsed.method !== "string") {
			res.writeHead(400, { "content-type": "application/json" });
			res.end(JSON.stringify({ error: "Invalid JSON-RPC request" }));
			return;
		}

		const id = parsed.id;
		const isNotification = typeof id !== "number" && typeof id !== "string";
		if (isNotification) {
			res.writeHead(202, { "content-length": "0", "mcp-session-id": this.sessionId }).end();
			return;
		}

		const method = parsed.method;
		if (method === "initialize") {
			this.replyResult(res, id, {
				protocolVersion: MCP_PROTOCOL_VERSION,
				capabilities: { tools: { listChanged: true } },
				serverInfo: { name: MCP_SERVER_NAME, version: "0.2.0" },
			});
			return;
		}

		if (method === "tools/list") {
			this.replyResult(res, id, {
				tools: this.tools.map((tool) => ({
					name: tool.mcpName,
					description: tool.description,
					inputSchema: tool.inputSchema,
					_meta: { "antigravity/alwaysLoad": true },
				})),
			});
			for (const waiter of [...this.toolsListWaiters]) waiter();
			this.toolsListWaiters.clear();
			return;
		}

		if (method === "tools/call") {
			this.handleToolCall(req, res, id, parsed.params);
			return;
		}

		this.replyError(res, id, -32601, `Method not found: ${method}`);
	}

	private handleToolCall(req: IncomingMessage, res: ServerResponse, rpcId: number | string, params: unknown): void {
		const record = isRecord(params) ? params : {};
		const mcpName = typeof record.name === "string" ? record.name : "";
		const projected = this.byMcpName.get(mcpName);
		if (!projected) {
			this.replyResult(res, rpcId, {
				isError: true,
				content: [{ type: "text", text: `Unknown Pi tool: ${mcpName || "(empty)"}` }],
			});
			return;
		}

		const handler = this.parkHandler;
		if (!handler) {
			this.replyResult(res, rpcId, {
				isError: true,
				content: [{ type: "text", text: "No active Pi turn is accepting tool calls." }],
			});
			return;
		}

		req.setTimeout(0);
		res.setTimeout(0);

		const callId = `agy_mcp_${++this.callSeq}_${randomUUID().slice(0, 8)}`;
		const open: OpenPark = { callId, res, rpcId, settled: false };
		this.openParks.set(callId, open);

		res.on("close", () => {
			if (!open.settled && !res.writableEnded) {
				open.settled = true;
				this.openParks.delete(callId);
				this.parkCancelHandler?.(callId);
			}
		});

		const parked: ParkedMcpCall = {
			callId,
			mcpName: projected.mcpName,
			piName: projected.piName,
			args: normalizeArgs(record.arguments),
			resolve: (result) => this.finishPark(open, result),
			reject: (message) =>
				this.finishPark(open, {
					content: [{ type: "text", text: message }],
					isError: true,
				}),
		};

		let accepted = false;
		try {
			accepted = handler(parked);
		} catch (error) {
			this.finishPark(open, {
				content: [{ type: "text", text: error instanceof Error ? error.message : String(error) }],
				isError: true,
			});
			return;
		}

		if (!accepted && !open.settled) {
			this.finishPark(open, {
				content: [{ type: "text", text: "Pi declined to execute this tool call." }],
				isError: true,
			});
		}
	}

	private finishPark(park: OpenPark, result: McpToolResultPayload): void {
		if (park.settled) return;
		park.settled = true;
		this.openParks.delete(park.callId);
		if (park.res.writableEnded || park.res.destroyed) return;
		this.replyResult(park.res, park.rpcId, {
			content: toMcpContent(result.content),
			...(result.isError ? { isError: true } : {}),
		});
	}

	private replyResult(res: ServerResponse, id: number | string, result: unknown): void {
		if (res.writableEnded || res.destroyed) return;
		res.writeHead(200, {
			"content-type": "application/json",
			"mcp-session-id": this.sessionId,
		});
		res.end(JSON.stringify({ jsonrpc: "2.0", id, result }));
	}

	private replyError(res: ServerResponse, id: number | string, code: number, message: string): void {
		if (res.writableEnded || res.destroyed) return;
		res.writeHead(200, {
			"content-type": "application/json",
			"mcp-session-id": this.sessionId,
		});
		res.end(JSON.stringify({ jsonrpc: "2.0", id, error: { code, message } }));
	}

	private readBody(req: IncomingMessage): Promise<string> {
		return new Promise<string>((resolve, reject) => {
			const chunks: Buffer[] = [];
			let total = 0;
			req.on("data", (chunk: Buffer) => {
				total += chunk.length;
				if (total > MAX_BODY_BYTES) {
					reject(new Error("Request body exceeded limit"));
					req.destroy();
					return;
				}
				chunks.push(chunk);
			});
			req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
			req.on("error", reject);
		});
	}
}
