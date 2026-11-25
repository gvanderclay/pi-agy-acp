import assert from "node:assert/strict";
import http from "node:http";
import { test } from "node:test";
import { McpBridgeServer, MCP_TOKEN_HEADER, type ParkedToolCall } from "../src/mcp-server.ts";
import { projectPiTools } from "../src/tools.ts";

async function postJson(url: string, headers: Record<string, string>, body: unknown) {
	const res = await fetch(url, {
		method: "POST",
		headers: { "content-type": "application/json", ...headers },
		body: JSON.stringify(body),
	});
	const text = await res.text();
	return { status: res.status, body: text ? JSON.parse(text) : null };
}

test("McpBridgeServer enforces auth, serves tools with alwaysLoad, and pushes list_changed over SSE", async () => {
	const server = new McpBridgeServer();
	const url = await server.start();
	const auth = { [MCP_TOKEN_HEADER]: server.authToken };

	try {
		// Unauthenticated POST is rejected; well-known probe returns 404.
		const unauth = await fetch(url, { method: "POST", body: "{}" });
		assert.equal(unauth.status, 401);
		const wellKnown = await fetch(url.replace("/mcp", "/.well-known/oauth-protected-resource"));
		assert.equal(wellKnown.status, 404);

		// Initialize & notifications/initialized
		const init = await postJson(url, auth, { jsonrpc: "2.0", id: 1, method: "initialize", params: {} });
		assert.equal(init.status, 200);
		assert.equal(init.body.result.capabilities.tools.listChanged, true);

		const notif = await postJson(url, auth, { jsonrpc: "2.0", method: "notifications/initialized", params: {} });
		assert.equal(notif.status, 202);

		// Set initial tools and verify tools/list includes antigravity/alwaysLoad
		await server.updateTools(
			projectPiTools([{ name: "codemode", description: "Run JS", parameters: { type: "object", properties: {} } as never }]),
		);
		const list1 = await postJson(url, auth, { jsonrpc: "2.0", id: 2, method: "tools/list", params: {} });
		assert.equal(list1.body.result.tools.length, 1);
		assert.deepEqual(list1.body.result.tools[0]._meta, { "antigravity/alwaysLoad": true });

		// Open GET /mcp SSE stream and verify notifications/tools/list_changed push
		const sseReceived = new Promise<string>((resolve, reject) => {
			const req = http.get(url, { headers: { ...auth, accept: "text/event-stream" } }, (res) => {
				assert.equal(res.statusCode, 200);
				let buf = "";
				res.on("data", (chunk: Buffer) => {
					buf += chunk.toString("utf8");
					if (buf.includes("notifications/tools/list_changed")) {
						req.destroy();
						resolve(buf);
					}
				});
			});
			req.on("error", (err: NodeJS.ErrnoException) => {
				if (err.code !== "ECONNRESET") reject(err);
			});
		});

		// Wait briefly for GET /mcp to register in sseStreams
		await new Promise((r) => setTimeout(r, 50));
		assert.equal(server.activeSseCount, 1);

		const updatePromise = server.updateTools(
			projectPiTools([
				{ name: "codemode", description: "Run JS", parameters: { type: "object", properties: {} } as never },
				{ name: "web_search", description: "Search", parameters: { type: "object", properties: {} } as never },
			]),
			500,
		);

		const sseFrame = await sseReceived;
		assert.match(sseFrame, /notifications\/tools\/list_changed/);
		// Simulate client responding with tools/list
		const list2 = await postJson(url, auth, { jsonrpc: "2.0", id: 3, method: "tools/list", params: {} });
		await updatePromise;
		assert.equal(list2.body.result.tools.length, 2);
	} finally {
		await server.stop();
	}
});

test("McpBridgeServer parks tools/call and resolves text, images, isError, and client disconnects", async () => {
	const server = new McpBridgeServer();
	const url = await server.start();
	const auth = { [MCP_TOKEN_HEADER]: server.authToken };

	try {
		await server.updateTools(
			projectPiTools([{ name: "codemode", description: "Run JS", parameters: { type: "object", properties: {} } as never }]),
		);

		let parked: ParkedToolCall | undefined;
		const cancelledIds: string[] = [];
		server.setParkHandler(
			(call) => {
				parked = call;
				return true;
			},
			(id) => cancelledIds.push(id),
		);

		// 1. Resolve with text + image + isError
		const callPromise = postJson(url, auth, {
			jsonrpc: "2.0",
			id: 10,
			method: "tools/call",
			params: { name: "codemode", arguments: { code: "return 42" } },
		});

		await new Promise((r) => setTimeout(r, 30));
		assert.ok(parked);
		assert.equal(parked.piName, "codemode");
		assert.deepEqual(parked.args, { code: "return 42" });

		parked.resolve({
			content: [
				{ type: "text", text: "result=42" },
				{ type: "image", data: "AAAA", mimeType: "image/png" },
			],
			isError: true,
		});

		const res = await callPromise;
		assert.equal(res.status, 200);
		assert.equal(res.body.result.isError, true);
		assert.equal(res.body.result.content.length, 2);
		assert.equal(res.body.result.content[0].text, "result=42");
		assert.equal(res.body.result.content[1].type, "image");

		// 2. Client abort triggers onCancelled
		parked = undefined;
		const ac = new AbortController();
		const abortedCall = fetch(url, {
			method: "POST",
			headers: { "content-type": "application/json", ...auth },
			body: JSON.stringify({ jsonrpc: "2.0", id: 11, method: "tools/call", params: { name: "codemode", arguments: {} } }),
			signal: ac.signal,
		}).catch(() => undefined);

		await new Promise((r) => setTimeout(r, 30));
		const second = parked as { callId: string } | undefined;
		assert.ok(second);
		const callId = second.callId;
		ac.abort();
		await abortedCall;
		await new Promise((r) => setTimeout(r, 50));
		assert.deepEqual(cancelledIds, [callId]);
		assert.equal(server.openParkCount, 0);
	} finally {
		await server.stop();
	}
});
