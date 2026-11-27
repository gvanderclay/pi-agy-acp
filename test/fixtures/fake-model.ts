// A model of another provider that writes the tools each request declares to
// FAKE_MODEL_TOOLS_OUT, and registers a tool on the first prompt, as pi-squire does
// when its delegate routing changes.
import { appendFileSync } from "node:fs";
import { createAssistantMessageEventStream, getCurrentTools, type AssistantMessage, type Model } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function fakeModel(pi: ExtensionAPI): void {
	const model: Model<"fake"> = {
		id: "model",
		name: "Fake",
		api: "fake",
		provider: "fake",
		baseUrl: "fake://",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 100_000,
		maxTokens: 1_000,
	};
	const stream: any = (m: Model<"fake">, context: { messages: { role: string }[] }) => {
		appendFileSync(process.env.FAKE_MODEL_TOOLS_OUT!, `${JSON.stringify(getCurrentTools(context.messages).map((tool) => tool.name).sort())}\n`);
		const events = createAssistantMessageEventStream();
		const message: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "ok" }],
			api: m.api,
			provider: m.provider,
			model: m.id,
			usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
			stopReason: "stop",
			timestamp: Date.now(),
		};
		queueMicrotask(() => {
			events.push({ type: "start", partial: message });
			events.push({ type: "done", reason: "stop", message });
		});
		return events;
	};
	pi.registerProvider({
		id: "fake",
		name: "Fake",
		baseUrl: "fake://",
		auth: { apiKey: { name: "none", check: async () => ({ type: "api_key", source: "fake" }), resolve: async () => ({ auth: {}, source: "fake" }) } },
		getModels: () => [model],
		stream,
		streamSimple: stream,
	} as any);
	let registered = false;
	pi.on("before_agent_start", () => {
		if (registered) return;
		registered = true;
		pi.registerTool({
			name: "late_tool",
			label: "Late",
			description: "Registered after startup.",
			parameters: Type.Object({}),
			execute: async () => ({ content: [{ type: "text", text: "late" }], details: {} }),
		});
	});
}
