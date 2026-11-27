import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { Tool } from "@earendil-works/pi-ai";
import { defaultConfig, loadAgyNativeConfig, matchToolPattern, selectMcpTools } from "../src/config.ts";
import { agyRoleInstructions } from "../src/provider.ts";

const makeTool = (name: string): Tool => ({ name, description: name, parameters: { type: "object", properties: {} } as never });

test("loadAgyNativeConfig returns defaults when absent and parses valid overrides", () => {
	const dir = mkdtempSync(join(tmpdir(), "pi-agy-config-"));
	try {
		assert.deepEqual(loadAgyNativeConfig(join(dir, "missing.json")), defaultConfig());

		const file = join(dir, "agy-native.json");
		writeFileSync(
			file,
			JSON.stringify({
				mcpTools: ["delegate", "delegation_*", "session_mail_*"],
				systemPromptSections: ["squire_delegate_task", "squire_parent_delegation"],
				permissions: "agy",
				sandbox: true,
			}),
		);
		assert.deepEqual(loadAgyNativeConfig(file), {
			mcpTools: ["delegate", "delegation_*", "session_mail_*"],
			systemPromptSections: ["squire_delegate_task", "squire_parent_delegation"],
			permissions: "agy",
			sandbox: true,
		});
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test("selectMcpTools matches globs and always excludes reserved Pi and agy_* tools", () => {
	assert.equal(matchToolPattern("delegation_*", "delegation_status"), true);
	assert.equal(matchToolPattern("delegation_*", "delegate"), false);

	const all = [
		makeTool("read"),
		makeTool("bash"),
		makeTool("codemode"),
		makeTool("agy_command"),
		makeTool("agy_tool"),
		makeTool("delegate"),
		makeTool("delegation_status"),
		makeTool("session_mail_send"),
		makeTool("set_session_name"),
	];

	assert.deepEqual(
		selectMcpTools(all, ["*"]).map((t) => t.name),
		["delegate", "delegation_status", "session_mail_send", "set_session_name"],
	);
	assert.deepEqual(
		selectMcpTools(all, ["delegate", "delegation_*"]).map((t) => t.name),
		["delegate", "delegation_status"],
	);
});

test("agyRoleInstructions forwards configured system prompt sections in order", () => {
	const text = agyRoleInstructions(
		{
			appendSystemPrompt: "Extra role.",
			sections: {
				squire_delegate_task: "Delegate task.",
				squire_parent_delegation: "Parent delegation guide.",
				unrelated: "Skip me.",
			},
		},
		["squire_delegate_task", "squire_parent_delegation"],
	);
	assert.equal(text, "Delegate task.\n\nParent delegation guide.\n\nExtra role.");
});
