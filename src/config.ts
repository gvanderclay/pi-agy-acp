/**
 * Optional configuration loaded from `<agentDir>/agy-native.json`.
 *
 * Defaults keep `agy-native` pure stock Agy (no MCP tools projected, only
 * Squire delegate task instructions forwarded). Users can opt into projecting
 * specific Pi extension tools (such as Squire or Session Mail) over MCP.
 */

import { readFileSync } from "node:fs";
import type { Tool } from "@earendil-works/pi-ai";
import { isRecord } from "./acp-wire.ts";

export const DEFAULT_SYSTEM_PROMPT_SECTIONS = ["squire_delegate_task"] as const;

/** Built-in Pi file/shell tools that must never be projected; Agy uses its own stock tools. */
const RESERVED_PI_TOOLS = new Set(["read", "write", "edit", "bash", "codemode"]);

export interface AgyNativeConfig {
	/** Glob patterns or exact Pi tool names to project to Agy via MCP. */
	mcpTools: string[];
	/** Pi system prompt section keys appended via `_meta.antigravity.systemPrompt.append`. */
	systemPromptSections: string[];
}

export function defaultConfig(): AgyNativeConfig {
	return {
		mcpTools: [],
		systemPromptSections: [...DEFAULT_SYSTEM_PROMPT_SECTIONS],
	};
}

function stringList(value: unknown): string[] | undefined {
	if (!Array.isArray(value)) return undefined;
	return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0).map((item) => item.trim());
}

/** Loads `<agentDir>/agy-native.json` if present; returns defaults when absent. */
export function loadAgyNativeConfig(configPath: string, log?: (message: string) => void): AgyNativeConfig {
	let raw: string;
	try {
		raw = readFileSync(configPath, "utf8");
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
			log?.(`Failed to read ${configPath}: ${error instanceof Error ? error.message : String(error)}`);
		}
		return defaultConfig();
	}
	try {
		const parsed: unknown = JSON.parse(raw);
		if (!isRecord(parsed)) return defaultConfig();
		return {
			mcpTools: stringList(parsed.mcpTools) ?? [],
			systemPromptSections: stringList(parsed.systemPromptSections) ?? [...DEFAULT_SYSTEM_PROMPT_SECTIONS],
		};
	} catch (error) {
		log?.(`Failed to parse ${configPath}: ${error instanceof Error ? error.message : String(error)}`);
		return defaultConfig();
	}
}

/** Matches a simple `*`-glob pattern against a Pi tool name. */
export function matchToolPattern(pattern: string, toolName: string): boolean {
	if (!pattern.includes("*")) return pattern === toolName;
	const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replaceAll("*", ".*");
	return new RegExp(`^${escaped}$`).test(toolName);
}

/** True for Pi built-in file/shell tools and Agy's own approval tools. */
export function isReservedPiTool(toolName: string): boolean {
	return RESERVED_PI_TOOLS.has(toolName) || toolName.startsWith("agy_");
}

/** Selects Pi tools matching `patterns`, always excluding reserved built-in and `agy_*` tools. */
export function selectMcpTools(tools: readonly Tool[], patterns: readonly string[]): Tool[] {
	if (patterns.length === 0) return [];
	return tools.filter(
		(tool) =>
			tool.name.length > 0 &&
			!isReservedPiTool(tool.name) &&
			patterns.some((pattern) => matchToolPattern(pattern, tool.name)),
	);
}
