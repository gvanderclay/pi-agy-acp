/**
 * Projects selected Pi tool definitions into MCP tool declarations with
 * deterministic sanitized names and normalized object JSON Schemas.
 */

import { createHash } from "node:crypto";
import type { Tool } from "@earendil-works/pi-ai";

export interface ProjectedTool {
	/** MCP-safe advertised tool name (`[a-zA-Z0-9_-]{1,64}`). */
	mcpName: string;
	/** Exact Pi tool name for emitted `toolCall` blocks. */
	piName: string;
	description: string;
	inputSchema: Record<string, unknown>;
}

const EMPTY_INPUT_SCHEMA = { type: "object", properties: {} } as const;

function sanitizeMcpName(name: string): string {
	const safe = name
		.replace(/[^a-zA-Z0-9_-]+/g, "_")
		.replace(/^_+|_+$/g, "")
		.slice(0, 64);
	return safe || "tool";
}

function jsonClone(value: unknown): Record<string, unknown> | undefined {
	try {
		const cloned: unknown = JSON.parse(JSON.stringify(value));
		if (cloned && typeof cloned === "object" && !Array.isArray(cloned)) {
			return cloned as Record<string, unknown>;
		}
		return undefined;
	} catch {
		return undefined;
	}
}

function isObjectSchema(value: Record<string, unknown>): boolean {
	return value.type === "object" && typeof value.properties === "object" && value.properties !== null;
}

function unwrapRootComposition(schema: Record<string, unknown>, depth: number): Record<string, unknown> | undefined {
	if (depth > 8) return undefined;
	if (isObjectSchema(schema)) return schema;

	if (Array.isArray(schema.allOf)) {
		const properties: Record<string, unknown> = {};
		const required = new Set<string>();
		let canMerge = true;
		for (const branch of schema.allOf) {
			const record = jsonClone(branch);
			if (!record || !isObjectSchema(record)) {
				canMerge = false;
				break;
			}
			for (const key of Object.keys(record)) {
				if (key !== "type" && key !== "properties" && key !== "required") {
					canMerge = false;
					break;
				}
			}
			if (!canMerge) break;
			for (const key of Object.keys(record.properties as Record<string, unknown>)) {
				if (Object.hasOwn(properties, key)) {
					canMerge = false;
					break;
				}
			}
			if (!canMerge) break;
			Object.assign(properties, record.properties as Record<string, unknown>);
			if (Array.isArray(record.required)) {
				for (const req of record.required) {
					if (typeof req === "string") required.add(req);
				}
			}
		}
		if (canMerge) {
			return {
				type: "object",
				properties,
				...(required.size > 0 ? { required: [...required] } : {}),
			};
		}
	}

	for (const key of ["oneOf", "anyOf"] as const) {
		const branches = schema[key];
		if (!Array.isArray(branches)) continue;
		const concrete = branches.filter((branch) => {
			const record = jsonClone(branch);
			return !(record && record.type === "null");
		});
		if (concrete.length !== 1) continue;
		const only = jsonClone(concrete[0]);
		if (!only) continue;
		const unwrapped = unwrapRootComposition(only, depth + 1);
		if (unwrapped) return unwrapped;
	}

	return undefined;
}

/** Ensures a Pi tool's JSON Schema is a valid object schema for MCP `inputSchema`. */
export function normalizeInputSchema(parameters: unknown): Record<string, unknown> {
	const cloned = jsonClone(parameters);
	if (!cloned) return { ...EMPTY_INPUT_SCHEMA };
	const normalized = unwrapRootComposition(cloned, 0);
	if (normalized) return normalized;
	if (cloned.type === "object") {
		return { properties: {}, ...cloned };
	}
	return cloned;
}

/** Projects Pi `Tool[]` into deterministic MCP tool definitions with unique safe names. */
export function projectPiTools(tools: readonly Tool[]): ProjectedTool[] {
	const seenPiNames = new Set<string>();
	const usedMcpNames = new Set<string>();
	const projected: ProjectedTool[] = [];

	for (const tool of tools) {
		if (!tool.name || seenPiNames.has(tool.name)) continue;
		seenPiNames.add(tool.name);

		const base = sanitizeMcpName(tool.name);
		let mcpName = base;
		let suffix = 2;
		while (usedMcpNames.has(mcpName)) {
			const tag = `_${suffix++}`;
			mcpName = `${base.slice(0, 64 - tag.length)}${tag}`;
		}
		usedMcpNames.add(mcpName);

		projected.push({
			mcpName,
			piName: tool.name,
			description: tool.description ?? "",
			inputSchema: normalizeInputSchema(tool.parameters),
		});
	}

	return projected;
}

/** Stable SHA-256 fingerprint of a projected tool catalog. */
export function fingerprintProjectedTools(tools: readonly ProjectedTool[]): string {
	return createHash("sha256").update(JSON.stringify(tools)).digest("hex").slice(0, 16);
}
