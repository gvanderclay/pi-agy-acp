/**
 * WAL-safe Agy conversation checkpoints and copy-on-rewind SQLite forks.
 *
 * The source conversation DB is always opened read-only and copied with
 * `VACUUM INTO` so live WAL state is captured without mutating the source.
 * Any schema mismatch, fingerprint drift, or SQLite error fails closed so the
 * provider falls back cleanly to a fresh session with transcript replay.
 */

import { createHash, randomUUID } from "node:crypto";
import { appendFileSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { isRecord } from "./acp-wire.ts";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const FINGERPRINT_PREFIX = "v1:";
const REQUIRED_TABLES = ["trajectory_meta", "steps", "parent_references", "gen_metadata", "executor_metadata"] as const;
const KNOWN_TABLES = new Set<string>([...REQUIRED_TABLES, "trajectory_metadata_blob", "battle_mode_infos"]);

export interface ConversationCheckpoint {
	steps: number;
	generations: number;
	executions: number;
}

export interface ConversationStore {
	recordCheckpoint(geminiDir: string, acpSessionId: string, timestamp: number): ConversationCheckpoint | undefined;
	forkConversation(geminiDir: string, sourceId: string, timestamp: number): string | undefined;
	deleteConversation(geminiDir: string, id: string): void;
}

export function isNativeForkDisabled(): boolean {
	return process.env.PI_AGY_ACP_DISABLE_NATIVE_FORK === "1";
}

/** Store path of a validated ACP conversation UUID; rejects path components. */
export function conversationStorePath(geminiDir: string, id: string): string {
	if (!UUID.test(id)) throw new Error(`Invalid Antigravity conversation id: ${id}`);
	return join(geminiDir, "antigravity-cli", "conversations", `${id}.db`);
}

function assertCheckpoint(value: ConversationCheckpoint): void {
	if (![value.steps, value.generations, value.executions].every((count) => Number.isSafeInteger(count) && count > 0)) {
		throw new Error("Antigravity checkpoint must contain completed conversation rows");
	}
}

function assertSupportedSchema(db: DatabaseSync): void {
	const rows = db
		.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
		.all() as Array<{ name: string }>;
	const names = new Set(rows.map((row) => row.name));
	for (const required of REQUIRED_TABLES) {
		if (!names.has(required)) throw new Error(`Antigravity conversation DB is missing table '${required}'`);
	}
	for (const name of names) {
		if (KNOWN_TABLES.has(name)) continue;
		const safeName = name.replaceAll('"', '""');
		const columns = db.prepare(`PRAGMA table_info("${safeName}")`).all() as Array<{ name: string }>;
		if (columns.some((col) => col.name === "idx")) {
			throw new Error(`Antigravity conversation DB has unsupported indexed table '${name}'`);
		}
	}
}

function assertIdentity(db: DatabaseSync, id: string): void {
	const rows = db.prepare("SELECT cascade_id FROM trajectory_meta").all() as Array<{ cascade_id: unknown }>;
	if (rows.length !== 1 || rows[0]?.cascade_id !== id) {
		throw new Error("Antigravity conversation identity mismatch");
	}
}

function nextIndex(db: DatabaseSync, table: "steps" | "gen_metadata" | "executor_metadata"): number {
	const row = db.prepare(`SELECT coalesce(max(idx) + 1, 0) AS next FROM ${table}`).get() as { next: number } | undefined;
	return Number(row?.next ?? 0);
}

function highWaterMarks(db: DatabaseSync): ConversationCheckpoint {
	return {
		steps: nextIndex(db, "steps"),
		generations: nextIndex(db, "gen_metadata"),
		executions: nextIndex(db, "executor_metadata"),
	};
}

function prefixFingerprint(db: DatabaseSync, checkpoint: ConversationCheckpoint): string {
	const hash = createHash("sha256");
	const tables = [
		["steps", checkpoint.steps],
		["parent_references", checkpoint.steps],
		["gen_metadata", checkpoint.generations],
		["executor_metadata", checkpoint.executions],
	] as const;
	for (const [table, end] of tables) {
		hash.update(table);
		// agy compacts generation blobs in place after recording later generations.
		const columns = table === "gen_metadata" ? "idx" : "*";
		const rows = db.prepare(`SELECT ${columns} FROM ${table} WHERE idx < ? ORDER BY idx`).all(end) as Array<
			Record<string, unknown>
		>;
		for (const row of rows) {
			for (const key of Object.keys(row).sort()) {
				hash.update(JSON.stringify(key));
				const value = row[key];
				if (value instanceof Uint8Array) {
					hash.update(`blob:${value.byteLength}:`);
					hash.update(value);
				} else {
					const text = value === null ? "null" : String(value);
					hash.update(`${typeof value}:${text.length}:${text}`);
				}
			}
			hash.update("\n");
		}
	}
	return FINGERPRINT_PREFIX + hash.digest("hex");
}

function findRecordedCheckpoint(
	store: string,
	timestamp: number,
): { checkpoint: ConversationCheckpoint; fingerprint: string } | undefined {
	let raw: string;
	try {
		raw = readFileSync(`${store}.checkpoints`, "utf8");
	} catch {
		return undefined;
	}
	const lines = raw.split("\n");
	for (let i = lines.length - 1; i >= 0; i--) {
		const line = lines[i]?.trim();
		if (!line) continue;
		try {
			const record: unknown = JSON.parse(line);
			if (
				!isRecord(record) ||
				record.timestamp !== timestamp ||
				!isRecord(record.checkpoint) ||
				typeof record.fingerprint !== "string" ||
				!record.fingerprint.startsWith(FINGERPRINT_PREFIX)
			) {
				continue;
			}
			const { steps, generations, executions } = record.checkpoint;
			if (typeof steps !== "number" || typeof generations !== "number" || typeof executions !== "number") continue;
			const checkpoint: ConversationCheckpoint = { steps, generations, executions };
			assertCheckpoint(checkpoint);
			return { checkpoint, fingerprint: record.fingerprint };
		} catch {
			// Ignore malformed lines.
		}
	}
	return undefined;
}

/** Records a turn-end checkpoint in `<id>.db.checkpoints` without modifying the DB. */
export function recordConversationCheckpoint(
	geminiDir: string,
	acpSessionId: string,
	timestamp: number,
): ConversationCheckpoint | undefined {
	if (isNativeForkDisabled() || !UUID.test(acpSessionId)) return undefined;
	const store = conversationStorePath(geminiDir, acpSessionId);
	const db = new DatabaseSync(store, { readOnly: true });
	let checkpoint: ConversationCheckpoint;
	let fingerprint: string;
	try {
		assertSupportedSchema(db);
		assertIdentity(db, acpSessionId);
		checkpoint = highWaterMarks(db);
		assertCheckpoint(checkpoint);
		fingerprint = prefixFingerprint(db, checkpoint);
	} finally {
		db.close();
	}
	appendFileSync(`${store}.checkpoints`, `${JSON.stringify({ timestamp, checkpoint, fingerprint })}\n`, { mode: 0o600 });
	return checkpoint;
}

/**
 * Forks a recorded checkpoint into a brand-new conversation UUID via WAL-safe
 * `VACUUM INTO`, truncating rows past the checkpoint on the copy only.
 */
export function forkConversationStore(geminiDir: string, sourceId: string, timestamp: number): string | undefined {
	if (isNativeForkDisabled() || !UUID.test(sourceId)) return undefined;
	const store = conversationStorePath(geminiDir, sourceId);
	const recorded = findRecordedCheckpoint(store, timestamp);
	if (!recorded) return undefined;

	const { checkpoint, fingerprint } = recorded;
	const forkId = randomUUID();
	const target = conversationStorePath(geminiDir, forkId);
	try {
		const source = new DatabaseSync(store, { readOnly: true });
		try {
			assertSupportedSchema(source);
			assertIdentity(source, sourceId);
			source.prepare("VACUUM INTO ?").run(target);
		} finally {
			source.close();
		}

		const fork = new DatabaseSync(target);
		try {
			fork.exec("BEGIN IMMEDIATE");
			try {
				assertSupportedSchema(fork);
				assertIdentity(fork, sourceId);
				const current = highWaterMarks(fork);
				if (
					checkpoint.steps > current.steps ||
					checkpoint.generations > current.generations ||
					checkpoint.executions > current.executions
				) {
					throw new Error("Antigravity checkpoint exceeds its conversation store");
				}
				if (prefixFingerprint(fork, checkpoint) !== fingerprint) {
					throw new Error("Antigravity checkpoint history no longer matches its recorded state");
				}
				fork.prepare("DELETE FROM steps WHERE idx >= ?").run(checkpoint.steps);
				fork.prepare("DELETE FROM parent_references WHERE idx >= ?").run(checkpoint.steps);
				fork.prepare("DELETE FROM gen_metadata WHERE idx >= ?").run(checkpoint.generations);
				fork.prepare("DELETE FROM executor_metadata WHERE idx >= ?").run(checkpoint.executions);
				fork.prepare("UPDATE trajectory_meta SET cascade_id = ?, trajectory_id = ?").run(forkId, randomUUID());
				fork.exec("COMMIT");
			} catch (error) {
				try {
					fork.exec("ROLLBACK");
				} catch {
					// Already rolled back.
				}
				throw error;
			}
			const check = fork.prepare("PRAGMA integrity_check").get() as { integrity_check?: string } | undefined;
			if (check?.integrity_check !== "ok") {
				throw new Error(`Antigravity forked DB failed integrity check: ${String(check?.integrity_check)}`);
			}
		} finally {
			fork.close();
		}

		appendFileSync(`${target}.checkpoints`, `${JSON.stringify({ timestamp, checkpoint, fingerprint })}\n`, { mode: 0o600 });
		return forkId;
	} catch (error) {
		deleteConversationFiles(geminiDir, forkId);
		throw error;
	}
}

export function deleteConversationFiles(geminiDir: string, id: string): void {
	if (!UUID.test(id)) return;
	const store = conversationStorePath(geminiDir, id);
	for (const suffix of ["", "-wal", "-shm", ".checkpoints"]) {
		rmSync(`${store}${suffix}`, { force: true });
	}
}

export const defaultConversationStore: ConversationStore = {
	recordCheckpoint: recordConversationCheckpoint,
	forkConversation: forkConversationStore,
	deleteConversation: deleteConversationFiles,
};
