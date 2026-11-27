import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, rmSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { test } from "node:test";
import {
	conversationStorePath,
	deleteConversationFiles,
	forkConversationStore,
	recordConversationCheckpoint,
	sweepStaleForks,
} from "../src/conversation-store.ts";

function seedConversationDb(geminiDir: string, id: string): string {
	const dbPath = conversationStorePath(geminiDir, id);
	mkdirSync(dirname(dbPath), { recursive: true });
	const db = new DatabaseSync(dbPath);
	try {
		db.exec(`
			PRAGMA journal_mode = WAL;
			CREATE TABLE trajectory_meta (trajectory_id TEXT PRIMARY KEY, cascade_id TEXT, trajectory_type INTEGER, source INTEGER);
			CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_payload BLOB);
			CREATE TABLE parent_references (idx INTEGER PRIMARY KEY, data BLOB);
			CREATE TABLE gen_metadata (idx INTEGER PRIMARY KEY, data BLOB, size INTEGER NOT NULL DEFAULT 0);
			CREATE TABLE executor_metadata (idx INTEGER PRIMARY KEY, data BLOB);
			CREATE TABLE trajectory_metadata_blob (id TEXT PRIMARY KEY DEFAULT 'main', data BLOB);
			CREATE TABLE battle_mode_infos (idx INTEGER PRIMARY KEY, data BLOB);
		`);
		db.prepare("INSERT INTO trajectory_meta (trajectory_id, cascade_id, trajectory_type, source) VALUES (?, ?, 1, 1)").run(
			randomUUID(),
			id,
		);
		db.prepare("INSERT INTO steps (idx, step_payload) VALUES (?, ?)").run(0, Buffer.from("step-0"));
		db.prepare("INSERT INTO parent_references (idx, data) VALUES (?, ?)").run(0, Buffer.from("ref-0"));
		db.prepare("INSERT INTO gen_metadata (idx, data, size) VALUES (?, ?, 6)").run(0, Buffer.from("gen-0"));
		db.prepare("INSERT INTO executor_metadata (idx, data) VALUES (?, ?)").run(0, Buffer.from("exec-0"));
	} finally {
		db.close();
	}
	return dbPath;
}

function appendTurn(dbPath: string, idx: number): void {
	const db = new DatabaseSync(dbPath);
	try {
		db.prepare("INSERT INTO steps (idx, step_payload) VALUES (?, ?)").run(idx, Buffer.from(`step-${idx}`));
		db.prepare("INSERT INTO parent_references (idx, data) VALUES (?, ?)").run(idx, Buffer.from(`ref-${idx}`));
		db.prepare("INSERT INTO gen_metadata (idx, data, size) VALUES (?, ?, 6)").run(idx, Buffer.from(`gen-${idx}`));
		db.prepare("INSERT INTO executor_metadata (idx, data) VALUES (?, ?)").run(idx, Buffer.from(`exec-${idx}`));
	} finally {
		db.close();
	}
}

test("forkConversationStore copies WAL state, truncates past checkpoint, and preserves source DB", () => {
	const dir = mkdtempSync(join(tmpdir(), "pi-agy-conv-store-"));
	try {
		const sourceId = randomUUID();
		const dbPath = seedConversationDb(dir, sourceId);

		const cp1 = recordConversationCheckpoint(dir, sourceId, 100);
		assert.deepEqual(cp1, { steps: 1, generations: 1, executions: 1 });

		// Simulate a second turn written to the source DB (including WAL) and gen_metadata compaction.
		appendTurn(dbPath, 1);
		const db = new DatabaseSync(dbPath);
		db.prepare("UPDATE gen_metadata SET data = ? WHERE idx = 0").run(Buffer.from("compacted-gen-0"));
		db.close();

		const forkId = forkConversationStore(dir, sourceId, 100);
		assert.ok(forkId && forkId !== sourceId);

		// Forked DB has only turn 0 and updated cascade_id.
		const forkDb = new DatabaseSync(conversationStorePath(dir, forkId), { readOnly: true });
		try {
			const meta = forkDb.prepare("SELECT cascade_id FROM trajectory_meta").get() as { cascade_id: string };
			assert.equal(meta.cascade_id, forkId);
			assert.equal((forkDb.prepare("SELECT count(*) AS n FROM steps").get() as { n: number }).n, 1);
			assert.equal((forkDb.prepare("SELECT count(*) AS n FROM gen_metadata").get() as { n: number }).n, 1);
			assert.equal((forkDb.prepare("SELECT count(*) AS n FROM executor_metadata").get() as { n: number }).n, 1);
		} finally {
			forkDb.close();
		}

		// Source DB is untouched and still has both turns.
		const sourceDb = new DatabaseSync(dbPath, { readOnly: true });
		try {
			assert.equal((sourceDb.prepare("SELECT count(*) AS n FROM steps").get() as { n: number }).n, 2);
		} finally {
			sourceDb.close();
		}

		// Delete helper cleans up all fork files.
		deleteConversationFiles(dir, forkId);
		assert.equal(existsSync(conversationStorePath(dir, forkId)), false);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test("forkConversationStore fails closed on prefix tamper or unsupported indexed table", () => {
	const dir = mkdtempSync(join(tmpdir(), "pi-agy-conv-store-"));
	try {
		const sourceId = randomUUID();
		const dbPath = seedConversationDb(dir, sourceId);
		recordConversationCheckpoint(dir, sourceId, 100);

		// Tamper with step 0 payload -> fingerprint mismatch.
		const db = new DatabaseSync(dbPath);
		db.prepare("UPDATE steps SET step_payload = ? WHERE idx = 0").run(Buffer.from("tampered"));
		db.close();
		assert.throws(() => forkConversationStore(dir, sourceId, 100), /no longer matches/);

		// Unknown table with `idx` column -> schema drift guard.
		const sourceId2 = randomUUID();
		const dbPath2 = seedConversationDb(dir, sourceId2);
		const db2 = new DatabaseSync(dbPath2);
		db2.exec("CREATE TABLE future_table (idx INTEGER PRIMARY KEY, data BLOB)");
		db2.close();
		assert.throws(() => recordConversationCheckpoint(dir, sourceId2, 200), /unsupported indexed table/);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test("sweepStaleForks removes expired forks and orphaned checkpoints while keeping originals and fresh forks", () => {
	const dir = mkdtempSync(join(tmpdir(), "pi-agy-conv-sweep-"));
	try {
		const sourceId = randomUUID();
		const sourceDb = seedConversationDb(dir, sourceId);
		recordConversationCheckpoint(dir, sourceId, 100);

		const staleForkId = forkConversationStore(dir, sourceId, 100)!;
		const freshForkId = forkConversationStore(dir, sourceId, 100)!;

		// Backdate both the original source DB and the stale fork by 40 days.
		const fortyDaysAgo = new Date(Date.now() - 40 * 24 * 60 * 60 * 1000);
		utimesSync(sourceDb, fortyDaysAgo, fortyDaysAgo);
		utimesSync(conversationStorePath(dir, staleForkId), fortyDaysAgo, fortyDaysAgo);

		// Create an orphaned .checkpoints sidecar whose .db was deleted.
		const orphanId = randomUUID();
		const orphanDb = seedConversationDb(dir, orphanId);
		recordConversationCheckpoint(dir, orphanId, 100);
		rmSync(orphanDb);

		const removed = sweepStaleForks(dir, 30 * 24 * 60 * 60 * 1000);
		assert.equal(removed, 2);
		assert.equal(existsSync(sourceDb), true);
		assert.equal(existsSync(conversationStorePath(dir, freshForkId)), true);
		assert.equal(existsSync(conversationStorePath(dir, staleForkId)), false);
		assert.equal(existsSync(`${orphanDb}.checkpoints`), false);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});
