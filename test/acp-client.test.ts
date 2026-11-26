import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { sweepStaleAppData } from "../src/acp-client.ts";

const DAY_MS = 24 * 60 * 60 * 1000;
const HOUR_MS = 60 * 60 * 1000;

function touchWithAge(filePath: string, nowMs: number, ageMs: number): void {
	const ts = (nowMs - ageMs) / 1000;
	utimesSync(filePath, ts, ts);
}

test("sweepStaleAppData prunes stale conversations, logs, and tmp files while preserving active data", (t) => {
	const root = mkdtempSync(join(tmpdir(), "pi-agy-sweep-"));
	t.after(() => rmSync(root, { recursive: true, force: true }));

	const nowMs = Date.now();
	for (const dir of ["conversations", "brain/old-1", "brain/fresh-1", "brain/wal-fresh", "brain/orphan-old", "annotations", "implicit", "log", "crashes"]) {
		mkdirSync(join(root, dir), { recursive: true });
	}

	// 1. Stale conversation (20 days old) + companion artifacts -> should be pruned
	const oldDb = join(root, "conversations", "old-1.db");
	const oldWal = join(root, "conversations", "old-1.db-wal");
	const oldShm = join(root, "conversations", "old-1.db-shm");
	const oldAnn = join(root, "annotations", "old-1.pbtxt");
	const oldImp = join(root, "implicit", "old-1.pb");
	for (const p of [oldDb, oldWal, oldShm, oldAnn, oldImp]) {
		writeFileSync(p, "x");
		touchWithAge(p, nowMs, 20 * DAY_MS);
	}

	// 2. Fresh conversation (2 days old) -> must be preserved
	const freshDb = join(root, "conversations", "fresh-1.db");
	const freshAnn = join(root, "annotations", "fresh-1.pbtxt");
	writeFileSync(freshDb, "x");
	writeFileSync(freshAnn, "x");
	touchWithAge(freshDb, nowMs, 2 * DAY_MS);

	// 3. Old .db with recent -wal (active session) -> must be preserved
	const walDb = join(root, "conversations", "wal-fresh.db");
	const walFile = join(root, "conversations", "wal-fresh.db-wal");
	writeFileSync(walDb, "x");
	writeFileSync(walFile, "x");
	touchWithAge(walDb, nowMs, 20 * DAY_MS);
	touchWithAge(walFile, nowMs, 1 * HOUR_MS);

	// 4. Orphan brain dir (20 days old, no DB) -> should be pruned
	touchWithAge(join(root, "brain", "orphan-old"), nowMs, 20 * DAY_MS);

	// 5. Logs & crashes: 10d old pruned, 2d old preserved
	const oldLog = join(root, "log", "acp-server-old.log");
	const freshLog = join(root, "log", "acp-server-fresh.log");
	const oldCrash = join(root, "crashes", "crash-old.log");
	writeFileSync(oldLog, "x");
	writeFileSync(freshLog, "x");
	writeFileSync(oldCrash, "x");
	touchWithAge(oldLog, nowMs, 10 * DAY_MS);
	touchWithAge(freshLog, nowMs, 2 * DAY_MS);
	touchWithAge(oldCrash, nowMs, 10 * DAY_MS);

	// 6. Top-level .tmp files: 2h old pruned, 10m old preserved
	const oldTmp = join(root, "jetbox_summaries_proto.pb.abc.tmp");
	const freshTmp = join(root, "jetbox_summaries_proto.pb.def.tmp");
	writeFileSync(oldTmp, "x");
	writeFileSync(freshTmp, "x");
	touchWithAge(oldTmp, nowMs, 2 * HOUR_MS);
	touchWithAge(freshTmp, nowMs, 10 * 60 * 1000);

	// 7. Shared summary DB must never be touched
	const summaryDb = join(root, "conversation_summaries.db");
	writeFileSync(summaryDb, "x");
	touchWithAge(summaryDb, nowMs, 30 * DAY_MS);

	sweepStaleAppData(root, nowMs);

	assert.equal(existsSync(oldDb), false);
	assert.equal(existsSync(oldWal), false);
	assert.equal(existsSync(oldShm), false);
	assert.equal(existsSync(join(root, "brain", "old-1")), false);
	assert.equal(existsSync(oldAnn), false);
	assert.equal(existsSync(oldImp), false);
	assert.equal(existsSync(join(root, "brain", "orphan-old")), false);

	assert.equal(existsSync(freshDb), true);
	assert.equal(existsSync(join(root, "brain", "fresh-1")), true);
	assert.equal(existsSync(freshAnn), true);
	assert.equal(existsSync(walDb), true);
	assert.equal(existsSync(walFile), true);
	assert.equal(existsSync(join(root, "brain", "wal-fresh")), true);

	assert.equal(existsSync(oldLog), false);
	assert.equal(existsSync(oldCrash), false);
	assert.equal(existsSync(freshLog), true);

	assert.equal(existsSync(oldTmp), false);
	assert.equal(existsSync(freshTmp), true);
	assert.equal(existsSync(summaryDb), true);
});
