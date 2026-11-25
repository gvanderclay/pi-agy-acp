# pi-agy-acp

A native [Pi](https://docs.pi.dev/latest/) provider (`agy-acp`) backed by Google Antigravity CLI's built-in ACP server (`agy acp`), bridging **all** tool execution back into Pi over an ephemeral in-memory Streamable HTTP MCP server.

## How it works

1. **Zero built-in tool bloat & full Pi tool ownership:**
   Spawns `agy acp --stdio --agent pi-host --include-default-customizations=false` inside an isolated per-process `--gemini-dir` under `~/.pi/agent/cache/agy-acp/instances/pid-<pid>-<seq>/` with `excludeDefaultComponents: true` and `inheritMcp: true`. All 15 Antigravity built-in fs/shell/web/subagent tools are stripped from the model schema (~550–650 token baseline instead of ~11k+), and Pi's tools (`codemode`, `pi-bg-bash`, `pi-web-access`, etc.) are declared eagerly (`_meta: { "antigravity/alwaysLoad": true }`).
2. **In-memory Streamable HTTP MCP bridge (`timeoutSeconds: -1`):**
   Every tool call parks the ACP turn and emits a real Pi `toolCall` (`stopReason: "toolUse"`), so Pi's `pi-permission-system`, `auto-mode`, `pi-codemode-diff`, and TUI widgets run normally. Dynamic tool changes (`web_enable`, `pi.setActiveTools`) push `notifications/tools/list_changed` over the persistent `GET /mcp` SSE stream mid-session.
3. **Cross-restart `session/resume` & prompt caching:**
   Persists the ACP session binding and a fingerprint of the synced conversation prefix via `pi.appendEntry("agy-acp-state", ...)`. While that prefix still matches, including after a restart, the provider sends only new messages and notices. Pi's tools section can change without resuming the session because MCP already updates the tool list. Other system prompt changes request an append update with `session/resume` on the same session at the next ACP prompt boundary. If that request fails, it keeps the warm session with the previous append. A fresh session replays Pi's transcript when there is no reusable binding or the prefix diverges (for example, after a transcript edit or compaction).
4. **Stateless auxiliary calls:**
   One-shot Pi calls (such as `handoff-lite`, compaction, or branch summarization) run on a separate no-tool ACP session so they never interfere with the live conversation.

## Requirements

- Antigravity CLI (`agy` >= 1.5.17) installed and signed in (`agy --login`), on `PATH` or pointed to via `PI_AGY_ACP_BIN`.
- Pi `>= 1.6.0` on Node `>= 22.19.0`.

## Installation

Add to `~/.pi/agent/settings.json` under `packages`:

```json
"~/workspace/pi-agy-acp"
```

(Or `git:github.com/gvanderclay/pi-agy-acp` once published.)

Then `/reload` Pi and pick any `agy-acp/*` model via `/model`.

## Environment variables

- `PI_AGY_ACP_BIN`: Explicit path to the `agy` binary (optional; auto-discovered from `PATH`, `/opt/homebrew/bin/agy`, `/usr/local/bin/agy`).
- `PI_AGY_ACP_DEBUG=1`: Append the loaded extension path, ACP transport lifecycle, replay reasons, and raw per-turn ACP usage to `~/.pi/agent/cache/agy-acp/debug.log` (under `PI_CODING_AGENT_DIR` if set). No stderr redirect is needed.

## Development

```bash
pnpm typecheck
pnpm test
pnpm test:live   # Runs real end-to-end smoke test against `agy acp`
```
