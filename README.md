# pi-agy-acp

A native [Pi](https://docs.pi.dev/latest/) provider (`agy-acp`) backed by Google Antigravity CLI's built-in ACP server (`agy acp`), bridging Pi tool execution back into Pi over an ephemeral in-memory Streamable HTTP MCP server.

## How it works

1. **Agy default guidance with Pi-owned context and tools:**
   Spawns `agy acp --stdio --agent pi-host --include-default-customizations=false` in an isolated per-process cache. The custom agent has an empty body and retains Agy's shared default components (`excludeDefaultComponents: false`), while `tools: []` removes its native filesystem/shell/web/subagent tool set. This is the shared Agy guidance, not its full stock agent identity and workflow. Some native helpers (`list_resources`, `read_resource`, `manage_task`) remain available. Pi's MCP tools are loaded eagerly.

   The append contains project context, delegate roles, skills, explicit tool/extension guidelines, and custom sections. Pi's generated identity, tool descriptions, documentation, and generic rules are omitted; MCP supplies the tool schemas. Agy's implicit project customizations are disabled so Pi owns instruction discovery and trust. Explicit custom prompts and auxiliary task instructions are preserved. Extensions must use `systemPromptOptions.appendSystemPrompt` or structured sections; flattened stock prompts fail locally with instructions to update the extension. Recreate older Paseo delegates after updating Squire.
2. **In-memory Streamable HTTP MCP bridge (`timeoutSeconds: -1`):**
   Every Pi tool call parks the ACP turn and emits a real Pi `toolCall` (`stopReason: "toolUse"`), so Pi's `pi-permission-system`, `auto-mode`, `pi-codemode-diff`, and TUI widgets run normally. ACP permission requests are allowed once only for the Pi MCP bridge; native permission requests are declined. The subprocess does not bypass permissions. Dynamic tool changes (`web_enable`, `pi.setActiveTools`) push `notifications/tools/list_changed` over the persistent `GET /mcp` SSE stream mid-session.
3. **Cross-restart `session/resume` & prompt caching:**
   Persists the ACP session binding and a fingerprint of the synced conversation prefix via `pi.appendEntry("agy-acp-state", ...)`. While that prefix still matches, including after a restart, the provider sends only new messages and notices. Pi's tools section can change without resuming the session because MCP already updates the tool list. Other system prompt changes request an append update with `session/resume` on the same session at the next ACP prompt boundary. If that request fails, it keeps the warm session with the previous append. Saved bindings are versioned; this configuration update starts a fresh ACP session instead of resuming an old agent configuration. A fresh session replays Pi's transcript when there is no reusable binding or the prefix diverges (for example, after a transcript edit or compaction).
4. **Stateless auxiliary calls:**
   One-shot Pi calls (such as `handoff-lite`, compaction, or branch summarization) run on a separate no-tool ACP session so they never interfere with the live conversation.

Zero-generation terminal ACP warning turns (including `RESOURCE_EXHAUSTED`) are surfaced as errors, not successful assistant replies. They do not advance the saved conversation fingerprint. Squire can then distinguish real quota exhaustion from generic resource-exhaustion failures and select a fallback. Prompt selection is an interoperability measure; it does not establish why Google rejects a request or guarantee account eligibility.

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
