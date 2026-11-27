# pi-agy-acp

A [Pi](https://docs.pi.dev/latest/) provider (`agy-native`) that runs Google's
stock Antigravity agent (`agy acp`) inside Pi. Agy keeps its own prompt, tools
and conversation history; Pi supplies the interface and decides which of Agy's
tool calls may run.

## How it works

1. **Stock Agy.** The provider starts `agy acp` with your own Antigravity
   configuration and an honest `clientInfo`. It sends no tools. Of Pi's
   system prompt it sends only the role instructions, appended to Agy's own
   prompt through `_meta.antigravity.systemPrompt.append` on every
   `session/new` and `session/resume`: the `--append-system-prompt` text
   and Squire's `squire_delegate_task` section. A normal Pi session has
   neither, so Agy runs with its stock prompt.
2. **Every Agy tool call goes through Pi.** On first use the provider adds a
   `pi-gate` `PreToolUse` entry to `~/.gemini/config/hooks.json` (other entries
   are kept) that runs `~/.pi/agent/cache/agy-native/pi-gate-hook.mjs`. The
   script asks the Pi session that started this `agy acp`. An Agy started
   outside Pi gets `allow` and keeps its own permission flow.
3. **Approval tool calls.** Pi sees each Agy tool call as a tool call of its
   own, so pi-permission-system, auto-mode and the approval dialog apply:
   - `run_command` → `agy_command { command, cwd }` (configure it as a shell
     tool: `"shellTools": { "agy_command": { "commandArgument": "command", "workdirArgument": "cwd" } }`);
   - `read_file` → Pi's own `read { path }`, checked against read rules;
   - `edit_file`, `write_file` → `agy_edit { path }`;
   - anything else → `agy_tool { tool, arguments, path? }`.

   If Pi blocks the call, Agy is told Pi's reason. If Pi allows it, Agy runs
   it and its output becomes the Pi tool result. Subagent calls are gated the
   same way and marked `subagent`; Pi cannot see their output. Calls that
   arrive while no Pi turn is running are denied.
4. **Turns and side requests.** Every turn reaches Agy, including ones an
   extension starts (mail, finished background tasks) and the messages
   extensions add. Side requests (summaries, handoffs) run on a separate Agy
   session that is closed afterwards; their instructions go first in the
   prompt and their tool calls are denied. Compaction keeps Agy's session and
   writes a fixed summary instead of asking a model.
5. **History.** Each Agy reply records its Agy session id. The next message
   continues that session, after a restart too (`session/resume`),
   with turns another model answered in between sent as context. After a fork,
   or a `/tree` jump that drops Agy's latest reply, Agy starts a fresh session
   and is sent the conversation Pi kept.
6. **Turn control.** Esc cancels the Agy turn; a message typed while Agy works
   is sent as a steering prompt.

Pi's context meter counts Pi's own transcript; Agy's token counts are still
reported for cost.

## Requirements

- Antigravity CLI (`agy` 1.5.18) installed and signed in, on `PATH` or set in
  `PI_AGY_ACP_BIN`.
- Pi `>= 1.6.0` on Node `>= 22.19.0`.

## Installation

Add to `~/.pi/agent/settings.json` under `packages`:

```json
"git:github.com/gvanderclay/pi-agy-acp"
```

Then `/reload` Pi and pick an `agy-native/*` model with `/model`.

## Environment variables

- `PI_AGY_ACP_BIN`: path to the `agy` binary (default: found on `PATH`,
  `/opt/homebrew/bin` or `/usr/local/bin`).
- `PI_AGY_ACP_DEBUG=1`: log the ACP lifecycle, hook decisions and per-turn usage
  to `~/.pi/agent/cache/agy-native/debug.log`.

## Development

```bash
pnpm typecheck
pnpm test
pnpm test:live   # real agy acp; installs the pi-gate hook entry
```
