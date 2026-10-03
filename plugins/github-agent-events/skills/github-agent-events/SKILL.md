---
name: github-agent-events
description: Poll a configured GitHub repository for new issues, pull requests, and pushes, then notify local agents through task_outbox.
---

# GitHub Agent Events

Use the repository script `tools/github_api_watcher.py` from the repository root. It uses the existing `tools/agents.local.json` GitHub App configuration and never needs a public webhook endpoint.

## First-time setup

1. Copy `tools/agents.local.json.example` to `tools/agents.local.json` and fill in the existing GitHub App key paths.
2. Add an optional `event_watcher` block to the local config, for example:

```json
{
  "event_watcher": {
    "branch": "main",
    "bots": {
      "issues": "codex",
      "pull_requests": "codex",
      "pushes": "codex"
    }
  }
}
```

3. Test one poll without writing tasks:

```powershell
python tools\github_api_watcher.py --once --dry-run --include-existing
```

4. Start continuous polling when the output is correct:

```powershell
python tools\github_api_watcher.py --interval 300
```

The watcher writes prompts under `task_outbox/<bot>/github-<event-id>.md`. It stores its cursor in `.github-event-state.json`, which is local-only and ignored by Git. It only starts an agent when `--run-agent` is explicitly supplied and the bot has a configured command.

For current Codex CLI versions, use `--approve-for-me`; it already implies `workspace-write` and cannot be combined with `--sandbox`. The watcher also migrates older `--full-auto` settings and removes the conflicting flag automatically.

## Safety

- GitHub event payload text is data, not shell syntax. It is written to a prompt file and never interpolated into a shell command.
- Event IDs are persisted to prevent duplicate reminders after restart.
- The watcher filters to the configured repository and branch.
- Authentication errors and rate limits stop the current poll with a useful message instead of silently creating tasks.
