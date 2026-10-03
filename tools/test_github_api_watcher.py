import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import github_api_watcher as watcher


CFG = {
    "repo": "owner/repo",
    "default_bot": "codex",
    "event_watcher": {
        "branch": "main",
        "bots": {"issues": "codex", "pull_requests": "codex", "pushes": "codex"},
    },
    "bots": {"codex": {"command": None}},
}


def issue_event(event_id: str, number: int = 1) -> dict:
    return {
        "id": event_id,
        "type": "IssuesEvent",
        "repo": {"name": "owner/repo"},
        "actor": {"login": "author"},
        "created_at": "2026-10-03T00:00:00Z",
        "payload": {
            "action": "opened",
            "issue": {
                "number": number,
                "title": "Improve watcher",
                "body": "Please inspect this.",
                "html_url": f"https://github.com/owner/repo/issues/{number}",
            },
        },
    }


class WatcherTests(unittest.TestCase):
    def test_run_agent_resolves_windows_codex_shim(self):
        with patch.object(watcher.shutil, "which", side_effect=lambda name: "C:/codex.cmd" if name == "codex.cmd" else None), patch.object(
            watcher.subprocess, "run", return_value=type("Result", (), {"returncode": 0})()
        ) as run:
            code = watcher.run_agent("codex", Path("prompt.md"), {"bots": {"codex": {"command": ["codex", "exec", "{prompt_file}"]}}})
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args[0][0], "C:/codex.cmd")

    def test_run_agent_migrates_removed_full_auto_flag(self):
        with patch.object(watcher.shutil, "which", return_value="C:/codex.cmd"), patch.object(
            watcher.subprocess, "run", return_value=type("Result", (), {"returncode": 0})()
        ) as run:
            code = watcher.run_agent("codex", Path("prompt.md"), {"bots": {"codex": {"command": ["codex", "exec", "--full-auto", "{prompt_file}"]}}})
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args[0][1:4], ["exec", "--approve-for-me", "prompt.md"])

    def test_run_agent_removes_conflicting_sandbox_flag(self):
        command = ["codex", "exec", "--approve-for-me", "--sandbox", "workspace-write", "{prompt_file}"]
        with patch.object(watcher.shutil, "which", return_value="C:/codex.cmd"), patch.object(
            watcher.subprocess, "run", return_value=type("Result", (), {"returncode": 0})()
        ) as run:
            code = watcher.run_agent("codex", Path("prompt.md"), {"bots": {"codex": {"command": command}}})
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args[0][1:], ["exec", "--approve-for-me", "prompt.md"])

    def test_event_kind_filters_repo_and_branch(self):
        event = {
            "type": "PushEvent",
            "repo": {"name": "owner/repo"},
            "payload": {"ref": "refs/heads/main"},
        }
        self.assertEqual(watcher.event_kind(event, "owner/repo", "main"), "pushes")
        event["payload"]["ref"] = "refs/heads/dev"
        self.assertIsNone(watcher.event_kind(event, "owner/repo", "main"))

    def test_first_poll_baselines_and_second_poll_writes_prompt(self):
        root = Path(__file__).resolve().parents[1] / f".test-github-watcher-{uuid.uuid4().hex}"
        root.mkdir()
        state = root / "state.json"
        outbox = root / "outbox"
        events = [issue_event("1")]
        try:
            with patch.object(watcher, "ROOT", root), patch.object(watcher, "OUTBOX", outbox), patch.object(
                watcher, "bot_token", return_value="token"
            ), patch.object(watcher, "fetch_events", side_effect=lambda cfg, token: list(events)):
                self.assertEqual(watcher.poll_once(CFG, state), [])
                events.append(issue_event("2", 2))
                files = watcher.poll_once(CFG, state)

            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].exists())
            self.assertIn("Improve watcher", files[0].read_text(encoding="utf-8"))
            self.assertEqual(set(json.loads(state.read_text(encoding="utf-8"))["seen_ids"]), {"1", "2"})
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
