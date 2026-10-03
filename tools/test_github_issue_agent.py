import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import github_issue_agent as agent


class GithubIssueAgentTests(unittest.TestCase):
    def test_claim_skips_pull_requests_and_claims_only_one_issue(self):
        cfg = {"repo": "owner/repo", "bots": {"codex": {}}}
        tasks = [
            {"number": 8, "pull_request": {"html_url": "https://example/pr/8"}},
            {"number": 9, "title": "first"},
            {"number": 10, "title": "second"},
        ]
        with patch.object(agent.agent_github, "bot_token", return_value="token"), patch.object(
            agent.agent_github, "list_tasks", return_value=tasks
        ), patch.object(agent.agent_github, "_req", return_value=[]), patch.object(
            agent.agent_github, "add_comment"
        ) as add_comment:
            result = agent.claim_issue("codex", cfg)

        self.assertEqual(result["status"], "claimed")
        self.assertEqual(result["issue"]["number"], 9)
        add_comment.assert_called_once()

    def test_claim_reports_empty_when_all_tasks_are_claimed(self):
        cfg = {"repo": "owner/repo", "bots": {"codex": {}}}
        tasks = [{"number": 9, "title": "already claimed"}]
        comments = [{"body": "[CLAIMED codex] I am processing this issue."}]
        with patch.object(agent.agent_github, "bot_token", return_value="token"), patch.object(
            agent.agent_github, "list_tasks", return_value=tasks
        ), patch.object(agent.agent_github, "_req", return_value=comments):
            self.assertEqual(agent.claim_issue("codex", cfg), {"status": "empty"})

    def test_publish_creates_draft_and_links_issue(self):
        cfg = {
            "repo": "owner/repo",
            "event_watcher": {"branch": "main"},
            "bots": {"codex": {}},
        }
        with tempfile.TemporaryDirectory() as directory:
            body_file = Path(directory) / "summary.md"
            body_file.write_text("Changed files and tests.", encoding="utf-8")
            calls = []

            def request(method, path, token, payload=None):
                calls.append((method, path, payload))
                if method == "GET":
                    return []
                if method == "POST" and path.endswith("/pulls"):
                    return {"number": 42, "html_url": "https://github.com/owner/repo/pull/42"}
                return {}

            with patch.object(agent.agent_github, "bot_token", return_value="token"), patch.object(
                agent.agent_github, "_req", side_effect=request
            ), patch.object(agent.agent_github, "add_comment") as add_comment:
                result = agent.publish_pr("codex", 9, "codex/issue-9-fix", "Fix issue", str(body_file), cfg)

        self.assertEqual(result["url"], "https://github.com/owner/repo/pull/42")
        pull_payload = next(payload for method, path, payload in calls if method == "POST" and path.endswith("/pulls"))
        self.assertTrue(pull_payload["draft"])
        self.assertIn("Closes #9", pull_payload["body"])
        add_comment.assert_called_once()


if __name__ == "__main__":
    unittest.main()
