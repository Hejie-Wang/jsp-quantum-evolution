#!/usr/bin/env python3
"""Claim one labelled GitHub Issue and publish a draft PR for it.

The Issue body is treated as data throughout this module.  It is never copied
into a shell command, which keeps the scheduled-agent entry point safe for
untrusted GitHub text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_github


CLAIM_PREFIX = "[CLAIMED "


def _comments(repo: str, issue: int, token: str) -> list[dict]:
    return agent_github._req(
        "GET", f"/repos/{repo}/issues/{issue}/comments?per_page=100", token
    )


def _claimed(comments: list[dict], bot: str) -> bool:
    marker = f"{CLAIM_PREFIX}{bot}]"
    return any(marker in (comment.get("body") or "") for comment in comments)


def claim_issue(bot: str, cfg: dict | None = None) -> dict:
    """Claim the first open, unclaimed Issue assigned to ``bot``."""
    cfg = cfg or agent_github.load_config()
    repo = cfg["repo"]
    token = agent_github.bot_token(bot, cfg)
    tasks = agent_github.list_tasks(repo, f"agent:{bot}", token)

    for issue in tasks:
        # The GitHub Issues API also returns pull requests; this command only
        # claims Issues, and one invocation must claim at most one task.
        if "pull_request" in issue or issue.get("state_reason") == "not_planned":
            continue
        if _claimed(_comments(repo, issue["number"], token), bot):
            continue

        agent_github.add_comment(
            repo,
            issue["number"],
            f"[CLAIMED {bot}] I am processing this issue.",
            token,
        )
        return {"status": "claimed", "issue": issue}

    return {"status": "empty"}


def _existing_pr(repo: str, branch: str, base: str, token: str) -> dict | None:
    head = quote(f"{repo.split('/', 1)[0]}:{branch}", safe="")
    base_q = quote(base, safe="")
    pulls = agent_github._req(
        "GET",
        f"/repos/{repo}/pulls?state=open&head={head}&base={base_q}&per_page=100",
        token,
    )
    return pulls[0] if pulls else None


def publish_pr(
    bot: str,
    issue: int,
    branch: str,
    title: str,
    body_file: str,
    cfg: dict | None = None,
) -> dict:
    """Create or reuse a draft PR and report its URL as JSON."""
    cfg = cfg or agent_github.load_config()
    repo = cfg["repo"]
    base = cfg.get("event_watcher", {}).get("branch", "main")
    body = Path(body_file).read_text(encoding="utf-8").strip()
    if not body:
        raise ValueError("摘要文件不能为空")
    closes = f"Closes #{issue}"
    if closes.lower() not in body.lower():
        body = f"{body}\n\n{closes}"

    token = agent_github.bot_token(bot, cfg)
    pr = _existing_pr(repo, branch, base, token)
    if pr is None:
        pr = agent_github._req(
            "POST",
            f"/repos/{repo}/pulls",
            token,
            {
                "title": title,
                "head": branch,
                "base": base,
                "body": body,
                "draft": True,
            },
        )

    url = pr.get("html_url")
    if not url:
        raise RuntimeError("GitHub 未返回 Draft PR 链接")
    agent_github.add_comment(
        repo,
        issue,
        f"Draft PR 已创建：{url}",
        token,
    )
    return {"status": "published", "url": url, "number": pr.get("number")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    claim = sub.add_parser("claim", help="claim at most one open Issue")
    claim.add_argument("--bot", required=True)
    claim.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    publish = sub.add_parser("publish", help="publish a draft PR")
    publish.add_argument("--bot", required=True)
    publish.add_argument("--issue", required=True, type=int)
    publish.add_argument("--branch", required=True)
    publish.add_argument("--title", required=True)
    publish.add_argument("--body-file", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "claim":
            result = claim_issue(args.bot)
        else:
            result = publish_pr(args.bot, args.issue, args.branch, args.title, args.body_file)
    except Exception as exc:  # keep the scheduled command's failure machine-readable
        result = {"status": "error", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
