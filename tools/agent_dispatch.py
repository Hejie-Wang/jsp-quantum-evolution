#!/usr/bin/env python3
"""多 agent 协作调度器：轮询 GitHub 上分配给某个 bot 的任务并触发对应 agent。

工作方式（在仓库根目录运行）：
    python tools/agent_dispatch.py --bot codex
    python tools/agent_dispatch.py --bot kimi --prepare-only

流程：
  1. 用该 bot 的 GitHub App 私钥换取安装令牌（tools/agent_github.py）；
  2. 拉取带 `agent:<bot>` 标签的开放 Issue / PR；
  3. 跳过已被认领（含 `[CLAIMED <bot>]` 评论）的任务；
  4. 认领任务（以 bot 身份发 `[CLAIMED <bot>]` 评论）；
  5. 生成任务提示词文件 task_outbox/<bot>/task-<编号>.md，然后：
     - 若配置了 command（CLI 型 agent，如 codex）：无头执行，任务全自动；
     - 否则（GUI 型 agent）：仅生成提示词文件，由人在客户端里粘贴运行。

建议用 Windows 任务计划程序每 15-30 分钟运行一次，见 docs/agent_collaboration.md。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_github import ROOT, add_comment, bot_token, list_tasks, load_config

OUTBOX = ROOT / "task_outbox"


def comments_contain(repo: str, num: int, marker: str, token: str) -> bool:
    import agent_github
    try:
        cs = agent_github._req("GET", f"/repos/{repo}/issues/{num}/comments?per_page=100", token)
    except urllib.error.HTTPError as e:
        print(f"  拉取评论失败(#{num}): {e}")
        return False
    return any(marker in c.get("body", "") for c in cs)


def build_prompt(task: dict, bot: str, cfg: dict) -> str:
    is_pr = "pull_request" in task
    kind = "Pull Request" if is_pr else "Issue"
    lines = [
        f"你是仓库 {cfg['repo']} 的协作 agent（bot 名：jsp-agent-{bot}）。"
        f"请完成以下分配给你的任务（{kind} #{task['number']}）。",
        "",
        f"标题：{task['title']}",
        f"链接：{task['html_url']}",
        "正文：",
        task.get("body") or "（无正文）",
        "",
        "完成要求：",
        f"- 在本地仓库 {ROOT} 中工作；如任务是 PR，先 git fetch 并切到该 PR 的分支再审查/修改。",
        f"- 身份纪律：提交代码、评论一律以 bot 身份进行。可用 tools/agent_github.py 换取令牌"
        f"（bot 名 {bot}，配置见 tools/agents.local.json），并用它调用 GitHub API。",
        "- 完成后以 bot 身份在本任务下发一条评论，注明做了什么、结论或产出的分支/PR 链接；"
        "若任务是 Issue 且工作已完结，以 bot 身份关闭它。",
    ]
    return "\n".join(lines)


def run_command(command: list[str], prompt_file: Path, bot: str) -> int:
    cmd = [p.format(prompt=str(prompt_file), prompt_file=str(prompt_file)) for p in command]
    print(f"  启动 CLI agent：{' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=ROOT)
    except FileNotFoundError as e:
        print(f"  找不到可执行文件：{e}。请在 tools/agents.local.json 检查 command 配置。")
        return 127
    return r.returncode


def dispatch(bot: str, prepare_only: bool, dry_run: bool) -> None:
    cfg = load_config()
    repo = cfg["repo"]
    b = cfg["bots"][bot]
    label = f"agent:{bot}"
    token = bot_token(bot, cfg)
    tasks = list_tasks(repo, label, token)
    if not tasks:
        print(f"[{bot}] 当前没有待办任务。")
        return

    claim_marker = f"[CLAIMED {bot}]"
    command = b.get("command")
    if prepare_only:
        command = None

    for task in tasks:
        num, title = task["number"], task["title"]
        if task.get("state_reason") == "not_planned":
            continue
        print(f"[{bot}] 发现任务 #{num}：{title}")
        if comments_contain(repo, num, claim_marker, token):
            print("  已被认领过，跳过。")
            continue
        if dry_run:
            print("  （dry-run：不认领，仅列出）")
            continue

        add_comment(repo, num, f"🤖 {claim_marker} 我已认领此任务，正在处理。", token)
        print("  已发认领评论。")

        out_dir = OUTBOX / bot
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = out_dir / f"task-{num}.md"
        prompt_file.write_text(build_prompt(task, bot, cfg), encoding="utf-8")
        print(f"  提示词已写入 {prompt_file}")

        if command:
            code = run_command(command, prompt_file, bot)
            summary = (f"✅ `[DONE {bot}]` 任务 #{num} 的 CLI agent 已退出（exit {code}）。"
                       f"请查看该 agent 留下的评论与产出。")
            add_comment(repo, num, summary, token)
            print(f"  CLI agent 已执行（exit {code}），已发完成记录。")
        else:
            print("  GUI 型 agent：请在对应客户端中打开上面的提示词文件，粘贴执行。")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bot", required=True, choices=["codex", "kimi", "glm", "deepseek"])
    p.add_argument("--prepare-only", action="store_true",
                   help="只生成提示词文件，不启动 CLI agent（GUI 客户端用）")
    p.add_argument("--dry-run", action="store_true", help="只列出任务，不认领")
    args = p.parse_args()
    dispatch(args.bot, args.prepare_only, args.dry_run)


if __name__ == "__main__":
    main()
