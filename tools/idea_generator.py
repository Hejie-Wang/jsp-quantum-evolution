#!/usr/bin/env python3
"""构想生成器：用 OpenAI API 生成项目构想，以 Issue 形式发回仓库。

代替"ChatGPT 网页端提构想"的自动化版本（网页版无 API，钩子递不进消息）。

前置条件：
  - 设置环境变量 OPENAI_API_KEY；
  - tools/agents.local.json 已配置（发 Issue 用 glm bot 的身份）。

用法（在仓库根目录）：
    python tools/idea_generator.py --topic "下一步算法改进方向"
    python tools/idea_generator.py                 # 不带 --topic 时自动取仓库 README 生成
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_github import ROOT, _req, bot_token, load_config

OPENAI_API = "https://api.openai.com/v1/chat/completions"


def call_openai(prompt: str, model: str) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("缺少环境变量 OPENAI_API_KEY")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content":
                "你是项目的首席构想官，负责提出有价值的研究/工程构想。"
                "输出用中文 Markdown，包含：构想摘要、预期收益、可行路径、风险。"
                "内容须具体可执行，避免空话。"},
            {"role": "user", "content": prompt},
        ],
    }
    r = urllib.request.Request(OPENAI_API, data=json.dumps(payload).encode(),
                               headers={"Authorization": f"Bearer {key}",
                                        "Content-Type": "application/json"})
    with urllib.request.urlopen(r) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topic", default=None, help="构想主题；缺省时基于仓库 README 生成")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--bot", default="glm", help="发 Issue 用的 bot 名")
    args = p.parse_args()

    cfg = load_config()
    repo = cfg["repo"]
    if args.topic:
        user_prompt = f"项目主题：{args.topic}。请提出下一个值得尝试的构想。"
    else:
        readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="ignore")
        user_prompt = "以下为项目 README，请提出下一个值得尝试的构想：\n\n" + readme[:6000]

    print("正在调用 OpenAI API 生成构想……")
    idea = call_openai(user_prompt, args.model)

    token = bot_token(args.bot, cfg)
    issue = _req("POST", f"/repos/{repo}/issues", token, {
        "title": "💡 新构想（由 idea_generator 生成）",
        "body": idea + f"\n\n---\n由 `tools/idea_generator.py` 以 bot `jsp-agent-{args.bot}` 身份发布。",
        "labels": ["agent:codex", "agent:kimi", "agent:glm", "agent:deepseek"],
    })
    print(f"构想已发布：{issue['html_url']}")


if __name__ == "__main__":
    main()
