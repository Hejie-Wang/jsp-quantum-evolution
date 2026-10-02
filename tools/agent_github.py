#!/usr/bin/env python3
"""GitHub App 机器人共享模块：认证、评论、Issue 操作。

供 agent_dispatch.py / idea_generator.py 等脚本复用。
配置读取 tools/agents.local.json（见 agents.local.json.example，不提交到仓库）。

用法（在仓库根目录）：
    from tools.agent_github import load_config, bot_token, api, add_comment
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import jwt

API = "https://api.github.com"
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "tools" / "agents.local.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.exit(
            f"缺少配置文件 {CONFIG_PATH}\n"
            "请复制 tools/agents.local.json.example 为 tools/agents.local.json 并填写。"
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _req(method: str, path: str, token: str, payload: dict | None = None) -> dict | list:
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(
        f"{API}{path}", data=data, method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        })
    with urllib.request.urlopen(r) as resp:
        return json.loads(resp.read()) if resp.length != 0 else {}


def bot_token(bot: str, cfg: dict | None = None) -> str:
    """用 bot 的 App 私钥换取 1 小时有效的安装令牌。"""
    cfg = cfg or load_config()
    b = cfg["bots"][bot]
    key_path = ROOT / b["key_path"]
    app_id = str(b["app_id"])
    repo = cfg["repo"]
    with open(key_path, "rb") as f:
        key = f.read()
    now = int(time.time())
    app_jwt = jwt.encode({"iat": now - 60, "exp": now + 540, "iss": app_id}, key, algorithm="RS256")
    inst = _req("GET", f"/repos/{repo}/installation", app_jwt)
    tok = _req("POST", f"/app/installations/{inst['id']}/access_tokens", app_jwt, {})
    return tok["token"]


def add_comment(repo: str, issue_num: int, body: str, token: str) -> dict:
    return _req("POST", f"/repos/{repo}/issues/{issue_num}/comments", token, {"body": body})


def list_tasks(repo: str, label: str, token: str) -> list[dict]:
    """列出带指定标签的开放 Issue 与 PR（GitHub 的 issues 接口同时返回 PR）。"""
    return _req("GET", f"/repos/{repo}/issues?labels={label}&state=open&per_page=30", token)
