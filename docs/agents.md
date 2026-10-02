# Agent 参与指南（机器人账号）

本仓库的 AI agent 通过 **GitHub App 机器人身份**参与，不使用真人小号。

## 当前注册的机器人

| 机器人 | App ID | 用途 |
|---|---|---|
| `jsp-agent-codex[bot]` | 5161309 | Codex 系列 agent |

## 工作方式

1. agent 使用 App 私钥换取 1 小时有效的安装令牌（`agent_github.py token`）；
2. 推送到特性分支（**无法直推 main**，分支保护 ruleset 强制）；
3. 以机器人身份开 PR（`agent_github.py open-pr`），PR 自动触发 CI；
4. CI 通过 + 仓库所有者批准后，squash 合入。

## 权限边界

每个 App 仅有 `Contents: write` / `Pull requests: write` / `Issues: write`，
无法修改仓库设置、CI 配置和密钥。新增 agent = 新建一个 App，互不影响。
