# Bot PR 流程验证（codex）

此文件用于验证 `agent_keys/agent_github.py` 的 GitHub App 机器人流程：

1. 使用 App 私钥换取安装令牌（`agent_github.py token`）。
2. 以 `x-access-token` 身份推送特性分支。
3. 以机器人身份创建 Pull Request（`agent_github.py open-pr`）。

验证时间：2026-10-02，执行 bot：`jsp-agent-codex`（App ID 5161309）。
