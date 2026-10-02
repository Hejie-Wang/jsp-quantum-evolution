# Bot PR 流程验证（deepseek）

此文件用于验证 `agent_keys/agent_github.py` 的 GitHub App 机器人流程：

1. 用 App 私钥换取安装令牌（`agent_github.py token`）。
2. 以 `x-access-token` 推送分支到远端。
3. 以机器人身份开启 Pull Request（`agent_github.py open-pr`）。

验证时间：2026-10-02，执行 bot：`jsp-agent-deepseek`（App ID 5161710）。
