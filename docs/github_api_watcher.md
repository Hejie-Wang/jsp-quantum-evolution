# GitHub API 本地事件提醒

本仓库使用 GitHub API 轮询代替公网 Webhook。它不需要 Cloudflare、ngrok、域名或付费服务器，也不会要求本机暴露端口。

## 一次性配置

在仓库根目录复制配置模板：

```powershell
Copy-Item tools\agents.local.json.example tools\agents.local.json
```

打开 `tools/agents.local.json`，确认：

- `repo` 是目标仓库，例如 `Hejie-Wang/jsp-quantum-evolution`；
- `bots.<name>.app_id` 和 `key_path` 与现有 GitHub App 配置一致；
- `event_watcher.branch` 是需要监听的分支；
- `event_watcher.bots` 指定三类事件交给哪个本地 agent。

私钥放在 `agent_keys/`，`tools/agents.local.json` 不要提交到 Git。

## 先做一次安全测试

第一次建议只轮询一次并预览，不写任务文件：

```powershell
python tools\github_api_watcher.py --once --dry-run --include-existing
```

第一次正式启动时，观察器会把当前事件记录为“已见过”，建立基线，不会把旧事件全部重新提醒。要明确处理当前 API 返回的事件，才使用：

```powershell
python tools\github_api_watcher.py --once --include-existing
```

## 持续运行

每 5 分钟检查一次：

```powershell
python tools\github_api_watcher.py --interval 300
```

收到新的 Issue、PR 或 `main` 分支 Push 后，提示文件会写到：

```text
task_outbox/<bot>/github-<event-id>.md
```

默认只生成提示文件，不会自动执行 agent。确认任务内容和配置无误后，如需启动已配置的 CLI agent，显式添加：

```powershell
python tools\github_api_watcher.py --interval 300 --run-agent
```

当前 Codex CLI 使用 `--approve-for-me`（该参数已隐含 `workspace-write`）。观察器会自动把旧配置中的 `--full-auto` 转换为该参数，并移除冲突的 `--sandbox` 参数。

## 去重和状态

已处理的事件 ID 保存于 `.github-event-state.json`，该文件已加入 `.gitignore`。删除它会重新建立基线；使用 `--include-existing` 才会处理当前可见事件。

## Windows 任务计划程序

可以创建一个“登录时启动”的任务，程序填写 Python 路径，参数填写：

```text
tools\github_api_watcher.py --interval 300
```

“起始于”填写仓库根目录。首次建议不加 `--run-agent`，先观察 `task_outbox` 是否正常生成。

## 与现有调度器的关系

`github_api_watcher.py` 只负责发现事件和生成本地提醒；`tools/agent_dispatch.py` 仍负责处理带 `agent:<bot>` 标签的 Issue/PR、写入认领评论和执行现有 agent 命令。两者可以同时使用，但同一个任务建议只选择一种入口。
