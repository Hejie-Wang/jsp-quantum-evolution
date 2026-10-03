# 多 Agent 协作流程（GitHub 消息总线）

本仓库以 GitHub 为唯一的协作枢纽：所有 agent 之间不直接通信，一切消息都通过
Issue、PR 评论和标签传递，天然留痕、可回溯。

## 角色分工

| Agent | 职责 | 本机形态 |
|---|---|---|
| ChatGPT（API 版） | 提出构想 | 从网页端自动提交到仓库 |
| kimi | 仓库管理 | GUI客户端 |
| codex/ glm | 编写程序、测试数据效果 | CLI/GUI （提示词投递） |
| deepseek | 查 OA 文献、审查代码 | GUI 客户端（提示词投递） |

## 事件链路

1. **PR 打开** → GitHub Actions（`.github/workflows/pr-opened-request-review.yml`）
   在 PR 下发"审核请求"评论并打 `agent:codex` 标签。
2. **本地调度器认领** → `python tools/agent_dispatch.py --bot codex`
   发现带标签的未认领任务 → 以 bot 身份发 `[CLAIMED codex]` 评论 → 生成提示词 →
   CLI 型 agent 直接无头执行，GUI 型 agent 生成 `task_outbox/<bot>/task-<编号>.md` 由人粘贴。
   定时任务使用 `python tools/github_issue_agent.py claim --bot codex --json`，每次最多认领一个
   Issue；完成后用 `publish` 子命令创建 Draft PR 并回链原 Issue。
3. **PR 合并** → GitHub Actions（`.github/workflows/pr-merged-notify.yml`）
   按改动内容分类开"🔄 新工作"Issue，只给相关 agent 打标签：
   - 代码/实验类（`code/`、`qskit/`、`task_data/`）→ `agent:codex` + `agent:glm` + `agent:deepseek`；
   - 仓库管理类（其余路径）→ `agent:kimi`；
   - 两类兼有则取并集；ChatGPT 职责是提出构想，不参与合并跟进。
4. **构想生成** → `tools/idea_generator.py` 调 OpenAI API 生成构想，
   以 bot 身份开 Issue 分发给全部 agent。

## 一次手动全流程（新手先跑通这个）

```bash
# 0. 一次性配置：复制配置模板并确认内容
cp tools/agents.local.json.example tools/agents.local.json

# 1. 看看现在有什么任务（dry-run 不会认领）
python tools/agent_dispatch.py --bot codex --dry-run

# 2. 认领并执行 codex 的审核任务（全自动）
python tools/agent_dispatch.py --bot codex

# 3. GUI 客户端（如 deepseek）只生成提示词，由你粘贴进客户端
python tools/agent_dispatch.py --bot deepseek --prepare-only
```

定时任务入口示例：

```bash
python tools/github_issue_agent.py claim --bot codex --json
python tools/github_issue_agent.py publish --bot codex --issue 12 \
  --branch codex/issue-12-short-name \
  --title "修复 Issue #12" --body-file /tmp/pr-summary.md
```

## 定时自动化（可选，两选一）

- **Windows 任务计划程序**：新建基本任务，每 15–30 分钟运行一次
  `python tools\github_issue_agent.py claim --bot codex --json`（"起始于"填仓库根目录）。
- **agent 客户端自带的定时任务**：如 ZCode(GLM) 支持定时自动化，
  可让 agent 自己轮询自己的标签，无需任务计划程序。

## 身份纪律

- 密钥在本地 `agent_keys/`（已 gitignore，绝不入库），配置在 `tools/agents.local.json`。
- agent 的一切远端动作（评论、开 PR、关 Issue）都用各自 bot 的安装令牌，
  在 PR/Issue 上显示为 `jsp-agent-xxx`，人一眼能区分机器产出。
- 认领标记 `[CLAIMED <bot>]` 防止多个实例重复认领同一任务。
