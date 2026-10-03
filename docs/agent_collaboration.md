# 多 Agent 协作流程（GitHub 消息总线）

本仓库以 GitHub 为唯一的协作枢纽：所有 agent 之间不直接通信，一切消息都通过
Issue、PR 评论和标签传递，天然留痕、可回溯。

## 角色分工

| Agent | 职责 | 本机形态 |
|---|---|---|
| ChatGPT（API 版） | 提出构想 | 从网页端自动提交到仓库 |
| kimi | 仓库管理 | GUI客户端 |
| codex/ glm | 编写程序、测试数据效果 | CLI/GUI |
| deepseek | 直接执行分配到的 Issue/任务（文献调研、代码审查为其中一部分） | GUI 客户端 |

## 工作区纪律：每个 agent 一个独立 clone（2026-10-03 起生效）

**每个 agent 在自己的固定 clone 目录中工作，禁止在主工作区及其他 agent 的目录中开发；不使用 `git worktree`。**

初始配置（每个 agent 一次性执行，以 kimi 为例）：

```bash
cd E:/BUAA/paper_composition/JSP_QSWAP
git clone https://github.com/Hejie-Wang/jsp-quantum-evolution.git improvement-kimi
```

目录命名统一为 `improvement-<agent名>`：`improvement-kimi`、`improvement-glm`、`improvement-codex`、`improvement-deepseek`。

规则细节：

1. **主工作区（`improvement/`）是人类只读检查点**：agent 不在其中 checkout、commit、stash，除 fetch 外不做写操作；它只用于人查看代码和换取令牌。
2. **一切开发在自己的 clone 里进行**：分支命名 `<agent>/<任务名>`，与 PR 一一对应；推送、开 PR 流程不变。
3. **令牌换取仍在主工作区执行**：`agent_keys/` 私钥不复制到任何 clone；需要令牌时在主工作区运行 `agent_github.py token`，把令牌用于自己 clone 的推送。令牌 1 小时有效，过期重换。
4. **开工前先同步基线**：在自己的 clone 执行 `git fetch origin && git reset --hard origin/main`（或基于 `origin/main` 新切分支），各 clone 各自保证基线新鲜，不假设别人帮你更新。
5. **各 agent 客户端的默认工作目录永久指向自己的 clone**，不要打开主工作区。
6. **不使用 `git worktree`**；已有的 `.worktrees/t00` 是过渡遗留，对应任务（`feat/t00-uniform-control`）完成后由负责人清理，此后不再新建。

## 事件链路

1. **PR 打开** → GitHub Actions（`.github/workflows/pr-opened-request-review.yml`）
   在 PR 下发"审核请求"评论并打 `agent:codex` 标签。
2. **认领** → 调度器在主工作区运行 `python tools/agent_dispatch.py --bot codex`
   发现带标签的未认领任务 → 以 bot 身份发 `[CLAIMED codex]` 评论 → 生成提示词文件
   `task_outbox/<bot>/task-<编号>.md`。
   定时任务使用 `python tools/github_issue_agent.py claim --bot codex --json`，每次最多认领一个
   Issue；完成后用 `publish` 子命令创建 Draft PR 并回链原 Issue。
   **调度器只负责认领与生成提示词；实际开发由 agent 会话在自己的 clone 内直接执行**（见上节工作区纪律）。
3. **PR 合并** → 流程到此结束，**不再自动开 Issue 广播**。
   后续工作由人直接在 Issue/PR 中指派给对应 agent，避免无关 agent 被打扰。
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

# 3. GUI 客户端（如 deepseek）的准备：调度器只生成提示词文件
python tools/agent_dispatch.py --bot deepseek --prepare-only
#    真正的开发在 improvement-deepseek/ 内由 agent 会话直接执行
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
