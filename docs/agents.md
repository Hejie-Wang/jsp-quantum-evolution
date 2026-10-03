# Agent 参与指南（机器人账号）

本仓库的 AI agent 通过 **GitHub App 机器人身份**参与，不使用真人小号。

## 当前注册的机器人

| 机器人 | App ID | 用途 |
|---|---|---|
| `jsp-agent-codex[bot]` | 5161309 | Codex 系列 agent |
| `jsp-agent-deepseek[bot]` | 5161710 | DeepSeek 系列 agent |
| `jsp-agent-glm[bot]` | 5161646 | GLM 系列 agent |

## 工作方式

1. 每个 agent 在自己的 clone `improvement-<agent名>/` 中开发，主工作区 `improvement/` 只作为
   人类只读检查点与令牌换取点（详见 `agent_collaboration.md` 的工作区纪律一节）；
2. agent 使用 App 私钥换取 1 小时有效的安装令牌（`agent_github.py token`），私钥不复制进 clone；
3. 在自己的 clone 内推送到特性分支 `<agent>/<任务名>`（**无法直推 main**，分支保护 ruleset 强制）；
4. 以机器人身份开 PR（`agent_github.py open-pr`），PR 自动触发 CI；
5. CI 通过 + 仓库所有者批准后，squash 合入。

## 权限边界

每个 App 仅有 `Contents: write` / `Pull requests: write` / `Issues: write`，
无法修改仓库设置、CI 配置和密钥。新增 agent = 新建一个 App，互不影响。

## 贡献规范

这里是一个贡献示例：

```markdown
# 核心贡献（比如：提升并行计算效率/优化量子计算效果）

2026年10月3日09点37分 由 jsp-agent-codex[bot] 提交，提升了量子计算模拟的效率。

## 贡献内容（一句话介绍贡献）

针对量子计算中的量子门深度过深的问题，基于Grover算法优化了量子电路的编译过程，减少了量子门的数量，从而提升了模拟效率。

## 实现方法（简要描述实现方法）

原来的量子电路门采用了多个`CNOT`门和`H`门的组合，现有方法采用如下门组合：

$$
\bar{U} = H \cdot CNOT \cdot H
\bar{\phi} = H \cdot CNOT \cdot H
$$

将原来需要100层的量子门深度优化为10层，并且可以达到的运算效果与原来相当。

## 结果验证（简要描述验证结果）

在tai50*20数据集上3个不同的示例上运行，其结果如下：

|     | tai50*20_test_1 | tai50*20_test_2 | tai50*20_test_3 |
|-----|-----------------|-----------------|-----------------|
|  原始方法运行时间 | 120s            | 130s            | 125s            |
|  优化方法运行时间 | 12s             | 13s             | 12.5s           |
```