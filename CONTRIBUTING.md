# 贡献指南

本仓库是一个开放式的 JSP（作业车间调度）量子算法 playground：人类和 AI agent 都可以通过 Issue 和 Pull Request 参与提出、实现、审查算法改进。

## 工作方式

1. **讨论先于实现**：新想法先在 Issue（使用"算法改进提议"模板）中写清背景、假设和验证方案，再动手写代码。
2. **改动走 PR**：直接向 `main` 的推送被分支保护阻止。请 fork 或在分支上开发，通过 PR 提交。PR 需通过 CI 测试并获得仓库所有者批准后才会合入（squash merge）。
3. **演进可追溯**：改进型 PR 请在正文中链接来源，例如 `Builds on #8` / `Closes #12`。

## 写作纪律（重要）

本仓库的文档严格区分三类陈述，提交时请遵守同样的标准：

- **设计决定**：做了什么选择、为什么，标注理由即可；
- **待验证假设**：必须明确标注为假设，不得写成定理或结论；
- **已证结论**：必须附验证方法与数据（命令、benchmark、日志）。

**负结果必须保留**：实验失败、性能回退同样是有效贡献，请如实记录并打 `negative-result` 标签，不要删除或粉饰。

## 测试要求

- 涉及 `code/` 的改动必须通过现有单元测试：
  - `cd code/candidate_quantum_demo && python -m unittest -v`
  - `cd code/candidate_quantum_medium && python -m unittest -v`
- CI 会自动运行上述测试；新增功能请附带相应测试。

## 安全红线

- **禁止提交任何密钥、token、license code**（包括 kaiwu license、IBM Quantum token）。仓库已开启 secret scanning 和 push protection，含密钥的推送会被拒绝。
- **禁止修改 `.github/workflows/` 下的 CI 配置**（该目录仅仓库所有者可改）。
- 外部贡献者的 PR 触发的 CI 需要所有者手动批准后才运行。

## 标签约定

| 标签 | 含义 |
|---|---|
| `agent-contribution` | 由 AI agent 提交 |
| `needs-review` | 等待审查 |
| `verified-reproduction` | 结果已被独立复现 |
| `negative-result` | 负结果（保留，不删除） |
| `docs` / `paper` | 文档 / 论文相关 |
