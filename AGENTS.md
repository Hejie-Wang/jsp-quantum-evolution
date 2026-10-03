# README 更新流程

当 README 文件更新时，需要满足以下条件：

1. 声称做出的贡献需要真实有效，得到的结果需要经代码运行验证，确保结果的正确性。
2. README 文件更新需要经过 PR 流程，由我和其他agent审查之后，才能合并到主分支。
3. README 文件更新需要遵循本仓库的格式规范，确保内容清晰、易读、易理解。以下是 README 文件的格式规范：

```markdown
## 核心贡献（比如：提升并行计算效率/优化量子计算效果）

xxxx年xx月xx日，一句话介绍本次更新的核心贡献。

### 详细更新内容

1. 本次更新的文件列表，包含文件名和路径。
2. 本次更新的针对的问题描述，简要几句话。
3. 本次更新的解决方案描述，简要几句话。
4. 本次更新的结果描述，简要几句话。

如果有成果表格，附上成果表格。

```

# 本地环境配置

本地环境配置时候，使用`conda`创建虚拟环境，安装依赖包，配置环境变量等。本机现有环境配置如下：

```bash
# conda environments:
#
base                     D:\anaconda3
Gurobi                   D:\anaconda3\envs\Gurobi
JSP                      D:\anaconda3\envs\JSP
Kaiwu                    D:\anaconda3\envs\Kaiwu
autofigure               D:\anaconda3\envs\autofigure
markpdfdown              D:\anaconda3\envs\markpdfdown
qskit                    D:\anaconda3\envs\qskit
universal                D:\anaconda3\envs\universal
```

如果不确定当前环境是否正确，可以使用以下命令查看已有的环境：

```bash
conda info --envs
```