# 基准与记录协议（T00 冻结版）

本文件冻结 T00–T10 共用的**记录字段、身份口径、计时口径与界的适用范围**。冻结的含义是：后续任务可以新增字段，但不得改名、不得改变已有字段的含义、不得把 `None` 解释为"没问题"。

- 依据：Issue #19 / PR #18 拆解文档 §3、§6、§7。
- 实现：`code/candidate_quantum_gates/experiment_manifest.py`（唯一构造与校验入口）。
- D0 数据：`code/candidate_quantum_gates/data/`，由 `make_d0_data.py` 生成与校验。
- 状态：**T00 冻结的是记录格式、身份口径与 D0 数据层**。§3.2 的统计协议（bootstrap、确认集、预注册主指标）是设计约束，尚未实现，标注为"待实现"，不得引用为已完成。

## 1. 记录字段（§6 冻结集）

每次 run 至少保存以下字段。`experiment_manifest.REQUIRED_RECORD_FIELDS` 是同一集合的代码副本，`validate_record` 会拒绝缺字段的记录。

| 字段 | 含义 | 允许值 / 说明 |
|---|---|---|
| `schema_version` | 记录格式版本 | 当前 `1` |
| `code_sha` | 产生该数字的提交 | git 不可用时为 `None`，不得留空字符串 |
| `instance_sha256` | 实例内容哈希 | 由 `data_identity.instance_sha256` 计算，禁止用文件名代替 |
| `split` | 数据划分 | `development` / `validation` / `confirmation` / `diagnostic` |
| `seed` | 随机种子 | 整数；D0 的开发/确认种子见 §3 |
| `pool_sha256` | 候选池内容哈希 | 动态池目前记 `None`（尚未逐 run 内容寻址），不得填别的哈希 |
| `method` | 被测模块 | `tabu` / `independent_uniform` / `witness_xy_joint_simulation` 等 |
| `quantum_execution` | 量子执行方式 | `none` / `simulator` / `qpu`；真实 QPU 必须同时有 `hardware_jobs_submitted > 0` |
| `bound_scope` | 界的适用范围 | `global` / `relaxation` / `pool` / `none` |
| `solver_status` | 求解器状态 | 未证明最优时不得写 `optimal` |
| `global_lower_bound` | 全局合法下界 | 仅 `bound_scope` 为 `global`/`relaxation` 时可填；`pool` 作用域填此字段会被校验拒绝 |
| `pool_lower_bound` | 池内下界 | 不得与 `global_lower_bound` 混用 |
| `verified_upper_bound` | 经独立验证的上界 | 只有通过独立排程验证的工期可填 |
| `bound_evidence` | 界的依据 | 求解器状态/版本/目标值/池哈希等 |
| `wall_seconds` | 端到端墙钟秒 | 含初始化与 JIT，见 §2 |
| `cost_breakdown` | 成本分项 | 至少含 `setup_seconds`、`search_seconds`、`batch_evaluation_seconds`、`proposal_seconds`、`proposal_calls`、`training_evaluations` |
| `graph_evaluations` | 完整图评价次数 | 计费口径的统一分母 |
| `raw_counts_path` | 原始采样计数文件 | 无采样时为 `None` |
| `trace_path` | 逐迭代轨迹文件 | 必填路径，审查者据此复现 |
| `failure_reason` | 失败原因 | 未失败为 `None`；失败样本不得丢弃 |

校验不变量（`validate_record`）：

1. 缺任一字段即报错；
2. `bound_scope="pool"` 时 `global_lower_bound` 必须为 `None`；
3. `quantum_execution="qpu"` 时必须有已提交作业；
4. 未经独立排程验证的结果不得写入记录。

## 2. 计时口径

1. **两把尺子分列。** `search_seconds` 是搜索预算内的墙钟；`wall_seconds` 是含初始化、池构建与 numba JIT 的端到端时间。主结论以端到端为准，但两者都要报告。
2. **不许累加代替墙钟。** CUDA 侧的累计计时与独立墙钟必须同时记录；两者不一致时以独立墙钟为准，并保留差异作为证据。
3. **冷启动与预热分列。** 首次调用含 JIT/编译成本，`setup_seconds` 单列，不得混进搜索预算。
4. **搜索臂同预算。** 同一实例、同一种子下比较的三个臂必须使用相同的 `seconds`、`max_iterations`、`shots`、`train_budget`、`neighborhood` 与 `initial_schedules`；唯一允许变化的是提议模块。
5. **准备成本必须单列。** 独立均匀对照的准备成本是 `O(machines)`；历史路径（`legacy_uniform=True`）会物化 `prod_m K_m` 的合法子空间，这部分成本必须单列，不得摊进"经典对照"。

## 3. 数据身份与 D0 分层

1. **身份以内容为准。** `instance_sha256` 覆盖全部工序时间与机器路线；只比较文件名不算核验。
2. **D0 家族（已冻结并落盘）。** 规模 2×2、3×3、4×3；处理时间为 1–9 的独立均匀整数；每作业路线为机器集合的一个随机排列；每规模 10 个种子，**种子 0–4 开发、5–9 验证**。
3. **生成可复现。** 生成器 `data_identity.build_d0` 只用 `numpy.random.default_rng(seed)`，抽取顺序固定（先时间矩阵、再逐作业路线）。同 `(jobs, machines, seed)` 必得同哈希；`make_d0_data.py verify` 从磁盘重算并比对。
4. **精确最优与诊断隔离。** 4×3 的全排列空间为 `(4!)^3 = 13,824`，可暴力枚举得到精确最优；结果写入 `data/window_oracle_outputs/d0_exact_optima.json`，该文件是诊断真值，**任何采样器、在线算法或提议器都不得读取**。
5. **池界不是全局界。** 受限池的 `C_Π^*` 只能与精确最优做"小于等于/大于等于"的关系陈述，不得作为全局界发布；`bound_scope` 必须如实填写。

## 4. D0 校验产出的记录

`make_d0_data.py build` 写出：

- `data/<jobs>x<machines>/seed<k>.txt`：实例文本（工序时间行 + 1-based 路线行）；
- `data/manifest.json`：全部实例的哈希、划分、规模与清单哈希 `manifest_sha256`；
- `data/window_oracle_outputs/d0_exact_optima.json`：精确最优、枚举组合数、可行组合数与独立验证标记；
- `reports/d0_verification.json`：核对报告。

`make_d0_data.py verify` 的通过条件（全部必须为真）：清单存在、数据版本一致、实例数一致、规模覆盖完整、划分合法、文件齐全、文件哈希与清单一致、由冻结种子可重新生成同哈希、时间与路线逐一相符、诊断文件存在且覆盖完整、精确最优不低于平凡下界。

**正确性门槛：** 上述核验必须 100% 通过；不得以性能抵消错误。

## 5. 尚未冻结 / 待实现

以下内容来自拆解文档 §3.2 与 §3.3，T00 **未实现**，引用时必须注明"设计约束，尚未实现"：

- 10,000 次分层配对 bootstrap 与 95% 区间；
- 确认集 100–119 的采集与冻结（D2）；
- 预注册主指标、主基线、主预算的登记流程；
- 目标到达时间 τ、差距积分 `A_cert`、池覆盖 ρ 等指标的自动化计算；
- TTS 截断与成功率的同时报告。

在 T01/T02 开始产出正式数字前，这些必须补齐，否则 T10 的正贡献门槛无法执行。
