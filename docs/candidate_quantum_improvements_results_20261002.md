# 机器候选量子搜索改进实施结果（2026-10-02）

实施基线：`e0c53fa`（含 [原交接任务](quantum_candidate_codex_tasks.md) 与本次
[改进交接](candidate_quantum_improvements_20261002.md)）。本文记录按 P0 → P1 →
P2 → P3 顺序完成的代码修改、回归测试与本地运行结果；运行环境为 conda `qskit`
（Qiskit 2.5.2、Aer 0.17.2、NumPy 2.4.6、SciPy 1.17.1），全程无云提交
（`hardware_jobs_submitted = 0`）。

代码位置：`code/candidate_quantum_gates/`（修改 `circuits.py`、`run_offline.py`、
`test_circuits.py`、`__init__.py`、`README.md`；新增 `search_loop.py`、
`pool_optimum_milp.py`、`search_comparison.py`、`test_search_loop.py`、
`results_p3_20261002/`）。

## P0：约束与相位语义修正

修改内容（`circuits.py`、`run_offline.py`）：

1. **同机多关系交集投影**：新增 `allowed_labels` 与 `derive_witness` 作为规范
   投影——候选标签必须满足同机*所有*先后关系才被允许；未涉及机器放开全部标签。
   与 `medium_qjsp.witness_support` 逐例对齐（25 组随机见证交叉校验通过），
   消除两套语义。文档示例 `allowed_labels([(0,1,2),(2,0,1),(1,2,0)],
   [(0,1),(1,2)]) == (0,)` 有测试固化。
2. **路径相位补上 penalty**：环与超工期路径统一施加 `circuit.p(-angle*penalty,
   target)`，`include_objective` 时编码目标即
   `H = T + penalty × (环违约数 + 超工期路径违约数)`；`penalty` 必须为正。
3. **真实见证替换手写假见证**：demo 固定装置的见证全部由完整图评价
   （`medium_qjsp.decode/separate`）产生后投影，共 6 条
   （4 条路径 L=16/16/11/21 与 2 条环，见 `run_offline.demo_pool_and_witnesses`）。
   `(0,0,0)` 解码为可行排程（工期 16），不再被标成环。
4. **初态准备分离**：`build_phase_circuit` 移除 `initialize_uniform` 参数；新增
   `prepare_legal_basis`（确定性合法基态，即旧参数的真实行为）与
   `prepare_uniform_legal`（逐机精确制备合法 one-hot 均匀叠加）。
5. **恒真/恒假条件**：交集为空或 `fixed_t ≥ L(P)` 的见证不发射任何门（辅助位
   必然干净）；逐机恒真的见证只贡献全局相位，不再物化辅助位。

验收（`test_circuits.py`，20 项全部通过）：

* 小池枚举 8 个合法组合，电路相位与经典见证谓词逐位一致（含 penalty=23、
  `fixed_t=15` 下路径 L>15 才触发的组合语义）。
* T 寄存器模式覆盖 `T = L−1 / L / L+1`（L=3，T=2 触发、T=3/4 不触发），以及
  非单位惩罚与目标寄存器复合相位 `e^{-i·angle·(T + penalty·violation)}`。
* 随机叠加态演化：每个合法分量获得精确相位因子，辅助位布居 ≤ 1e-9。
* 恒真见证无任何数据依赖门；恒假见证零门且辅助位归零。

## P1：辅助位复用与联合门精确构造

修改内容：

1. **共享工作区**：相位电路工作区固定为 `machines + 3`（加 T 寄存器模式的
   `time_bits − 1` 个比较器工作位），不再按见证数量线性增长。1–6 条见证下电路
   总位宽恒定（测试固化）。
2. **联合门替换稠密综合**：`build_joint_mixer` 采用文档给定的精确构造——
   `CX(t,j)` 链 → 非目标左位 `X` → `2r−1` 控 `Rx(2θ)` → 逆序撤销，实现
   `exp[-iθ(|a><b|+|b><a|)]`，零辅助位。稠密矩阵仅保留为 `_local_transition_matrix`
   测试参照。

验收与资源（`test_circuits.py` + 本地编译）：

| r（支撑机数） | 等价性最大偏差 | 精确构造 CX | 稠密综合 CX | 深度（精确/稠密） | 辅助位 |
|---:|---:|---:|---:|---:|---:|
| 2 | 2.2e-16 | 26 | 95 | 48 / 189 | 0 / 0 |
| 3 | 2.3e-15 | 50 | 1783 | 78 / 3525 | 0 / 0 |
| 4 | 6.5e-15 | 94 | 29655 | 190 / 58629 | 0 / 0 |

* 合法编码保持与非目标基矢不变有专门测试；相同编译设置
  （`basis_gates=["u","cx"], optimization_level=0`）下逐矩阵比较。
* **中等规模资源编译**（`run_offline.py --medium-result
  .../15x20_derived_seed11_kaiwu.json`，20 机 × 8 候选）：
  44/44 条见证全部纳入、0 条省略；逻辑位宽 **183**（160 数据 + 23 共享工作），
  转译后 2056 CX、深度 2732；联合层 2 项零辅助位，转译 120 CX。
  旧布局 206 位仅容纳 2 条见证（44 条需 1172 位）。工件：
  `results_p3_20261002/run_offline_20261002.json`。

## P2：固定工期量子优化闭环

新增 `search_loop.py`：

* 多层电路 `合法初态 → [见证相位 → 单机 XY → 多机联合跃迁] × p → 测量`，
  层数、参数评价次数、shots、墙钟全部预算化并记录训练成本。
* 联合动作由当前见证掩码派生（`propose_joint_actions`）：支撑机取
  “0 < |allowed| < 池大小”的机器按约束强度排序，左端点在见证内（优先 incumbent
  标签），右端点取秩最近的被禁标签；不使用任何已知最优标签，也不机械取前几台
  机器/首个候选。
* 外层循环：冻结见证集调参采样 → 去重组合 → 现有完整图评价（`medium_qjsp.decode`
  + 独立 `validate_schedule`）→ 可行且更优才更新 T；不可行/超 T 的解码提取见证、
  去重加入。采样失败不视为不可行证明，量子能量不承诺单调下降（notes 固化）。
* 保留仅 XY 基线（`mode="xy"`）与同信息同联合动作的经典爬山基线
  （`classical_joint_search`）。

验收（`test_search_loop.py`，8 项全部通过；demo 池 12 量子位 ≤ 20 位预算）：

* 相位后接混合器在非退化小例上改变采样分布（均匀 vs 相位+混合器逐组合概率
  不同，statevector 精确验证）。
* demo 闭环（4 轮，87 次参数评价）：初始工期 16 → **11**，达到池内最优（与全枚举
  一致），输出排程独立验证通过，外层历史最好不劣，逐轮资源/概率/评价数全部记录。

## P3：池内最优诊断与多模式对比

**MILP 池内最优诊断**（`pool_optimum_milp.py`，one-hot x + 连续 s + C，
机器边 `s_v ≥ s_u + p_u − U(1−x_ma)`，`0 ≤ s_u ≤ U−p_u`，最小化 C；U 取该池
已验证初始排程工期；单独计时）：

| 池 | 原循环最好工期 | 本地复现池内最优 | 求解状态 | 对偶界 | 耗时 |
|---|---:|---:|---|---:|---:|
| 15×15，seed=7 | 1462 | **1462** | optimal | 1462 | 2.5 s |
| 20×15，seed=7 | 1865 | **1835** | optimal | 1835 | 22.8 s |
| 15×20 派生，seed=11 | 1675 | **1675** | optimal | 1675 | 4.8 s |

三个数值与交接文档一致；池哈希与保存报告一致，报告 `best_choice` 先经独立
验证（可行、工期=U）再作 U。证据含状态、界、解码后验证，已存
`results_p3_20261002/pool_milp_evidence.json`。20×15 池因此确认存在 30 的
改进空间（1865 → 1835）；对已最优的池（1462、1675）不再要求量子侧改善。

**多模式对比**（`search_comparison.py`，demo_3x3 + 两个随机 5×5（每机 2 候选、
8 逻辑数据位 + 8 工作位 ≤ 20 位预算），每模式 4 轮、256 shots、每轮 ≤ 8 次
图评价、参数评价 ≤ 48；精确 MILP 参照单独计时）：

| 模式 | 目标工期成功率 | 有改进案例 | 平均墙钟 | 平均训练成本 | 平均图评价 |
|---|---:|---:|---:|---:|---:|
| 均匀采样 | 3/3 | 1/3 | 1.3 s | 0 | 30.3 |
| 仅 XY | 3/3 | 1/3 | 159.0 s | 152.5 s | 30.3 |
| XY+联合跃迁 | 3/3 | 1/3 | 236.9 s | 226.2 s | 30.0 |
| 经典（同信息同动作） | 3/3 | 1/3 | 0.04 s | 0 | 12.7 |

诚实结论：在这三个小池上，label-0 初始排程本身已是一个 5×5 池的池内最优
（89、119），属退化案例；唯一非退化的 demo_3x3 上四种模式都从 16 改进到 11。
均匀采样在小池上已足够，量子回路训练主导墙钟成本，经典基线便宜三个数量级——
按交接要求保留全部失败与退化结果，纠缠或低能量不构成量子优势证据。原始逐例
数据（含可行率、逐轮概率、电路资源、MILP 计时）见
`results_p3_20261002/search_comparison.json`。

## 复现

```powershell
conda run --no-capture-output -n qskit python -m unittest discover `
  -s code/candidate_quantum_gates -p "test_*.py" -v
# 其余命令见 code/candidate_quantum_gates/README.md
```

回归测试合计 28 项（`test_circuits.py` 20 项 + `test_search_loop.py` 8 项），
全部通过。

## 局限

* 量子侧仅在 ≤ 20 逻辑位的小池上做状态向量训练/采样；更大规模只做资源编译
  （`ensure_simulation_budget` 强制拒绝超预算状态向量）。
* 联合动作来自见证掩码，方向性无保证；采样失败不证明不可行。
* 池内最优是池界而非 JSP 全局界；扩充候选池需保留当前最好排程并重新编译见证，
  属后续候选覆盖改进。
