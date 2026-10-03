# T00 报告：数据身份、时间口径与独立经典对照

日期：2026-10-03。分支：`feat/t00-uniform-control`。base：`e24f6bd`（PR #18 合并后）。任务：Issue #19 / PR #18 拆解文档 §5 **T00（P0）**。

本文只报告**已跑出并经代码验证**的数字；设计约束、未跑项与负结果分别标注。

## 1. 要解决的问题

1. **均匀对照不独立。** `adaptive_search.solve` 的 `uniform` 模式原先也构造 `CompactSimulator`：先把 `∏_m K_m` 个合法标签全部物化、逐条计算见证能量掩码，然后才抽样（`compact_simulator.py` 的 `__init__` 与 `sample()`）。这等于给**经典对照**附加了与量子模拟同阶的准备成本，主基线被人为削弱，三臂比较不成立。
2. **身份口径不统一。** 结果里的实例身份只有哈希字段，缺一份可核验的分层数据；历史结果无法确认是否同一实例。
3. **记录字段不统一。** 各任务自行组装的 JSON 字段名不一，界的适用范围（池内 vs 全局）没有强制字段。

## 2. 交付内容

| 文件 | 作用 |
|---|---|
| `code/candidate_quantum_gates/uniform_sampler.py` | 新增独立均匀采样器：逐机按候选标签直接抽样，准备成本 `O(machines)`，内存 `O(shots×machines)` |
| `code/candidate_quantum_gates/adaptive_search.py` | `uniform` 路径改走独立采样器；旧路径保留在 `legacy_uniform=True`；新增成本计数字段 |
| `code/candidate_quantum_gates/data_identity.py` | 实例内容身份与哈希、D0 生成器、全排列精确最优、受限池最优 |
| `code/candidate_quantum_gates/make_d0_data.py` | D0 数据生成与磁盘核验命令（`build` / `verify`） |
| `code/candidate_quantum_gates/experiment_manifest.py` | §6 冻结字段集的唯一构造与校验入口，含 `run_and_record` |
| `code/candidate_quantum_gates/t00_evidence.py` | 本次证据脚本（准备成本分离 + 固定实例双路径对照） |
| `code/candidate_quantum_gates/test_t00_control.py` | 24 个单元测试，覆盖本次全部验收条件 |
| `code/candidate_quantum_gates/data/` | D0 数据 30 个实例 + `manifest.json` + 隔离的诊断真值 |
| `docs/benchmark_protocol.md` | 冻结记录字段、计时口径、身份口径与界的适用范围 |
| `reports/d0_verification.json`、`reports/t00_evidence.json`（均位于 `code/candidate_quantum_gates/`） | 可复核的核验与证据输出 |

## 3. 已证结论（有代码与数据支撑）

### 3.1 均匀对照已与量子模拟解耦

在 D0 4×3 实例（`data/4x3/seed0.txt`，哈希 `4d65fb11…a665a29`）、种子 7、3 秒预算下：

| 指标 | 新路径（独立采样器） | 旧路径（`legacy_uniform=True`） |
|---|---:|---:|
| 提议后端 | `independent_uniform_sampler` | `compact_simulator_uniform_legacy` |
| `CompactSimulator` 构造次数 | **0** | 610 |
| 提议调用次数（同墙钟） | **827** | 610 |
| 迭代次数 | 33,107 | 24,433 |
| 最优工期 | 25 | 25 |
| 独立排程验证 | 通过 | 通过 |

同预算内提议次数提升约 36%，最优工期相同，两条路径都通过独立排程验证。

### 3.2 准备成本不再随 `∏K_m` 增长

同池、同 shots=64，各重复 20 次取均值（3 候选/机器）。下表为一次代表性运行；**权威数据以 `code/candidate_quantum_gates/reports/t00_evidence.json` 为准**，重跑会有正常的计时抖动：

| 活跃机器数 | 全空间状态数 | 独立采样器 | 旧路径准备 | 准备加速比 |
|---:|---:|---:|---:|---:|
| 4 | 81 | 3.76e-5 s | 9.71e-5 s | 2.6× |
| 5 | 243 | 4.17e-5 s | 1.19e-4 s | 2.9× |
| 6 | 729 | 4.76e-5 s | 1.81e-4 s | 3.8× |
| 7 | 2,187 | 5.43e-5 s | 3.20e-4 s | 5.9× |
| 8 | 6,561 | 6.32e-5 s | 1.25e-3 s | 19.8× |

独立采样器耗时几乎不随池规模变化；旧路径成本随全空间线性增长。池规模超过 `CompactSimulator` 上限时差异变成"能否运行"：`3^18 = 387,420,489` 状态下旧路径直接抛 `candidate subspace exceeds simulation budget`，独立采样器照常抽样。

### 3.3 D0 数据层可核验

`make_d0_data.py verify` 13 项检查全部通过（`code/candidate_quantum_gates/reports/d0_verification.json`）：

- 30 个实例（2×2、3×3、4×3 各 10 个种子；种子 0–4 开发、5–9 验证）；
- 30/30 文本往返哈希一致；由冻结种子可重新生成同哈希；时间与路线逐一相符；
- 诊断文件中全部精确最优经独立验证，且不低于平凡下界 `L_0`。

精确最优示例（诊断层，不得馈入任何在线算法）：

| 实例 | 下界 `L_0` | 精确最优 `C*` | 全排列数 | 可行组合数 |
|---|---:|---:|---:|---:|
| d0_2x2_seed0 | 14 | 14 | 4 | 3 |
| d0_3x3_seed0 | 19 | 19 | 216 | 66 |
| d0_4x3_seed0 | 23 | 25 | 13,824 | 1,976 |

同时验证了**池界不是全局界**：用一个刻意削薄的池（每机器仅保留路线序与逆序）得到 `C_Π^* ≥ C*`，与拆解文档的作用域规则一致。

### 3.4 记录字段已冻结并强制

`validate_record` 拒绝：缺字段、`bound_scope="pool"` 却填 `global_lower_bound`、声明 `qpu` 却无提交作业、未通过独立排程验证的记录。量子模拟一律记 `quantum_execution="simulator"`，不会与真机混淆。

## 4. 本地验证记录

```
python -m unittest test_t00_control test_search_loop test_accelerated   → Ran 42 tests ... OK
python make_d0_data.py build --out-dir data --report reports/d0_verification.json
    → {"instance_count": 30, "files_written": 30, "round_trip_hash_matches": 30,
       "correctness_gate_passed": true, "seconds": 2.05}
python make_d0_data.py verify --out-dir data --report reports/d0_verification.json
    → {"passed": true, "checks": 13, "failures": []}
python t00_evidence.py --report reports/t00_evidence.json --seconds 3
    → reports/t00_evidence.json
```

环境：Python 3.12.8、NumPy 2.5.3、numba 0.67.0，CPU 后端；未提交任何 QPU 作业。

## 5. 未完成 / 边界

1. **未做性能主张。** 本文只证明"对照公平、成本可分离"。量子臂相对经典臂是否更好属 T06/T10，需要 §3.2 的统计协议。
2. **统计协议未实现**（bootstrap、确认集 100–119、预注册主指标），已在 `benchmark_protocol.md` §5 标注为待实现。
3. **`pool_sha256` 仍为 `None`**：动态池尚未逐 run 内容寻址，需在 T01 的冻结窗口设计确定后补齐。
4. **CUDA 计时口径**：`bench_cuda_baseline.py` 的累计计时与独立墙钟对照尚未在本机复跑（需 GPU 环境）；本次只在 CPU 后端验证，该子项仍待完成。
5. **`candidate_quantum_gates` 现已纳入 CI 覆盖**（应所有者要求，单列一个独立提交）：新增 `pip install numba` 与三步测试步骤，只跑不依赖 qiskit 的 `test_t00_control`、`test_search_loop`、`test_accelerated`；`test_circuits.py` 仍需 qiskit，暂不纳入。
6. **D1 历史口径未重跑**：`candidate_adaptive_results_20261002` 的均匀列基于旧路径，**不得与新路径数字直接比较**；需要重跑时用 `legacy_uniform=True` 复现旧口径。

## 6. 对下游任务的影响

- T01（glm）可直接使用 `data/` 的 D0 与 `manifest.json`；精确最优必须留在诊断层，不得作为在线答案。
- T02（codex）可用 `experiment_manifest.validate_record` 统一记录下界与作用域。
- 任何改动 `adaptive_search.py` 的 PR 需注意 `uniform` 语义已变更；复现历史数字必须显式传 `legacy_uniform=True`。
