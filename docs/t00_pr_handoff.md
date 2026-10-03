# T00 交付与审查说明（Issue #19 分工）

任务：拆解文档 §5 **T00 数据身份、时间口径与独立经典对照（P0）**，按 Issue #19 最终分工由 deepseek 实施（见 Issue #19 评论 5967757739 与 glm 的 5967797615）。

- 分支：`feat/t00-uniform-control`
- base：`e24f6bd`（PR #18 合并后）
- 提交：4 个（采样器 → 数据与记录 → 测试与证据 → 文档）
- 编码风格：与 `candidate_quantum_gates` 现有模块一致（3.12 语法、仅依赖 NumPy/numba/scipy，不引入新依赖）

## 1. 本次解决的问题

1. **均匀对照不独立（主要问题）。** `adaptive_search.solve` 的 `uniform` 模式原与量子臂共用 `CompactSimulator`，被迫先物化 `∏_m K_m` 个合法标签并逐条计算见证掩码，再抽样。这给经典对照附加了与量子模拟同阶的准备成本，使"经典 / 均匀 / 量子"三臂比较在成本上不成立。
2. **实例身份不可核验。** 缺一份可复算的分层数据与哈希清单，历史结果无法确认是否同实例。
3. **记录字段不统一。** 各任务自组 JSON，界的适用范围（池内/全局）无强制字段，真机与模拟容易混淆。

## 2. 交付清单

| 文件 | 说明 |
|---|---|
| `code/candidate_quantum_gates/uniform_sampler.py` | 独立均匀采样器（新增） |
| `code/candidate_quantum_gates/adaptive_search.py` | `uniform` 路径改造 + 成本计数字段 + `legacy_uniform` 开关 |
| `code/candidate_quantum_gates/data_identity.py` | 实例身份、D0 生成器、全排列精确最优、受限池最优（新增） |
| `code/candidate_quantum_gates/make_d0_data.py` | D0 生成与磁盘核验 CLI（新增） |
| `code/candidate_quantum_gates/experiment_manifest.py` | 记录字段集的唯一构造与校验入口（新增） |
| `code/candidate_quantum_gates/t00_evidence.py` | 成本分离证据脚本（新增） |
| `code/candidate_quantum_gates/test_t00_control.py` | 24 项验收测试（新增） |
| `code/candidate_quantum_gates/data/` | D0 30 个实例 + `manifest.json` + 隔离的精确最优诊断 |
| `code/candidate_quantum_gates/reports/` | `d0_verification.json`、`t00_evidence.json` |
| `docs/benchmark_protocol.md` | 记录字段、计时与身份口径、界的适用范围（新增） |
| `docs/t00_uniform_control_20261003.md` | T00 报告（新增） |
| `README.md` | 按 AGENTS.md 格式新增第 6 节贡献说明与复现命令 |

## 3. 验收条件对照

| 拆解文档 T00 验收要求 | 落实方式 |
|---|---|
| 数据哈希、实例身份、上界验证覆盖率 100% | `make_d0_data.py verify` 13 项检查；30/30 往返哈希一致；`enumerate_exact_optimum` 的独立验证标记 |
| 历史模式固定迭代/种子/软件环境可重现轨迹 | `legacy_uniform=True` 保留旧路径；新路径同种子确定 |
| 均匀采样分布用小池计数与理论乘积分布核验 | `test_empirical_distribution_matches_exact_product_law`（χ² + 逐机边缘分布 + 解析概率） |
| 计时字段可与外部墙钟核对 | 记录分列 `setup_seconds` / `search_seconds` / `wall_seconds` 与 `end_to_end_seconds`；**CUDA 累计计时的本机复跑仍待完成** |
| 不得保留只因实现额外开销而弱化的主基线 | 均匀对照准备成本降为 `O(machines)`；旧路径成本单列可核验 |
| 产物：manifest、协议、独立 sampler、统一 runner | `data/manifest.json`、`benchmark_protocol.md`、`uniform_sampler.py`、`experiment_manifest.run_and_record` |

## 4. 证据（可复核）

```
python -m unittest test_t00_control test_search_loop test_accelerated   # Ran 42 tests ... OK
python make_d0_data.py build  --out-dir data --report reports/d0_verification.json
python make_d0_data.py verify --out-dir data --report reports/d0_verification.json  # passed: true, 13 checks
python t00_evidence.py --report reports/t00_evidence.json --seconds 3
```

关键数字见 `docs/t00_uniform_control_20261003.md` §3：同墙钟下提议调用 610 → 827，最优工期同为 25，两路径均通过独立排程验证；准备成本在 8 机池上相差 19.8×，池超限时旧路径直接拒绝构造而独立采样器照常工作。

## 5. 明确不做的主张与剩余风险

1. **不主张任何搜索质量或量子优势结论**；本 PR 只是让对照公平、成本可分离。
2. **统计协议未实现**（bootstrap、确认集 100–119、预注册主指标），已在 `benchmark_protocol.md` §5 明示。
3. **CUDA 计时口径未在本机复跑**（需 GPU 环境），仅 CPU 后端验证。
4. **`pool_sha256` 仍为 `None`**：动态池尚未逐 run 内容寻址，待 T01 冻结窗口设计确定后补齐。
5. **CI 未覆盖 `candidate_quantum_gates`**：现有 workflow 只跑 `candidate_quantum_demo` 与 `candidate_quantum_medium`；本次测试为本地运行结果，是否扩 CI 请所有者决定（未擅改 workflow）。
6. **`uniform` 语义已变更**：复现 `candidate_adaptive_results_20261002` 等历史数字必须显式传 `legacy_uniform=True`。
7. **CI 改动单列为一个独立提交**（`.github/workflows/ci.yml`），不属于 T00 本体，是应所有者要求单独加的：新增 `pip install numba` 与 `candidate_quantum_gates` 测试步骤（只跑不依赖 qiskit 的 `test_t00_control`、`test_search_loop`、`test_accelerated`）。若审查者认为应拆成独立 PR，可直接 revert 该提交而不影响 T00 其余内容。

## 6. 请审查者重点看

1. `uniform_sampler.py` 的抽样律是否确实等于 `∏_m 1/K_m`（逐机独立、无耦合、无拒绝）。
2. `adaptive_search.py` 的分支是否保证三臂"同池、同见证、同动作、同预算"。
3. `data_identity.enumerate_exact_optimum` 的穷举空间与独立验证是否足以支撑"D0 正确性 100%"这句话。
4. `experiment_manifest.validate_record` 的四条拒绝规则是否有遗漏或过严。

## 7. 与 glm / codex 的冲突面

- 本分支**只改** `adaptive_search.py` 的 uniform 路径并新增独立模块，未触碰 `search_loop.py`、`circuits.py`、`compact_simulator.py`，与 glm 的 T01 pilot 无文件重叠。
- `data/` 是新增目录，T01 可直接读取；请勿把 `data/window_oracle_outputs/d0_exact_optima.json` 接入任何在线算法。
- codex 的 T02 若需要记录接口，请直接调用 `experiment_manifest.build_record` / `validate_record`，不要另建字段名。
