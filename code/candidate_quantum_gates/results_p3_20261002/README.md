# P0–P3 运行结果与数据（2026-10-02）

本目录是 [改进交接](../../../docs/candidate_quantum_improvements_20261002.md)
P0–P3 实施的全部本地运行证据。环境：conda `qskit`（Qiskit 2.5.2、Aer 0.17.2、
NumPy 2.4.6、SciPy 1.17.1），所有运行 `hardware_jobs_submitted = 0`（无云提交）。
解读性结论见 [结果摘要](../../../docs/candidate_quantum_improvements_results_20261002.md)。

## 文件索引

| 文件 | 内容 | 生成命令 |
|---|---|---|
| `pool_milp_evidence.json` | 三个已保存候选池的池内最优 MILP 证据：池哈希核对、报告 `best_choice` 的独立验证、约束规模、求解状态、对偶界、解码后验证 | `pool_optimum_milp.py --report ... ×3` |
| `run_offline_20261002.json` | demo 采样分布对比（均匀 vs 相位+混合器）+ 15x20 派生池 44 条见证资源编译（183 逻辑位、转译 2056 CX）+ 环境版本 | `run_offline.py --medium-result .../15x20_derived_seed11_kaiwu.json` |
| `search_comparison.json` | 3 案例 × 4 模式对比：汇总行 + 每模式完整运行报告（逐轮 trace、训练参数、采样概率、电路资源）+ MILP 完整明细 | `search_comparison.py --cases demo_3x3 5x5 5x5 --seeds 7 11 13` |
| `demo_closed_loop_20261002.json` | demo_3x3 四种模式各自独立运行的完整闭环报告（互为对照） | `search_loop.py` 内 `run_search_loop` / `classical_joint_search` |

## 池内最优 MILP 证据（pool_milp_evidence.json）

| 池 | 原循环最好 | 池内最优 | 状态 | 对偶界 | U（已验证初始工期） | 耗时 |
|---|---:|---:|---|---:|---:|---:|
| 15×15，seed=7 | 1462 | 1462 | optimal | 1462 | 1462 | 2.5 s |
| 20×15，seed=7 | 1865 | **1835** | optimal | 1835 | 1865 | 22.8 s |
| 15×20 派生，seed=11 | 1675 | 1675 | optimal | 1675 | 1675 | 4.8 s |

三个数值与交接文档一致；`pool_sha256` 与保存报告逐一核对通过。20×15 池确认存在
30 的搜索改进空间（1865 → 1835）。

## demo_3x3 闭环（demo_closed_loop_20261002.json）

四种模式均从初始工期 16 改进到池内最优 **11**（与全枚举一致），所有接受排程
通过独立 `validate_schedule` 验证。逐轮 trace 中可查看：冻结见证数、联合动作、
训练参数与评价次数、采样概率 top-5、电路资源、每轮耗时。

## 多模式对比（search_comparison.json，rounds=4，shots=256）

| 模式 | 目标成功率 | 有改进案例 | 平均墙钟 | 平均训练成本 | 平均图评价 |
|---|---:|---:|---:|---:|---:|
| 均匀采样 | 3/3 | 1/3 | 1.3 s | 0 | 30.3 |
| 仅 XY | 3/3 | 1/3 | 159.0 s | 152.5 s | 30.3 |
| XY+联合跃迁 | 3/3 | 1/3 | 236.9 s | 226.2 s | 30.0 |
| 经典（同信息同动作） | 3/3 | 1/3 | 0.04 s | 0 | 12.7 |

诚实备注（数据原样保留，未做筛选）：

* 两个随机 5×5 池（seed=11/13）的 label-0 初始排程本身已是池内最优（89、119），
  属退化案例，任何方法都无法改进；唯一非退化的 demo_3x3 上四种模式全部改进成功。
* 小池上均匀采样已达到 100% 目标成功率；量子回路训练主导其墙钟成本；经典基线
  便宜约三个数量级。纠缠或低能量不构成量子优势证据。

## 检查要点

1. **相位语义**：`run_offline_20261002.json` 中 `distributions_differ=true` 且
   非法 one-hot 样本为 0——相位层单独不改变测量分布，混合器才是分布变化的来源。
2. **资源核算**：`medium_resources.witnesses_included=44/44`、
   `phase_logical.num_qubits=183`（160 数据 + 23 共享工作），对照旧布局的
   206 位/2 条见证。
3. **搜索闭环**：任取 `search_comparison.json` 中一个 `reports.xy_joint.trace`
   条目，核对“采样 → 图评价 → 见证加入 → T 更新”链条与
   `witnesses_frozen` 的单调增长。
4. **MILP 证书**：`milp.status="optimal"` 且 `mip_dual_bound == objective`
   才是最优证明；仅可行解不算（见文件内 `notes`）。

## 复现

```powershell
conda run --no-capture-output -n qskit python code/candidate_quantum_gates/pool_optimum_milp.py `
  --report code/candidate_quantum_medium/results/corrected_20261001/15x15_seed7_milp.json `
  --report code/candidate_quantum_medium/results/corrected_20261001/20x15_seed7_milp.json `
  --report code/candidate_quantum_medium/results/corrected_20261001/15x20_derived_seed11_milp.json `
  --output code/candidate_quantum_gates/results_p3_20261002/pool_milp_evidence.json

conda run --no-capture-output -n qskit python code/candidate_quantum_gates/run_offline.py `
  --medium-result code/candidate_quantum_medium/results/corrected_20261001/15x20_derived_seed11_kaiwu.json `
  --output code/candidate_quantum_gates/results_p3_20261002/run_offline_20261002.json

conda run --no-capture-output -n qskit python code/candidate_quantum_gates/search_comparison.py `
  --cases demo_3x3 5x5 5x5 --seeds 7 11 13 `
  --output code/candidate_quantum_gates/results_p3_20261002/search_comparison.json
```

种子固定，因此对比实验的优化结果可精确复现（墙钟计时会有正常波动）。
