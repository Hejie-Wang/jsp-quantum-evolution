**动态机器候选与批量评价：实现及对照结果（2026-10-02）**

基线提交：`10682e461f36e5c1a879858d249e3f2fd52ea252`。本次保留现有图评价、见证投影和量子门语义，新增可运行搜索及 CPU/CUDA 批量评价。结论：动态搜索显著优于当前固定候选池线路；本次量子模拟没有显示稳定的额外优势。

**实例和比较口径**

逐项比较工序时间与机器路线，确认仓库文件对应 ta01、ta11、ta21、ta61、ta62、ta63；后三个是全部三个 50×20 文件。公开上下界来自 [JSPLib](https://raw.githubusercontent.com/ScheduleOpt/benchmarks/main/jobshop/solutions/bks.json)，并与 [Optimizizer](https://optimizizer.com/TA.php) 交叉核对，六例 LB=UB。文件哈希、来源、匹配结果见 [benchmark_bounds.json](../code/candidate_quantum_gates/benchmark_bounds.json)。15×20 截取实例未套用 ta21 上界。

相对差距定义为 `100 × (C / UB - 1)`。相同实例/种子的三个动态模式使用同一个 240 次调度生成的初态；每次搜索预算 10 秒，初始化和 JIT 单列并计入端到端时间。下表固定 seed=7，旧固定池 MILP 也在同一环境补跑 10 秒；其算法未修改。

| 实例 | 已知最优 | 固定池 MILP | 动态经典 | 动态量子模拟 | 动态经典距最优 |
|---|---:|---:|---:|---:|---:|
| ta01，15×15 | 1231 | 1462 | 1281 | 1280 | 4.06% |
| ta11，20×15 | 1357 | 1865 | 1427 | 1441 | 5.16% |
| ta21，20×20 | 1642 | 2113 | 1748 | 1725 | 6.46% |
| ta61，50×20 | 2868 | 3517 | 3022 | 3049 | 5.37% |
| ta62，50×20 | 2869 | 3466 | 3117 | 3166 | 8.64% |
| ta63，50×20 | 2755 | 3249 | 2852 | 2945 | 3.52% |

相对同预算固定池结果，动态经典工期下降约 10.1%–23.5%。所有动态模式另跑 seed=11，共 36 组，加 6 组固定池参照；没有筛除退化或失败案例。

| 模式 | 12 组平均相对差距 | 50×20 三例各自最好工期（两种子） |
|---|---:|---|
| 动态经典 | 5.73% | 2973 / 3117 / 2852 |
| 动态 + 均匀组合提议 | 6.36% | 3010 / 3102 / 2931 |
| 动态 + 量子模拟提议 | 6.43% | 3018 / 3118 / 2945 |

量子提议有 17 次直接改善历史最好工期，但综合结果仍不优于经典模式；不能将经典搜索的收益归因于量子计算。这里只测试两个种子，不作统计显著性结论。计时停止使迭代数随机器负载变化；若需确定性轨迹，可使用足够大的时间预算并固定 `--iterations`。

仓库较早的 C++ 禁忌搜索曾在 ta62 达到 3051（记录预算 60 秒），优于本次 10 秒最好 3102；本次没有声称击败所有既有经典算法。较早 `ideal_quantum` 的三个 50×20 工期为 3903/4048/4015，但其预算、初态和环境不同，仅作历史质量参照，见 [历史记录](../code/candidate_quantum_gates/results_adaptive_20261002/historical_quality.json)。

**实际修改**

- `adaptive_search.py`：从所有关键机器块生成边界相邻动作，每 10 轮扩展为非相邻插入；配合禁忌记忆、特赦和停滞重启，持续更新整台机器的候选顺序，打破固定池上限。始终保存并独立验证历史最好排程。
- 每 40 轮可对最多 6 台活跃机器、每机最多 3 个候选进行联合提议；其余机器固定，但完整作业图和跨机器见证仍参与评价。每次换池重新投影见证，当前顺序保留为候选 0。活动见证最多 96 条，不把省略的约束视作已执行。
- `compact_simulator.py`：在合法候选基底上精确模拟现有相位、XY、联合旋转，避免分配非法 one-hot 状态及干净辅助位。只计算已有见证谓词，不枚举完整调度成本表；状态数硬上限 65,536。本次动态窗口最多 729 个状态，属于容易经典模拟的实验范围，不能据此宣称量子优势。
- `search_loop.py`：训练和采样默认使用上述精确模拟；保留 `simulation_backend="qiskit"` 作为原门级参考，训练传入实际混合模式并检查截止时间。门语义及理想分布未改变。
- `batch_evaluator.py` / `evaluate_batch.cu`：统一批量有向图评价。CPU 使用 Numba；CUDA 使用 CuPy 加载实际内核，按批分块、检测环、计算最早工期，统计传输和同步时间。显式 `cuda` 不可用则报错；`auto` 记录回退原因。

**加速与验证边界**

同进程、预热后的三次测量中位数：ta61 的 256 个邻域组合，原 Python 图评价 257.44 ms，Numba 批量评价 3.59 ms，约 **71.7 倍**；demo 同一 24 次参数评价，原 Qiskit 训练 93.83 ms，压缩模拟 1.23 ms，约 **76.5 倍**，参数和目标值一致。这些是组件的经典加速，不能当作端到端量子加速。

38 项门级及加速测试中 37 项通过，1 项设备 CUDA 测试因没有 GPU/CuPy 跳过。已实际编译并在 CPU 执行 `.cu` 内核主体，核对索引和环语义；这不能替代 NVRTC 编译与 GPU 实测。42 份结果均重新通过独立排程验证。当前没有 NVIDIA 设备，也没有提交 QPU 作业，因此未报告 GPU 加速比或量子硬件性能。

完整结果、验证和计时见 [实验目录](../code/candidate_quantum_gates/results_adaptive_20261002/summary.json)、[固定池对照](../code/candidate_quantum_gates/results_adaptive_20261002/fixed_pool/summary.json)、[组件计时](../code/candidate_quantum_gates/results_adaptive_20261002/components.json)。

**复现与下一步**

```bash
python -m pip install -r code/candidate_quantum_gates/requirements-speed.txt
python -m unittest discover -s code/candidate_quantum_gates -p 'test_*.py' -v
python code/candidate_quantum_gates/benchmark_adaptive.py --seconds 10 --output results/adaptive
python code/candidate_quantum_gates/benchmark_components.py task_data/tai50_20_01.txt --output results/components.json
# NVIDIA 环境安装匹配 CUDA 版本的 CuPy 后：
python code/candidate_quantum_gates/adaptive_search.py task_data/tai50_20_01.txt --mode quantum --backend cuda --seconds 30 --output results/ta61_cuda.json
python code/candidate_quantum_gates/benchmark_components.py task_data/tai50_20_01.txt --cuda --output results/components_cuda.json
```

当前默认使用动态经典搜索，量子提议作为显式实验选项。后续量子改进应优先检验超工期路径的长度加权、跨轮参数复用及按实际收益调整提议频率；这些尚未实现或验证。扩展全机器相干搜索时仍需重新核算门深度和硬件噪声，不能把当前小窗口的模拟成绩外推为 50×20 全局量子优势。
