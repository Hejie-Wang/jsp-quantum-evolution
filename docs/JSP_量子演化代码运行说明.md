# JSP 排列空间量子演化：可运行代码与实测结果

版本：2026-09-30。Python 3.10+。这是此前排列编码、成本相位和相邻交换演化的**经典状态向量模拟实现**。

已经实现：读取 JSP → 构建排列基 → 计算排程成本 → 制备均匀叠加 → 逐层复数振幅演化 → 测量采样 → 解码与独立校验排程 → 导出结果。另有小实例能谱计算、大实例资源估算及容量拦截。

目前可运行小实例，不能据此宣称已解决你的三个 50×20 实例的全局最优问题。代码没有调用量子硬件，没有实现硬件上的可逆成本电路，也没有用经典优化器替换演化步骤。

## 1. 直接运行

解压后进入本目录：

```bash
python -m pip install -r requirements.txt

python qjsp.py simulate --instance data/demo_3x3.json --tau 20 --layers 2000 --shots 1000 --seed 7 --progress --output my_run.json --schedule-csv my_schedule.csv
```

默认也是这个 3×3 实例，因此可以更短：

```bash
python qjsp.py simulate --output my_run.json --schedule-csv my_schedule.csv
```

程序打印 JSON；`--output` 另存报告，`--schedule-csv` 导出本次测量得到的最好可行排程。若全部测量均不可行，`best_measured_schedule` 为 `null`，不会伪造排程或以枚举最优解代替测量结果。

Python API：

```python
from qjsp import Instance, solve

instance = Instance.read("data/demo_3x3.json")
report, final_state, model = solve(
    instance, tau=20, layers=2000, s_final=0.95,
    shots=1000, seed=7,
)
print(report["sampling"]["best_measured_schedule"])
print(report["timings_seconds"])
```

NumPy 用于运行；SciPy 只用于独立验证。此次环境为 Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0。

## 2. 实测需要多久

同一台 Intel Xeon Platinum 8370C CPU 环境，BLAS/OMP 线程数设为 1。固定 τ=20、终点 s=0.95、2,000 层、1,000 次采样、seed=7；每个实例运行 3 次，表中为总时间中位数，**不包含 Python 启动、依赖安装、文件写出**。完整逐阶段、逐次数据见 `results/benchmark.json`。

| 实例 | 排列基状态数 | 成本表构建 | 演化时间 | 总时间 | 本次采样最好工期 | 枚举诊断最优工期 |
|---|---:|---:|---:|---:|---:|---:|
| 2×2 | 4 | 0.000053 秒 | 0.0244 秒 | 0.0247 秒 | 18 | 18 |
| 3×3 | 216 | 0.000831 秒 | 0.0759 秒 | 0.0771 秒 | 11 | 11 |
| 4×3 | 13,824 | 0.0544 秒 | 2.1122 秒 | 2.1658 秒 | 31 | 30 |

各列分别取中位数，因此各阶段中位数之和不必等于总时间中位数。这些是特定数据、参数和环境的测量值，不是同规模所有 JSP 的保证时间。

| 实例 | 初始均匀分布的最优概率 | 演化后单次测量的最优概率 | 演化后可行概率 |
|---|---:|---:|---:|
| 2×2 | 25.00% | 82.82% | 99.98% |
| 3×3 | 0.9259% | 14.7869% | 57.4129% |
| 4×3 | 0.007234% | 0.117795% | 24.8495% |

4×3 这次固定种子的 1,000 次采样没有命中最优解。演化提高了此实例的最优概率，但有限演化和有限采样没有变成全局最优保证。重复计时使用相同种子，因此也不是 3 次独立成功率实验。

复现计时：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py --tau 20 --layers 2000 --repeats 3 --output-dir rerun_results
```

上述线程环境变量写法适用于 Linux/macOS shell；Windows 可直接运行 `python benchmark.py`，线程设置和耗时可能不同。

## 3. 前面模型如何对应代码

每台机器的基是该机器全部工序排列。完整基是机器排列的笛卡尔积。经典仿真显式保存其复数振幅。

- `Instance`：读取加工时间和机器路线；工序 ID 为 `job * machines + position`，从 0 开始。
- `evaluate_orders`：把作业先后边与机器相邻边合并，拓扑计算最早开始时刻。有环返回惩罚 `U+1`，无环返回 makespan。
- `PermutationModel`：建立排列基、成本对角表及相邻交换映射。映射满足 `S²=I`。
- `uniform_state`：在完整排列基上制备均匀叠加，包括造成全局有环的排列组合。
- `apply_layer`、`evolve`：成本相位及相邻交换酉旋转。
- `sample_schedule`：按振幅模平方抽样，仅从测得的结果中选择最好可行排程。
- `validate_schedule`：直接检查机器不重叠与作业先后约束，独立于拓扑解码。

使用原先定义的规范化：

\[
H(s)=-\frac{1-s}{d}\sum_{m,k}S_{m,k}+s\operatorname{diag}(F-L_0),\qquad d=M(J-1).
\]

这里 `L0` 仅为作业总时长和机器总负载给出的有效下界，**不是提前算出的最优值**。减去它只改变整体相位。单层代码对应：

```python
psi = psi * np.exp(-1j * s * h * potential)
beta = (1 - s) * h / number_of_swaps
for swap in swaps:
    psi = np.cos(beta) * psi + 1j * np.sin(beta) * swap_state(psi, swap)
```

实际实现还处理只有一个作业、没有交换生成元的情况。`h=tau/layers`，`s` 取每个时间层的中点；交换按机器编号、位置编号递增施加。这是明确的**一阶分裂演化**，不是精确的连续时间传播子。

演化始终操作整体复数振幅；全局成本相位可以建立不同机器寄存器之间的纠缠。这里没有把量子演化替换成独立抽取机器排列的经典概率更新。

完整成本表在模拟前枚举得到，消耗实际经典时间。它方便实现和诊断，**不能被当作免费的量子成本预言机**。代码输出的 `enumerated_optimum_diagnostic`、`optimal_probability_diagnostic` 只用于评价，不用于初态、演化参数调节或补齐采样结果。

## 4. 如何调整参数

| 参数 | 作用 | 注意事项 |
|---|---|---|
| `--tau` | 设定无量纲演化时长 | 增大不保证有限步下成功率单调上升 |
| `--layers` | 分裂演化的层数 | 固定 τ 下增大通常减少离散化误差，并增加运行时间 |
| `--shots` | 末态测量次数 | 只增加采样机会，不改变末态的最优概率 |
| `--s-final` | 演化终点，默认 0.95 | 接近 1 时仍须关注真实谱与演化误差 |
| `--seed` | 抽样随机种子 | 也用于没有输入文件时的随机示例生成 |
| `--max-states` | 全状态模拟容量，默认 200,000 | 超过时在枚举前拒绝；提高上限不会改变指数增长 |
| `--max-memory-mib` | 计划内存预算，默认 512 MiB | 是保守预算检查，不是操作系统的硬内存限制 |

3×3 的步数加倍检查：2,000 层和 4,000 层的最优概率分别为 0.14786851 和 0.14784228，测量分布的总变差距离约 0.00012415。它支持该小实例的数值稳定性，不能作为所有实例的连续演化误差证书。详见 `results/demo_3x3_refinement.json`。

CPU 模拟可以保存一次末态并廉价抽样很多次。真实量子设备每次测量后必须重新制备和演化，不能把模拟中的 `sampling_and_decode` 时间直接用于量子硬件耗时预测。

τ 不是墙钟秒数。如果物理哈密顿量为 `E_ref * H`，理想连续演化物理时间满足 `T=ħτ/E_ref`；当前没有指定设备能量尺度、门时长、可逆成本电路或纠错开销，因此不能给出真实量子设备秒数。

## 5. 三个 50×20 数据现在能做什么

输入文件已包含在 `data/` 中。读取、资源估算和给定机器顺序的排程解码可以处理 1,000 道工序；完整振幅演化不能在现实资源下运行。

```bash
python qjsp.py estimate --instance data/tai50_20_01.txt --output large_resources.json
python qjsp.py simulate --instance data/tai50_20_01.txt
```

第二条命令会在分配排列基之前明确返回 `CapacityError`，退出码 2。三个用户实例都实际检查了这一行为，见 `results/large_capacity_checks.json`。

这三个实例均有：

\[
D=(50!)^{20},\quad \log_{10}D=1289.66149745.
\]

仅用 complex128 存储振幅就需要约 **7.34×10^1290 字节**，尚未计入成本表和工作数组。此时无法给出有意义的经典全状态模拟完成时间。

紧凑的分机器排列编号需要 4,300 个逻辑编码比特；这是编码位数，不包含可逆图解码、成本计算、工作寄存器或纠错，不能理解为“4,300 个量子比特即可完成算法”。

当前模拟器的主要计算量约为：

\[
O(DN)\text{ 构建成本表},\qquad O(LDd)\text{ 演化},
\]

其中 N 为工序数，L 为演化层数。全状态内存随 D 增长。因而小实例的毫秒、秒级数据不能线性外推到 50×20。

如果已有一个大实例的机器顺序，可单独解码：

```bash
python qjsp.py decode --instance data/tai50_20_01.txt --orders my_orders.json --output decoded.json --schedule-csv decoded.csv
```

`my_orders.json` 格式为 `{"orders_zero_based": [[...], [...], ...]}`：外层按机器 0 到 M−1 排列，内层为对应机器工序 ID 的完整排列。解码评估的是指定顺序，不会搜索更优顺序。它也可作为未来硬件成本电路的经典对照实现。

## 6. 输入格式

文本输入兼容这次上传文件：前 J 行为加工时间，后 J 行为机器路线；每行 M 个整数，机器编号为 1…M，每个作业恰好经过每台机器一次。当前版本限定于这一标准 JSP 输入。

JSON 示例：

```json
{
  "name": "demo_3x3",
  "durations": [[3, 2, 2], [2, 1, 4], [4, 3, 1]],
  "machines_1_based": [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
}
```

CSV 排程输出中的作业、工序、机器编号从 1 开始，`start`、`finish`、`duration` 使用输入加工时间的同一单位。

## 7. 能谱与验证

```bash
python qjsp.py spectrum --instance data/demo_3x3.json --points 41 --output gap_grid.json
python -m unittest -v test_qjsp.py
```

能谱接口只允许 D≤256。3×3 在 41 个取样点上测得最小能隙约 0.000207593，位于 s=0.95。**网格最小值不是连续路径最小能隙的认证下界**，程序不会据此自动宣称绝热运行时间有保证。

8 项测试已通过，包括：

- 3×3 全部 216 个排列组合，64 个可行，最优工期 11；输出排程独立验证。
- 交换算符自反性、哈密顿量厄米性及均匀初态。
- 非对易交换下的单层演化与精确矩阵指数比较，验证局部误差阶。
- 完整演化与高精度薛定谔方程数值积分比较，验证步长收敛。
- 容量检查先于排列枚举；测量结果不被未测到的枚举最优解替换。
- 端到端复现、全部采样不可行、单作业边界与非法输入。

证据文件：`results/verification.txt`。三组计时实例的最大记录态范数平方漂移小于 10^-12；演化中没有用重新归一化掩盖漂移。

## 8. 文件索引

| 文件 | 用途 |
|---|---|
| `qjsp.py` | 完整算法、排程解码、命令行 |
| `test_qjsp.py` | 独立数学检查及端到端检查 |
| `benchmark.py` | 可复现计时与用户数据资源估算 |
| `requirements.txt` | Python 依赖 |
| `data/` | 小实例及三个原始 50×20 数据 |
| `results/` | 实测 JSON、排程 CSV、能谱、验证日志 |

这一版将算法构想变成了可执行、可检查的演化程序。若下一阶段要实际优化 50×20，需要另行实现可扩展的状态表示或受限子问题求解机制，并重新分析它对原模型和最优性保证的影响；当前版本没有把这项尚未完成的工作包装成已有能力。
