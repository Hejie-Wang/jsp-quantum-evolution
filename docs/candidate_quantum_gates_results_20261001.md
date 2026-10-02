# Candidate Quantum Gates: Local Results

日期：2026-10-01

本阶段已完成 Kimi 候选池设计的门级离线实现。所有 Python 运行于 conda
`qskit`（Qiskit 2.5.2、Aer 0.17.2），未导入 IBM provider/runtime，未读取
账户或凭据，未发现云 backend，`hardware_jobs_submitted=0`。

## 已实现

- `CandidatePool`、pool 内容哈希和 pool-version 绑定的 `WitnessSpec`。
- one-hot 候选位上的 witness membership、cycle/path 合取相位，并反计算
  work 位；路径支持固定 T 和小端序 T 寄存器的 `T < length` 比较。
- Qiskit `XXPlusYYGate` 单机 XY 混合器，保持每台机器单激发数。
- 由 witness 提议的 2--4 机器联合局部两能级旋转；不读取最优标签，支持
  外部机器上恒等作用。
- 原始 counts 解码：非法 one-hot 样本被计数和丢弃，不做 argmax 修复。
- 本地 Aer 小样本入口，以及只做 `transpile` 的中规模资源入口。

代码入口：[circuits.py](../code/candidate_quantum_gates/circuits.py)、
[run_offline.py](../code/candidate_quantum_gates/run_offline.py)。

## 验证结果

`test_circuits.py` 共 8 项通过；既有 Kimi demo 6 项通过，中规模经典回归
22 项通过。门级测试覆盖 pool 版本错配、固定 T 相位、`T=L-1/L` 边界、
比较器反计算、XY/联合门幺正性、联合端点和非法编码处理。状态向量保护
在超过 20 qubits 时拒绝分配。

小 demo 使用本地 Aer、256 shots：8 个候选组合均保持合法 one-hot，非法样本
为 0。该结果只说明电路和解码链路可运行，不说明优化成功率或量子优势。

## 15x20 离线资源

输入为 ta21 前 15 个作业、全部 20 台机器的派生实例，K=8 候选池；它不是
公开的 15x20 benchmark。电路只纳入 44 条保存见证中的前 2 条和 2 个联合项，
因此是明确标注的部分 witness layer，未执行中规模状态向量模拟。

| 项目 | 结果 |
| --- | ---: |
| 候选数据 qubits | 160 |
| 可复用 work qubits | 46 |
| witness / omitted | 2 / 42 |
| phase logical qubits | 206 |
| phase logical ops | 68 CX, 2 MCX, 2 CCX, 2 P |
| phase transpiled | 128 CX, 118 U；depth 141 |
| joint logical | 2 local unitary；depth 2 |
| joint transpiled | 29,749 CX, 51,610 U；depth 58,746 |
| statevector simulation | false |

联合局部 `UnitaryGate` 的通用 basis-gate 展开代价很高，这是当前实际瓶颈，
不能把逻辑深度 2 当作硬件深度。结果文件为
[results_final_20261001.json](../code/candidate_quantum_gates/results_final_20261001.json)，
其中保存环境、输入 SHA256、见证省略数量和 `hardware_jobs_submitted=0`。

## 结论边界

本阶段证明了小规模门级语义和离线编译链路，不证明 15x20 已被量子电路求解，
不证明候选池覆盖原问题全局最优，也不声称联合混合器优于普通混合器。下一步
若继续研究，应优先压缩联合跃迁的受控门分解并在相同评价/时间预算下与经典
联合移动比较；仍不需要、也不应提交真机任务。
