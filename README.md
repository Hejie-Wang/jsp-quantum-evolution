# JSP 排列空间量子演化：从经典模拟到 IBM 真机验证

本仓库记录作业车间调度问题（JSP）排列空间量子演化模型的完整研究链路：数学建模 → 经典状态向量模拟 → Qiskit 电路迁移 → IBM Quantum 真机实验 → 交换网络并行化优化。

## 目录结构

| 路径 | 内容 |
|---|---|
| `JSP问题建模思路.md` | 问题的数学建模思路 |
| `算法性能分析.md` | 量子/经典/FPGA 三路性能分析与真机资源预算 |
| `算法改进方向.md` | 算法改进方向 |
| `docs/` | 数学推导与验证报告、量子演化代码运行说明、能隙与资源推导 |
| `qskit/` | IBM Quantum Hello World 复现：Bell 态与 100 比特 GHZ 真机实验（含结果图） |
| `code/JSP_算法原型与验证结果/` | JSP 经典算法原型与大规模验证（禁忌搜索基线等） |
| `code/JSP_量子演化可运行代码/` | 排列空间量子演化的经典状态向量模拟器（`qjsp.py`，8 项单元测试） |
| `code/JSP_受限量子资源优化代码/` | 受限量子资源下的优化代码与编译报告 |
| `code/jsp_qiskit_hardware/` | **本仓库核心**：2×2 JSP 的 Qiskit 电路迁移、真机结果、交换网络并行化 |

## 关键结果

### 1. IBM Quantum Hello World（`qskit/`，后端 ibm_fez，156 比特）

- **2 比特 Bell 态**（Job `dau66k2hcrkc73dv36kg`）：⟨ZZ⟩=0.940、⟨XX⟩=1.040，单比特期望值≈0，纠缠特征明确
- **100 比特 GHZ 态**（Job `dau66qlvr3kc73emd8s0`，XY4 动力学解耦）：⟨Z₀Zᵢ⟩ 随距离从 1.0 衰减至 ~0，直观展示噪声对大规模纠缠的破坏

### 2. 2×2 JSP 真机实验（`code/jsp_qiskit_hardware/`）

紧凑分机器排列编码（2 比特），成本相位精确分解为 `Rz+Rz+Rzz`，交换混合器即 `RX` 旋转；电路与 NumPy 演化逐振幅一致（误差 <1e-15）。ibm_fez 上 4 组少层电路（1024 shots/组）：

| 层数 | 双比特门 | 调度时长 | TV 距离 | 真机最优概率 / 理想 | 测得最优工期 |
|---|---:|---:|---:|---|---:|
| 1 | 2 | 1.9 µs | 0.090 | 0.361 / 0.419 | 18（最优） |
| 2 | 4 | 2.2 µs | 0.177 | 0.072 / 0.184 | 18 |
| 5 | 10 | 3.0 µs | 0.414 | 0.254 / 0.027 | 18 |
| 10 | 20 | 4.2 µs | 0.147 | 0.622 / 0.693 | 18 |

4 组实验全部在真机上测到最优排程（makespan=18）。分布对比见 `code/jsp_qiskit_hardware/hw_vs_ideal.png`。

### 3. 交换网络并行化（`code/jsp_qiskit_hardware/qjsp_parallel.py`，纯离线验证）

按奇偶边着色将每层 M(J−1) 个串行相邻对换压缩为 ≤2 个并行级（跨机器天然对易 + 机内不相邻对换对易）：

- 等价性：2×2、3×3 与串行演化**逐振幅完全相等**（TV=0，机器精度）；4×3 差异 TV≈1.5×10⁻⁴
- 深度压缩：50×20 实例每层 980 级 → 2 级（490×）
- 离线预测：`FakeFez` 噪声模拟与真机实测对比验证了假后端作为定性筛查工具的可用性（`parallel_tv_comparison.png`）

### 4. 经典模拟基线（`code/JSP_量子演化可运行代码/`）

Intel Xeon 8370C、单线程、τ=20、2000 层、1000 次采样：2×2 总耗时 0.025 秒（最优概率 82.8%）；3×3 0.077 秒（14.8%）；4×3 2.17 秒（0.12%）。50×20 实例的完整振幅演化需要 ~10^1290 字节，由容量检查明确拦截。

## 复现

```bash
# 经典模拟器（Python 3.10+，仅依赖 NumPy/SciPy）
cd code/JSP_量子演化可运行代码/jsp_quantum_evolution
pip install -r requirements.txt
python qjsp.py simulate --instance data/demo_3x3.json --output run.json
python -m unittest -v test_qjsp.py

# Qiskit 硬件链路（需 qiskit、qiskit-ibm-runtime、qiskit-aer）
cd code/jsp_qiskit_hardware
python qjsp_hardware.py --layers 1              # 理想态矢量验证
python qjsp_hardware.py --layers 1 --submit     # 提交真机（需已保存 IBM Quantum 凭证）
python qjsp_parallel.py                         # 并行化离线验证（不消耗 QPU）
```

## 安全说明

IBM Quantum API 凭证文件（`import.py`）已按 `.gitignore` 排除，请勿上传含明文令牌。真机作业 ID 保留在结果 JSON 中，可在 IBM Quantum Platform 上核验。

## 机器候选量子搜索设计（2026-10-01）

基于机器候选编码、跨机器环/路径投影与联合混合器的新路线：

- [数学设计与理论边界](docs/quantum_candidate_design.md)
- [本地 Codex 实现任务与验收条件](docs/quantum_candidate_codex_tasks.md)
- [最小可运行 demo 与验证结果](code/candidate_quantum_demo/README.md)

Demo 验证候选池内基态能量与最优工期对应，并给出约束生成和理想演化示例；尚未实现 CUDA、门级编译或大规模量子求解，未声称量子优势。

