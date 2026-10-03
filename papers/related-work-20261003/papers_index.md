# OA 论文索引（Issue #13 文献调研，2026-10-03）

本目录收录本次调研中**可合法获取的开放获取（OA）版本**。共 **25 篇**，合计约 **31.7 MB**。

- 全部 PDF 的 **`%PDF` 魔数已校验**（HTML 错误页不会被存成 PDF）；
- 全部 PDF 的**首页标题已与引用逐条比对**（过程见文末"校验方法"）；
- 获取来源、许可依据、获取状态见下表；
- **未获取的条目列在文末**，标注原因，而不是留一个可能失效的链接。

配套分析见 [literature_survey.md](literature_survey.md)。

---

## 1. 已获取（25 篇）

### 1.1 与本仓库建模直接相关

| 文件 | 论文 | 出处 | 核实程度 | 与本题的关系 |
|---|---|---|---|---|
| `pdf/schmid2025efficient.pdf` | Schmid, Braun, Sollacher & Hartmann, *Highly Efficient Encoding for JSP and its Application on Quantum Computers* | [arXiv:2401.16381](https://arxiv.org/abs/2401.16381)；*Quantum Sci. Technol.* **10**, 015051 (2025) | **全文已核** | **压力最大的一篇**：qubit 数 $=\lceil\log_2(N_{\rm op}!/\prod_k\lvert J_k\rvert!)\rceil$，自称 variable-efficiency 最优；**明确放弃 QUBO** 并**不需要任何罚项** |
| `pdf/bourreau2024indirect.pdf` | Bourreau, Fleury & Lacomme, *Indirect Job-Shop coding using rank: application to QAOA (IQAOA)* | [arXiv:2402.18280](https://arxiv.org/abs/2402.18280)；*J. Heuristics* **32**(2) | 摘要页已核 | rank 编码 + **无环由 Bierwirth 向量构造性保证**，故无需写环约束 |
| `pdf/venturelli2015quantum.pdf` | Venturelli, Marchand & Rojo, *Job Shop Scheduling Solver based on Quantum Annealing* | [arXiv:1506.08479](https://arxiv.org/abs/1506.08479)；COPLAS/ICAPS-16 | 摘要页已核 | 首个时间索引 JSP QUBO；**≤6×6**，D-Wave Vesuvius，**不声称优势** |
| `pdf/doucet2026thermodynamic.pdf` | Doucet, Mzaouali, Robertson, Gardas, Deffner & Domino, *Thermodynamic significance of QUBO encoding on quantum annealers* | [arXiv:2601.04402](https://arxiv.org/abs/2601.04402)；*New J. Phys.* **28**, 054512 (2026)，**CC BY** | 摘要+期刊页已核 | **罚项设计的核心证据**：罚项太弱 → 低能不可行流形；太强 → 压低有效能标、加剧不可逆耗散。**不存在安全罚项区间** |
| `pdf/palackal2023graph.pdf` | Palackal, Richter & Hess, *Graph-controlled Permutation Mixers in QAOA for the Flexible Job-Shop Problem* | [arXiv:2311.04100](https://arxiv.org/abs/2311.04100) | 摘要页已核 | **把约束放进 ansatz 而非罚项**，保可行性且遍历可行子空间——本仓库"见证罚项"路线的直接替代 |
| `pdf/ayodele2022penalty.pdf` | Ayodele, *Penalty Weights in QUBO Formulations: Permutation Problems* | [arXiv:2206.11040](https://arxiv.org/abs/2206.11040)；EvoCOP 2022 | 摘要页已核 | 与 Doucet 独立同向：排列类 QUBO 罚权太小→不可行，太大→收敛慢 |
| `pdf/friedl2026succinct.pdf` | Friedl, Gegő, Kabódi & Nemkin, *Succinct QUBO formulations for permutation problems by sorting networks* | [arXiv:2603.07579](https://arxiv.org/abs/2603.07579) | 摘要页已核 | 用 compare-exchange 网络做排列 QUBO，仅 $O(n\log^2 n)$ 变量；**摘要不涉及调度，也无下界定理** |
| `pdf/nakano2023dual.pdf` | Nakano et al., *Dual-Matrix Domain-Wall: Generating Permutations by QUBO/Ising with Quadratic Sizes* | [arXiv:2308.01024](https://arxiv.org/abs/2308.01024)；*Technologies* **11**, 143 | 摘要页已核 | 把排列 one-hot 的最大绝对系数从 $2n-4$ 压到 **2**——若 $\Lambda>U$ 导致系数动态范围过大，可借用 |
| `pdf/coupvent2025updating.pdf` | Coupvent des Graviers, Kobrosly, Guettier & Cazenave, *Updating Lower and Upper Bounds for the JSP Test Instances* | [arXiv:2504.16106](https://arxiv.org/abs/2504.16106)，**CC BY-NC-SA** | 摘要页已核 | 2025 年仍用 OR-Tools 收紧 Taillard 上下界并关闭 ta33——**"经典已停滞"这一前提不成立** |

### 1.2 量子硬件/退火实现与评估

| 文件 | 论文 | 出处 | 核实程度 | 与本题的关系 |
|---|---|---|---|---|
| `pdf/carugno2022evaluating.pdf` | Carugno, Ferrari Dacrema & Cremonesi, *Evaluating the job shop scheduling problem on a D-wave quantum annealer* | *Sci Rep* **12**, 6539 (2022)，**CC BY 4.0**（出版方 PDF） | 摘要页已核 | 引用最多的诚实评估：编码编译代价、qubit 需求、chain break、reverse annealing 收益有限 |
| `pdf/safi2026reality.pdf` | Safi, Wintersperger, von Sicard, Niedermeier & Mauerer, *A Reality Check on Quantum Optimisation: Evidence from an Industrial Case Study* | [arXiv:2607.13325](https://arxiv.org/abs/2607.13325) | 摘要页已核 | IBM + D-Wave + Fujitsu DA 三平台工业 JSP 对照，**不声称优势**；其"分配 + 排序"两阶段架构与本仓库高度一致 |
| `pdf/amaro2022filtering.pdf` | Amaro, Rosenkranz, Fitzpatrick, Hirano & Fiorentini, *A case study of variational quantum algorithms for a job shop scheduling problem* | [arXiv:2109.03745](https://arxiv.org/abs/2109.03745)；*EPJ Quantum Technol.* **9**, 5 | 摘要页已核 | 四种 VQA 在同一 JSP 上对比；硬件规模到 **23 qubit**。**注：这不是 F-VQE 方法论文（那篇是 arXiv:2106.10055 / QST 7, 015021）** |
| `pdf/lopezruiz2025nonvariational.pdf` | Lopez-Ruiz, Tucker, Arnold, Epifanovsky, Kaushik & Roetteler, *A Non-Variational Quantum Approach to the Job Shop Scheduling Problem* | [arXiv:2510.26859](https://arxiv.org/abs/2510.26859) | 摘要页已核 | Iterative-QAOA，IonQ Forte 真机；目前最大规模的 JSP 真机工程参考 |
| `pdf/toma2024hybrid.pdf` | Toma, Zajac & Störl, *Solving Distributed Flexible Job Shop Scheduling Problems in the Wool Textile Industry with Quantum Annealing* | [arXiv:2403.06699](https://arxiv.org/abs/2403.06699) | 摘要页已核 | D-Wave QPU 上 **50–250 变量**（可嵌入上限）；讨论 Lagrange 参数与 QPU 配置 |
| `pdf/schworm2024quantumannealing.pdf` | Schworm, Wu, Klar, Glatt & Aurich, *Multi-objective Quantum Annealing approach for solving flexible job shop scheduling in manufacturing* | [arXiv:2311.09637](https://arxiv.org/abs/2311.09637)；*J. Manufacturing Systems* **72** | 摘要页已核 | QASA（tabu+SA+QA）；大实例按瓶颈因子分解 |
| `pdf/dalal2024digitized.pdf` | Dalal et al., *Digitized Counterdiabatic Quantum Algorithms for Logistics Scheduling* | [arXiv:2405.15707](https://arxiv.org/abs/2405.15707)；*Phys. Rev. Applied* **22**, 064068 | 摘要页已核 | DCQO：同双比特门数下成功率较 QAOA 提升数个量级；云超导/离子阱实测 |
| `pdf/sawamura2025decomposition.pdf` | Sawamura, Araki, Maruyama, Haba & Ohzeki, *Quantum-classical hybrid algorithm using quantum annealing for multi-objective job shop scheduling* | [arXiv:2511.03257](https://arxiv.org/abs/2511.03257)；*J. Phys. Soc. Japan* | 摘要页已核 | **按决策层次分解**：资源分配 → QUBO/退火，任务排程 → MILP 经典求解 |
| `pdf/benammar2025survey.pdf` | Osaba, Perez Delgado, Mata Ali, Miranda-Rodríguez, Moreno Fdez de Leceta et al., *Quantum Computing in Industrial Environments: where do we stand and where are we headed?* | [arXiv:2505.00891](https://arxiv.org/abs/2505.00891) | 摘要页已核 | 工业量子计算综述。（**注**：另有 Ben Ammar, Marzouki & Driss, *Quantum Computing for Scheduling Problems: A Survey*, AMCAI 2025, DOI 10.1109/AMCAI66110.2025.11474388——**该篇未获取 OA 版本**） |

### 1.3 方法背景与理论边界

| 文件 | 论文 | 出处 | 核实程度 | 与本题的关系 |
|---|---|---|---|---|
| `pdf/ohzeki2020breaking.pdf` | Ohzeki, *Breaking limitation of quantum annealer in solving optimization problems under constraints* | [arXiv:2002.05298](https://arxiv.org/abs/2002.05298)；*Sci Rep* **10**, 3126，**CC BY** | 摘要+DOI 已核 | **subQUBO 的源头**：Hubbard–Stratonovich 变换避免嵌入，变量分组迭代。测试对象**非 JSP** |
| `pdf/zhao2021hybrid.pdf` | Zhao, Fan & Han, *Hybrid Quantum Benders' Decomposition For Mixed-integer Linear Programming* | [arXiv:2112.07109](https://arxiv.org/abs/2112.07109)；IEEE WCNC 2022，**CC BY** | 摘要页已核 | "量子 Benders"原型；**未做 JSP** |
| `pdf/bartschieidenbenz2020grover.pdf` | Bärtschi & Eidenbenz, *Grover Mixers for QAOA: Shifting Complexity from Mixer Design to State Preparation* | [arXiv:2006.00354](https://arxiv.org/abs/2006.00354)；IEEE QCE'20 | 摘要页已核 | GM-QAOA：混合器换成相对可行态均匀叠加的 Grover 扩散算子，**无 Trotter 误差**；含排列问题的均匀叠加制备 |
| `pdf/marshall2019power.pdf` | Marshall, Venturelli, Hen & Rieffel, *Power of Pausing: Advancing Understanding of Thermalization in Experimental Quantum Annealers* | [arXiv:1810.05881](https://arxiv.org/abs/1810.05881)；*Phys. Rev. Applied* **11**, 044083 | 摘要页已核 | **解释"为什么惩罚型 QUBO 被经典热噪声兜底"**；本报告 §6.5 的热化假设依赖此机制 |
| `pdf/king2017quantum.pdf` | King, Yarkoni, Raymond, Ozfidan, King, Nevisi, Hilton & McGeoch, *Quantum Annealing amid Local Ruggedness and Global Frustration* | [arXiv:1701.04579](https://arxiv.org/abs/1701.04579) | 摘要页已核 | perturbed-Hamming-weight 型势垒 gadget 的常引文献 |
| `pdf/amin2009first.pdf` | Amin & Choi, *First Order Quantum Phase Transition in Adiabatic Quantum Computation* | [arXiv:0904.1387](https://arxiv.org/abs/0904.1387)；*PRL* **102**, 100401 | 摘要页已核 | 随机 Ising 最小能隙指数小的机制 |

---

## 2. 未获取（4 篇，均为 HAL 作者存缴）

以下 4 篇的**标题、作者、出处已核实存在**，但本次**未能取回 PDF**。原因：`hal.science` 对脚本请求（含多种 URL 形式与浏览器 User-Agent）一律返回 `text/html`，未提供 PDF 字节；属于站点反自动化访问，**不是这些文献不存在**。请人工在浏览器中打开对应 HAL 页面下载。

| 文献 | HAL 页面 | 尝试过的 URL 形式 |
|---|---|---|
| Deleplanque, Pérez Armas & Aggoune, *Quantum annealing heuristics for the job shop scheduling problem with availability constraints*, *J. Heuristics* **32**(2), art. 19 (2026)，DOI 10.1007/s10732-026-09594-5 | <https://hal.science/hal-04532209> | `/document`、`v1/document`、`v1/file/...`、`/file/...` |
| Aggoune, *ROADEF 2023* 报告 | <https://hal.science/hal-04028024> | `/document`、`v1/document`、`v1/file/...` |
| Aggoune & Deleplanque, *Solving the Job Shop Scheduling Problem: QUBO model and Quantum Annealing*（2023） | <https://hal.science/hal-04037312> | `/document`、`v1/file/2023_Quantum_Scheduling_AggouneDeleplanque.pdf` |
| Farhani, Arbaoui & Benatchba, *Manual vs. Automated QUBO Formulations for Flow Shop Scheduling*, GECCO'25 Companion | <https://hal.science/hal-05209619> | `/document`、`v1/file/...` |

最值得注意的是第一篇：它报告地平线收紧带来 **QUBO 规模 ↓29.9%、二次项 ↓46.4%、物理 qubit ↓约 50%、嵌入时间 ↓64%**，平均最优性 gap **18.3% → 5.0% → 3.3%**（M1/M2/M3）。**这些数字与"约束/地平线收紧能降低资源"直接相关，建议优先人工取回。**

---

## 3. 校验方法（可复核）

1. **魔数校验**：每个响应前 4 字节必须是 `%PDF`，否则丢弃并记录为失败——因此不会把 HTML 错误页当成 PDF 提交。
2. **标题比对**：用 `PyMuPDF`（`fitz`）抽取每个 PDF 首页文本，与预期标题逐条比对。**本次该步骤实际拦下一个错误**：`carugno2022evaluating` 最初使用了一个臆测的 arXiv 编号，下载到的是宇宙学论文 *Reconstructing teleparallel gravity with cosmic structure growth*；已改用 Scientific Reports 的 **CC BY 4.0** 出版方 PDF 重新获取并复核。
3. **体量与许可**：优先选择 arXiv 预印本（arXiv 非独占许可）或明确 CC BY / CC BY-NC-SA 的出版方版本；`papers/` 目录的用途见 [../README.md](../README.md)。

## 4. 关于"可信度标记"的重要说明

[literature_survey.md](literature_survey.md) 中使用三级标记：**【已核】/【二手】/【未核】**。

其中相当一部分文献的**内容层**（而非标题与摘要层）是由两个独立的调研子任务通过文本抽取代理阅读 PDF 得到的，本索引把它们标为**【二手】**。**本索引中的"核实程度"列只描述标题/出处层面，不代表内容层已被逐字复核。**

本仓库既有的 [docs/JSP_数学推导与验证报告.md](../../docs/JSP_数学推导与验证报告.md) 第 11 节与 [算法改进方向.md](../../算法改进方向.md) 已完成过一次有限文献筛查。**本目录是其补充，不是替代，也不构成全面新颖性检索。**
