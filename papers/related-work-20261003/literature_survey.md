# JSP 量子优化的相关研究、建模现状与"全局机器关联"实测报告

日期：2026-10-03
作者：`jsp-agent-deepseek[bot]`
起因：Issue #13《论文查询》
配套材料：同目录 [papers_index.md](papers_index.md)（24 份已获取 OA PDF 的索引、4 篇未获取文献、4 篇仅引用未下载）、`pdf/`（已获取的 PDF）

---

## 0. 阅读前须知：本报告的陈述纪律

按 [CONTRIBUTING.md](../../CONTRIBUTING.md) 的三类陈述纪律，本报告严格区分：

- **【已核】** = 本次实际打开原始页面（arXiv 摘要页 / 出版方页面 / PDF 正文）并核对过；
- **【二手】** = 由本仓库既有文档或调研子任务转述，未由本文档作者逐字复核；
- **【未核】** = 标题/出处存在，但内容层未读到，**不足以支撑任何结论**；
- **【实测】** = 由本文档作者在本仓库代码上运行并得到输出。

**边界声明（必须保留）**：

1. 本报告是**有限检索的记录，不是全面新颖性检索**，也不是新颖性证明。
2. **"在本次检索范围内未发现先例"不等于"该设想正确或原创"。** 任何由此产生的构想都按"待验证假设"记录。
3. Issue #13 的「核心假设」与「验证方案」两栏为空，本报告不代其补齐。
4. 本次全程**只读**仓库既有代码，未提交任何量子硬件作业（`hardware_jobs_submitted = 0`）。

---

## 1. 一句话结论

> JSP 量子优化存在 **60 年历史、生产级成熟的理论工具箱**，但它整体位于运筹学（OR）一侧；量子侧只有研究原型，**没有任何生产级库能把析取图 JSP 自动编译成 Ising/QUBO 并保证解质量**，也没有公开的、在 Taillard 基准上可与 SOTA 经典方法比拟的量子退火 makespan 记录。
>
> 对本仓库当前的"每机器候选顺序 + 跨机器环/路径见证 + 基态能量 = 最优工期"建模：**三个要素分别都已发表；合成单一 QUBO 且基态能量恰等于最优 makespan 的完整组合，在本次检索范围内未发现对应工作。** 但这不是新颖性证明，且必须在 related work 中精确区分三篇最接近的工作，否则会被"要素都已有"直接驳回。
>
> 对本仓库当前的"全局机器关联感知"能力：**【实测】它没有被任何现存指标度量过，并且在所有可测的地方其边际贡献为 0 或为负。**

---

## 2. 成熟工具盘点：哪个工具覆盖我们问题的哪一部分

| 我们问题中的部分 | 应借用的工具/框架 | 成熟度 | 现成可用性 |
|---|---|---|---|
| JSP 建模与可行性语义（最长路、关键路、环） | 析取图理论：Roy–Sussmann 1964【未核】、Balas 1969【二手】、Adams–Balas–Zawack 1988【已核】 | ★★★★★ 理论 | 需自实现（最长路/关键块代码简单） |
| 求解 makespan 最优/近似的**诚实基线** | OR-Tools CP-SAT；IBM ILOG CP Optimizer；OptalCP；PyJobShop | ★★★★★ | 立即可用 |
| MILP 表述（big-M 析取 / Manne 时间索引 / rank） | Manne 1960【二手】；Naderi–Ruiz–Roshanaei 2023【二手】 | ★★★★★ | 立即可用 |
| 标准实例与上下界对照 | JSPLib（实例 + `bks.json` + 统一 600 s 引擎基准）【已核】；Optimizizer【已核】 | ★★★★★ | 免费 |
| 经典强启发式上界 | i-TSAB / N1–N7 邻域（Nowicki–Smutnicki；Zhang 等 2006）【二手】 | ★★★★ | 需自实现 |
| GPU 大规模并行搜索 | Bożejko & Uchroński GPU 并行 TS【二手】 | ★★★★ | 研究代码 |
| 「选一个边方向」↔ 自旋变量 | Disjunctive Programming meets QUBO（2024）【未核】；Casati & Giudici 2025【二手】 | ★★ 原型 | **无库，需自写** |
| JSP → QUBO（时间索引罚项） | Venturelli 等 2015 的 $h_1,h_2,h_3$ + 剪枝 + ICP shaving【已核】 | ★★ 配方清楚 | 无库 |
| 量子/量子启发硬件执行 | D-Wave Leap（QPU + hybrid + 非线性 solver）；Fujitsu Digital Annealer；Fixstars Amplify | ★★★ 产品 | 付费/申请；**规模上限约数十至数百变量** |
| 分解/代理求解 | Benders；Lagrangian（Chen & Luh）；CP-LNS；滚动时域 | ★★★★★ 经典 / ★ 量子混合 | 立即可用（经典部分） |
| 硬度/势垒论证 | Amin & Choi 2009；Knysh–Smelyanskiy；King 等 2017；Marshall 等 2019 | ★★★ 理论 | **JSP 专用势垒测量 = 空白** |
| 评价指标（反"刷 BKS"） | JSPLib 的几何平均 LB/UB ratio 与 shifted gap【已核】 | ★★★★★ | 立即可用 |

### 2.1 标准基线：必须先建，否则任何量子结果不可信

JSPLib（<https://scheduleopt.github.io/benchmarks/jsplib/>）在**统一 600 秒 / 4 核**下公开的结果【已核摘要级】：

| 组 | CPLEX | CP Optimizer | CP-SAT | OptalCP |
|---|---:|---:|---:|---:|
| `ta` 家族（80 实例） | 证明最优 0 | **40** | 23 | **46** |
| 全部 376 实例 | 27 | 192 | 165 | 223 |
| 工业 `bel`（20） | 0 | 20 | 20 | 20 |

`ta` 家族中仍有 **12 个实例没有最优性证明**。

JSPLib 同时公开批评"刷 BKS"这一行为（原文："Announcing a best-known solution while having little scientific interest in itself (e.g. a random solution) has sadly become a central 'contribution' of papers."），建议改用**几何平均的 LB/UB ratio 与 shifted gap** 汇报。**建议本仓库直接采纳该口径**，否则"我们改善了 ta62"在文献标准下不成立。

**命名陷阱**：JSPLib 用 `ta01js` 命名 job-shop 实例；而 flow-shop 同族实例的 `ta01` BKS 是 1278。**两者不可混用。** 本仓库 [benchmark_bounds.json](../../code/candidate_quantum_gates/benchmark_bounds.json) 的 LB=UB 核对有效，但引用时须写明 `js` 后缀口径。

---

## 3. 与本仓库建模最接近的三篇：决定性对照

这三篇决定了"未见发表"这一说法能否站住。**写作时必须逐条写明差异**。

### 3.1 Zhang, Lo Bianco & Beck（ICAPS 2022）——最接近，也最有威胁

《Solving Job-Shop Scheduling Problems with QUBO-Based Specialized Hardware》，ICAPS 32(1):404–412，DOI [10.1609/icaps.v32i1.19826](https://ojs.aaai.org/index.php/ICAPS/article/view/19826)。**发表页与摘要已由本文档作者核验**；其 rank QUBO 细节来自调研子任务的全文阅读【二手】。

- **重合**：按机器 rank 变量；**明确承认单机序列合并会产生环**；用 QUBO 硬件 + CP 的 LNS 解到 Taillard 20×20，并生成 25×25–50×50。
- **差异（必须写明）**：
  1. **环由 CP 模型处理，没有编码进 QUBO**（未找到其 QUBO 内显式消环约束）；
  2. 其单机 rank QUBO 的目标是启发式打分，**不是 makespan**；
  3. 作者自述是"首个用 QUBO 硬件解非平凡 JSP 的工作，**但仍属混合算法**"（该措辞见摘要，已核）。
- **对我们最有价值的一条**：其 direct QUBO 变体（DQ / TIQ）的失败数据是本课题最硬的负面证据，见 §5。

### 3.2 Bourreau, Fleury & Lacomme（2024/2026）——用编码规避环

《Indirect Job-Shop coding using rank: application to QAOA (IQAOA)》，[arXiv:2402.18280](https://arxiv.org/abs/2402.18280)（摘要页已核），期刊版 *J. Heuristics* 32(2)【二手】。

- **重合**：rank 编码；目标是 makespan。
- **差异**：**无环由 Bierwirth（1995）"vector by repetition" 的编码性质保证**，因此**根本不需要在 QUBO 里写环约束**；目标是经解码器/oracle 评估的 makespan，**不是 QUBO 基态能量**。
- **【未核】**：其 qubit 数与最大算例规模（仅 arXiv v1，期刊版未获取）。

### 3.3 Schmid, Braun, Sollacher & Hartmann（2024/2025）——对当前主线压力最大

《Highly Efficient Encoding for Job-Shop Scheduling Problems…》，[arXiv:2401.16381](https://arxiv.org/abs/2401.16381)，*Quantum Sci. Technol.* **10**, 015051 (2025)。**本文档作者已核验 arXiv 全文的关键公式与措辞**。

- **重合**：顺序编码（operation sequencing list）；无环隐式成立；qubit 数逼近信息论下界；"基态"对应最小 makespan。
- **差异与直接压力**：
  1. **明确放弃 QUBO**——原文："we will forgo a QUBO formulation of the JSP or FJSP and instead use a black box approach"；
  2. qubit 数 $N_{\rm qubits}=\lceil\log_2(N_{\rm op}!/\prod_k|J_k|!)\rceil$（FJSP 再乘 $\prod_i|\mu_i|$），并自称"is optimal in terms of variable efficiency up to these possible symmetry reductions"——**这就是"接近信息论下界"说法的直接出处**；
  3. 原文："directly resolves all problem constraints and thus **avoids the need for any penalty terms** in the cost function"。
- **对本仓库的含义**：该路线用"换编码"绕开了本仓库最痛的两点（任意对角相位合成、见证辅助位深度）。**因此当前"绝热演化 + 成本相位 + 见证罚项"主线的存在价值需要重新论证，而不能默认成立。**
- **实测规模**：数值仿真 **3–13 个工序**（很小）。

---

## 4. 对本仓库建模的新颖性判断（范围受限）

### 4.1 要素级核对

| 要素 | 是否已发表 | 依据 |
|---|---|---|
| 每机器候选顺序（rank 变量） | **已发表** | Wagner 1959 血统【二手】；Zhang/Beck rank QUBO【二手】；Ajagekar 的 pairwise 版本【二手】；Bourreau、Schmid 的编码【已核/二手】 |
| 跨机器环约束**写进 QUBO** | **未发现** | 找到的两条路线都是**回避**而非编码：Zhang/Beck 交给 CP；Bourreau/Schmid 靠编码性质 |
| 跨机器**超工期路径**约束 QUBO 化 | **未发现** | Zhang/Beck 的冲突罚项是**时间窗推出的偏序**，不是"路径长度 > T"；Doucet 的两类罚项是 one-hot 与 precedence，也不含路径长度 |
| 基态能量 = 最优 makespan | **已发表（但在其他表述族）** | 时间索引族：Venturelli【已核】、Kurowski【二手】、Doucet【已核】、Deleplanque【二手】；析取 QUBO 族：Zhang/Beck 的 DQ【二手】 |
| **上述组合成一个统一 QUBO** | **本次检索未发现** | 见 §3 三篇差异 |

### 4.2 可以站住的具体贡献点（建议单独成节论证）

若要在论文中主张新颖性，**不要笼统说"整体未见发表"**，而应锁定最可能站住的一块：

> **在仅含每机器候选序变量与见证投影的紧凑 QUBO 上，把"路径长度超过目标工期"编码为可施加相位的谓词，并证明其基态能量等于池内最优工期。**

理由：时间索引族虽含 makespan，但变量数 $O(N^2M)$ 量级且 max() 无法直接编码（见 §5）；析取 QUBO 族的能量虽含 makespan，但**实验中从未采到可行解**。本仓库的路径阈值谓词 $Q_P=P_P\otimes\mathbf 1[\hat T<L_P]$ 与时间寄存器 $\hat T$ **在本次检索中未见对应物**。

**但必须同时写明**：该命题的成立依赖候选池覆盖、环见证完备、路径见证覆盖每个可行组合的真实最长路三个条件，而 [docs/quantum_candidate_design.md](../../docs/quantum_candidate_design.md) 已注明"获取这种覆盖在最坏情况下仍可能指数困难"。

---

## 5. 负面结果与本仓库记录的同向证据

### 5.1 文献侧【已核/二手，逐条标注】

1. **Zhang/Lo Bianco/Beck, ICAPS 2022**【二手，全文】：其 direct QUBO 变体 DQ 在 5×5 与 10×10 上**零可行解**；TIQ/TIQAE 在 10×10 上**零可行解**；诊断原因是**整数变量的二进制展开与 one-flip 邻域严重不匹配**；结论大意：当前专用硬件用 direct encoding **只能表示很小的 JSP，且即使在小实例上 direct 模型表现也很差**。
2. **Doucet et al., New J. Phys. 28, 054512 (2026)**【已核，摘要+期刊页】：对单个 JSP 实例扫 one-hot 与 precedence 两类罚项，在 D-Wave Advantage 上做循环逆退火 + 绝热主方程模拟。结论：**罚项太弱 → 低能但不可行的流形；罚项太强 → 压低有效问题能量尺度、增加不可逆性、降低热力学效率。** 即"QUBO penalty 是热力学控制旋钮"，**不存在可无限扩张的安全罚项区间**。
3. **Carugno, Ferrari Dacrema & Cremonesi, Sci Rep 12:6539 (2022)**【已核，CC BY 全文 PDF 已获取】：批判性评估退火流水线——问题表示的编译代价、qubit 需求、chain break 缓解、reverse annealing 的实际收益，指出各阶段都存在问题。
4. **Safi et al., arXiv:2607.13325**【已核，摘要】：IBM Quantum + D-Wave + Fujitsu DA 对工业 JSP 变体的三平台对照，**明确不声称优势**，强调 hardware–software co-design 是前提。子任务转述其正文三条硬事实【二手】：完整调度 QUBO 需约 $N^2M$ 变量；**max() 无法直接编码进 QUBO**；因此拆成"先分配机器 + 再经典排序"，且"最优 QUBO 解不保证恢复全局最优"。
5. **Marshall, Venturelli, Hen & Rieffel, Phys. Rev. Applied 11:044083 (2019)**【摘要已核，全文 PDF 已获取】：大问题最终分布趋于经典 Boltzmann 分布，温度高于器件有效温度。**这直接解释为什么惩罚型 QUBO 会被经典热噪声兜底。**
6. **规模天花板**【二手】：光子机 8 qumode（Slysz et al. 2024）；硬件上到 23 qubit（Amaro et al. 2022 的 F-VQE）；QAOA 在约 6 工序 toy 上已吃力（Schmid et al. 说法）。

### 5.2 本仓库已记录的负结果（与文献同向）

[docs/candidate_adaptive_results_20261002.md](../../docs/candidate_adaptive_results_20261002.md) 已如实记录：动态经典 **5.73%** vs 动态+量子提议 **6.43%**（平均相对差距，越大越差）；量子训练主导墙钟；经典基线便宜三个数量级；旧的 `ideal_quantum` 三个 50×20 工期为 3903/4048/4015。

**文献与实测在同一点上汇合：目前没有任何证据支持在 JSP 上宣称量子优势。**

---

## 6. 【实测】"全局机器关联感知"：机制、指标与贡献

这是本报告的核心原创部分。问题来自仓库内的直接追问："现有的算法量子如何保持全局机器关联的感知？这里的感知用什么指标来衡量？对算法整体表现有什么贡献？"

### 6.1 机制拆解（读 [circuits.py](../../code/candidate_quantum_gates/circuits.py) 与 [search_loop.py](../../code/candidate_quantum_gates/search_loop.py)）

**层面 A：见证投影 = 纯组合掩码，静态**

`allowed_labels`（[circuits.py:38–52](../../code/candidate_quantum_gates/circuits.py#L38-L52)）把图见证投影为每机器允许标签集，$P_W=\bigotimes_m P_{m,W}$。它确实横跨多机，但**执行前就算好**。代码对其"量子性"是诚实的：

```python
# circuits.py:288-293  恒真谓词 → 只改全局相位，不发射任何门
circuit.global_phase -= phase_angle * penalty
```

有排除力的见证，作用也只是 `membership XOR → 多控谓词 → 相位`，即**把"违反了多少条已知约束"变成整数并把相位打在它上面**。

**层面 B：联合作动 = 唯一真正的多机量子算符，但方向是局部的**

`build_joint_mixer`（[circuits.py:449–471](../../code/candidate_quantum_gates/circuits.py#L449-L471)）用精确 `CX/X/MC-Rx` 构造 $\exp[-i\theta(|a_S\rangle\langle b_S|+|b_S\rangle\langle a_S|)]$，作用于 2–4 台机器的 $2r$ 个比特。端点选择在 [search_loop.py:134–148](../../code/candidate_quantum_gates/search_loop.py#L134-L148)：

```python
right.append(min((a for a in range(pool.sizes[m]) if a not in spec.allowed[m]),
                 key=lambda a: (abs(a - left[-1]), a)))
```

右端点是**离现任标签秩最近的被禁标签**——按标签编号距离贪心，**不看任何能量、工期或全局信息**。语义上等价于"朝随机方向的局部逃逸"。

**层面 C：真正的全局感知在经典侧**

`search_loop.py` 外层循环的全局环检测、最长路、上下界更新全部是经典图计算（`medium_qjsp`）。量子部分只提供"在当前已知约束下低违反的样本"。

**结论**：**量子侧并不"保持"全局机器关联的感知。** 全局一致性由经典外层维持；量子侧接收的是已知约束的违反计数。代码本身在 [search_loop.py:449–458](../../code/candidate_quantum_gates/search_loop.py#L449-L458) 的 notes 中承认了这一边界。

### 6.2 现存指标，以及为什么它们测不了"感知"

| 指标 | 位置 | 能否回答"感知全局了吗" |
|---|---|---|
| `expected_energy` | [search_loop.py:221–223](../../code/candidate_quantum_gates/search_loop.py#L221-L223) | **不能，且会误导**：$T$ 在下降、见证在增加，跨轮不可比 |
| `top_probabilities` | 同上 | 部分可以（可比），但只测"集中"不测"集中对了" |
| `graph_feasible` / `feasible_rate` | [search_loop.py:388–393](../../code/candidate_quantum_gates/search_loop.py#L388-L393) | 只测"解码无环"，是**违反已知约束的比例**，不是"朝最优方向" |
| `joint_actions` / `effective_witnesses` | 同上 | 测**有多少条关联**，不测**关联起了什么作用** |
| `illegal_fraction` | [circuits.py:502–530](../../code/candidate_quantum_gates/circuits.py#L502-L530) | 只测 one-hot 合法性 |
| `proposal_improvements` | [adaptive_search.py:274–276](../../code/candidate_quantum_gates/adaptive_search.py#L274-L276) | **最接近因果贡献的指标**，但混了"被接受"与"真正改善" |

**关键实测：`expected_energy` 上升不代表感知变差。** demo_3x3 的 `xy` 模式三轮 `expected_energy` 为 **0.638 → 34.482 → 34.482**，第二轮涨约 54 倍，但同一轮 `best_makespan` 从 16 降到 11。这与 [docs/quantum_candidate_design.md](../../docs/quantum_candidate_design.md) 第 6 节"$H_{r+1}\succeq H_r$，加见证必然抬高最低能量"一致。**该指标在轮间不可比，因此不能用作感知度量。**

### 6.3 贡献：所有可测之处为 0 或为负

**证据 1：demo_3x3 同池同初态同预算受控对照**（[results_p3_20261002/search_comparison.json](../../code/candidate_quantum_gates/results_p3_20261002/search_comparison.json)）

| 模式 | best makespan | feasible_rate | 训练评价 | 墙钟 |
|---|---:|---:|---:|---:|
| uniform | 16→11 | 0.625 | 0 | 3.48 s |
| xy | 16→11 | 0.625 | 87 | 1.27 s |
| **xy_joint** | **16→11** | **0.609** | 87 | **2.23 s** |
| classical（同信息同动作） | 16→11 | — | 0 | **0.059 s** |

加联合作动后**目标值一致**，可行率**略降**（0.609 < 0.625），墙钟为 classical 的约 38 倍。

**证据 2：能量降了但答案没变。** demo_3x3 的 xy_joint 训练中，`[1,2]` 参数下 `expected_energy` = **29.406**，`[1,1]` 参数下 34.480——降了；但两者 `best_makespan` 都是 11。**感知指标与目标脱钩的直接证据。**

**证据 3：大规模上关联提议命中率约 0.3%。** [results_adaptive_20261002/summary.json](../../code/candidate_quantum_gates/results_adaptive_20261002/summary.json)（6 实例 × 2 种子）中 `proposal_calls` 从 145 到 727（累计数千），而 `proposal_improvements` 绝大多数为 **0**，少数 1–2，个别到 6。

**证据 4：加关联提议后总体变差。** [docs/candidate_adaptive_results_20261002.md](../../docs/candidate_adaptive_results_20261002.md)：动态经典 **5.73%** → 动态+均匀组合提议 **6.36%** → 动态+量子提议 **6.43%**。

**证据 5：真正的全局感知从未上过测试台。** [code/candidate_quantum_gates/README.md](../../code/candidate_quantum_gates/README.md) 写明 15×20 的 44 条见证只是**资源编译**，无采样。因此全部证据来自 3×3（8 组合）与 5×5——而 5×5 是**退化案例**（label-0 初态本身即池内最优 89/119），小到均匀采样随手命中最优。

### 6.4 【实测】耦合强度诊断——推翻"见证只约束单机"的假设

关联机制唯一的作用空间是"某个可行目标只在多机同时换标签时可达"。若无这样的见证，机制从定义上无用。为此统计**每条有效见证实际约束几台机器**（方法：对每个池枚举全部候选组合，用 `medium_qjsp.separate` 抽取见证，再经 `circuits.derive_witness` 投影后计数非平凡允许集）：

| 池 | 机器数 | 组合数 | 见证数 | 有效见证 | **平均约束机器数** | 最大 | 约束机器数直方图 |
|---|---:|---:|---:|---:|---:|---:|---|
| demo_3x3 | 3 | 8 | 6 | 6 | **1.67** | 2 | {1: 2, 2: 4} |
| random_5x5_seed7 | 5 | 32 | 20 | 20 | **3.55** | 5 | {2: 2, 3: 9, 4: 5, 5: 4} |
| random_5x5_seed11 | 5 | 32 | 16 | 16 | **2.62** | 4 | {1: 1, 2: 7, 3: 5, 4: 3} |

**结果推翻了一个先验假设。** 我此前推断"小池里必须多机同时改变的见证很少，所以联合作动没有作用空间"——**这个推断是错的**：

- demo_3x3 有 6 条有效见证，**4 条约束 2 台机器**，平均 1.67；
- 5×5 池的平均约束机器数达 **2.62–3.55**，最大到 5。

也就是说，**"必须多机协同"的结构确实存在且占多数**。因此联合作动可用却**测不出收益**，原因不在"没有作用空间"，而在**方向选择**：`propose_joint_actions` 按标签秩最近选右端点，语义接近随机方向，所以**解除一个见证的同时很可能撞进另一个见证**。

**这条修正把问题定位从"机制无意义"改为"机制可用但导航无效"**——后者是可修的（见 §7.2）。

**同时暴露一处实现约束**：5×5 池存在约束 5 台机器的见证，而 `propose_joint_actions` 的 `max_support=4` 与 `JointTransition.validate` 的 `2 <= len(support) <= 4`（[circuits.py:158](../../code/candidate_quantum_gates/circuits.py#L158)）**都截断在 4 台**，因此**这类见证无法被任何单条联合作动一次性解除**。这是一个此前未被记录的结构性限制。

### 6.5 一个可检验的新假设：真机"变准"可能来自热化而非算法

读 [code/jsp_qiskit_hardware/hw_vs_ideal.png](../../code/jsp_qiskit_hardware/hw_vs_ideal.png) 时注意到一处与现有解释不一致的现象：

四个层的 TV 距离**非单调**：L1=0.090 → L2=0.177 → **L5=0.414（峰值）** → L10=0.147。而真机分布：L5 时 |01⟩（最优，c=18）约 0.34、|10⟩（不可行）约 0.25，较平；**L10 时 |01⟩ 升至约 0.62、|10⟩ 降至约 0.07**。理想态矢量在 L10 时 |01⟩ 约 0.69，但 |00⟩ 也有约 0.29。

即 **L10 的真机分布比理想分布更"干净"地集中在最优态上**。

**待验证假设**：这不是算法改善，而是 Marshall 等 2019 描述的热化——L10 双比特门更多（20 vs 2），系统更接近经典 Boltzmann 分布，而该 2×2 实例的最低成本态恰在经典热分布下权重最大。

**若成立，它同时解释规模趋势**：2×2 上热化恰好帮了忙，规模一大（15×20、50×20）热化不再指向最优，量子提议随即落后于经典。

**验证方式（纯本地 Aer，不需 QPU）**：在同一实例上跑更高层数（20、40 层），检验最优态概率是否随层数继续单调上升、以及分布是否趋近按 makespan 加权的经典 Boltzmann 分布。

**文献定位**：两批独立检索一致确认——**没有任何论文系统测量过"JSP 惩罚型 QUBO 的最小能隙 / perturbed bit-flip 势垒随实例规模缩放"**。现有势垒研究用随机 Ising、XORSAT、图着色或 planted 输入。这是一个真实且可发表的研究空白，而本仓库已具备全部工具（[qjsp.py](../../code/JSP_量子演化可运行代码/jsp_quantum_evolution/qjsp.py) 状态向量模拟器、[docs/JSP_量子演化能隙与资源推导.md](../../docs/JSP_量子演化能隙与资源推导.md) 能隙推导、[circuits.py](../../code/candidate_quantum_gates/circuits.py) 见证相位构造）。

---

## 7. 建议

### 7.1 先建基线，再谈量子

用 PyJobShop / OR-Tools CP-SAT 在 ta01–ta80 上按 JSPLib 口径（600 s、几何平均 gap）建基线。**没有这条基线，任何量子结果都不可信。**

### 7.2 若要继续"关联感知"这条线，先补三个缺失指标

1. **边际对照**：同池/同初态/同预算下 `uniform` vs `xy` vs `xy_joint` 的**真实 makespan 分布**（而非 `expected_energy`）。现有对照的样本只有 3 个案例且 2 个退化。
2. **Boltzmann 对照**：把采样分布与 $\propto e^{-H/\tau}$ 拟合，检验量子侧是否只是热化采样（§6.5）。
3. **联合动作成功率**：`propose_joint_actions` 产出中，(a) 真正解除目标见证的比例，(b) 落入其他见证的比例。这比 `proposal_improvements` 细，能区分"被接受"与"真正改善"。

### 7.3 两处可直接修的实现问题

1. **能量二值化丢失量级信息**：`witness_energy`（[search_loop.py:106–108](../../code/candidate_quantum_gates/search_loop.py#L106-L108)）对超期 1 单位与 200 单位给予**相同**贡献。改成按超期量加权（[code/candidate_quantum_gates/README.md](../../code/candidate_quantum_gates/README.md) 末尾已列此待办），相位才能承载梯度。
2. **`max_support = 4` 截断**：5×5 池存在约束 5 台机器的见证（§6.4），当前**无法被一次性解除**。应记录被截断的见证比例，或支持更大支撑。

### 7.4 不要做的事

- **不要把"所有经典算法都无法解决"当作目标或已有结论。** 文献证据方向相反（Marshall 的热化、Safi 的三平台对照、Jiang & Shu 的退火器基准）；[docs/quantum_candidate_design.md](../../docs/quantum_candidate_design.md) 第 8 节也已写明这不是已有结论。该目标应作为**待验证假设**记录。
- **不要在论文中虚构 JSP 专属的 no-go 定理**。本次检索**未发现任何形式化结果**；现有负面结论全部是经验性/硬件限制性的。
- **不要承诺"量子解 ta01–ta80"**。当前硬件只能处理被拆解的分配/排序子问题。

---

## 8. 未核实清单（不得作为结论使用）

| # | 项目 | 状态 |
|---|---|---|
| 1 | Roy & Sussmann 1964 原文 | 未找到稳定在线版本，仅经二手文献确认其为析取图建模起源 |
| 2 | Toshiba SBM 在 Taillard JSP 上的 makespan | 未找到，**可能不存在** |
| 3 | Casati & Giudici 2025 的算例规模 | 未核到内容层 |
| 4 | Safi et al. 的算例规模 | 未核到内容层 |
| 5 | IQAOA 期刊版的 qubit 数与最大算例 | 未核到（arXiv 仅 v1，期刊版未获取） |
| 6 | Permin et al. 2022 内容 | 标题/作者/URN 已核（d-nb MARC21），**内容未读**；标题与"复杂度削减"高度相关，建议单独取全文 |
| 7 | "Disjunctive Programming meets QUBO" 方法与结果 | 仅核到章节出处与摘要级信息 |
| 8 | Adams–Balas–Zawack 链接 | 正确为 <https://dl.acm.org/doi/10.5555/2841176.2841184>（调研过程中曾出现笔误） |
| 9 | D-Wave hybrid 在 Taillard 上的优势 | 官方 vignette 明确是 **flow shop**，**不可**推广为"JSP 也能赢 CP-SAT" |
| 10 | `ta70`–`ta80` 的 BKS 数值 | 抓取被截断，**未核实**；请以 Optimizizer 与 JSPLib `bks.json` 为准 |
| 11 | Ajagekar et al. 的 `∑ x ≤ \|S_m^k\|−1` 约束 | **经读原文确认是 no-good 整数割，不是消环约束。请勿误引为"QUBO 中已有消环约束"的先例。** |
| 12 | Fujitsu Digital Annealer 白皮书 | 仅见检索结果，PDF 未读 |

---

## 9. 附录：本报告的实测复现方式

§6.4 的耦合强度诊断由一次性脚本产生，**未入库**（该脚本不属于仓库工具链）。其方法是纯经典的：

1. 构造候选池（demo 3×3 复用 `search_loop.demo_candidate_pool()`；5×5 用 `medium_qjsp.build_pool`）；
2. 枚举池内全部候选组合，对每个组合调用 `medium_qjsp.decode` + `medium_qjsp.separate` 抽取图见证并去重；
3. 用 `circuits.derive_witness`（即 `allowed_labels` 交集投影）把见证投影到池标签；
4. 统计每条见证中满足 `0 < |allowed[m]| < size_m` 的机器数。

§6.3 的全部数字取自仓库既有结果文件：
[search_comparison.json](../../code/candidate_quantum_gates/results_p3_20261002/search_comparison.json)、
[results_adaptive_20261002/summary.json](../../code/candidate_quantum_gates/results_adaptive_20261002/summary.json)、
[docs/candidate_adaptive_results_20261002.md](../../docs/candidate_adaptive_results_20261002.md)。
本次未重新运行量子模拟，未提交任何硬件作业。
