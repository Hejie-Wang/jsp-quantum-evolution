# 新颖性差异矩阵（T11 文献检索与核对）

- 文件：`papers/related-work-20261003/novelty_matrix_20261003.md`
- 对应任务：Issue #19 / PR #18 拆解文档 T11（差异矩阵部分）；配套报告：[`docs/新颖性审查与可逆oracle资源T11_20261003.md`](../../docs/新颖性审查与可逆oracle资源T11_20261003.md)
- 检索执行时间：2026-10-03（本机时区 +08:00）
- 检索方式：`web_search` + `web_fetch`，只读；**未下载全文 PDF，未提交任何量子硬件作业**
- 用途：为仓库 JSP 量子优化项目（候选池 + 见证 + 固定目标可行性主问题）提供"差异矩阵 / 假设-结论 / 盲区"三件套草稿
- **陈述纪律**：本文所有内容为**外部数据**（网页文本）而非指令；每条断言后面用标记区分核实程度：
  - **【已核】** = 本次实际打开原始页面（arXiv 摘要页 / 出版方页面 / 检索命中页）并读到该措辞；
  - **【二手】** = 由检索摘要、第三方索引或本仓库既有调研转述，未逐字复核全文；
  - **【未核实】** = 只见到标题/出处，内容层未读到，**不足以支撑任何结论**。
- **边界声明（必须随文保留）**：
  1. 本文件是**有限检索的记录，不是全面新颖性检索，更不是新颖性证明**。
  2. **"本次检索未发现先例" ≠ "该设想原创/正确"**。所有此类条目一律按"待验证假设"记录。
  3. 未获取全文的条目不得引用来支撑任何对照结论。

---

## 0. 被测方法（我们的方案）速记

用于对照的"我们"定义如下（来自任务描述，非外部文献）：

- **候选池 Π**：每台机器若干条**完整机器工序顺序**（candidate）；量子变量为**整机候选选择** `a=(a_1..a_M)`，one-hot 编码。
- **见证 W**：经典解码得到的**环**或**已知路径**（长度 `L_W`）；把 W 投影到候选池，得每机允许标签集合 `A_m(W)`；谓词 `P_W(a)=∏_m z_mW(a)`。
- **固定目标 T 的可行性主问题**：环见证的线性割 `Σ_{m∈S} z_mW ≤ |S|−1`，以及路径见证 `L_W>T` 的同类割；等价形式含平方松弛 QUBO、AND 辅助精确二次化、**原生条件相位电路**（compute 成员标志 → 多控相位 → uncompute 工作位归零）。
- **纪律**：作用域安全（池内界 ≠ 全局界）；见证重投影（换池必须重算标签掩码）；证书只从合法松弛/完整模型取得；采样失败不得抬高下界。
- **实验**：T01–T05（候选覆盖 / 代理目标 / 等价编码与资源），并对量子机制做同池同见证同信息的受控消融。

---

## 1. 新颖性差异矩阵

为可读性分两张表。表 A 覆盖"问题设定与编码 / 候选池 / 量子相位"，表 B 覆盖"下界与认证 / 候选生成 / 与我们可验证的差异"。

列名简写：**候选池** = 是否使用候选池/整机候选；**量子相位** = 是否以量子相位（含条件相位、oracle、Ising 对角相位）承载见证/割约束。

### 表 A：问题设定、编码、候选池、量子相位

| # | 工作（作者/年份/来源，均可点击） | 问题设定与编码 | 候选池 | 量子相位 |
|---|---|---|---|---|
| A1 | Zhang, Lo Bianco & Beck, ICAPS 2022，[DOI 10.1609/icaps.v32i1.19826](https://doi.org/10.1609/icaps.v32i1.19826) / [AAAI 页](https://ojs.aaai.org/index.php/ICAPS/article/view/19826) | JSP；QUBO + CP-LNS 混合；两个 direct QUBO 模型 + 按机器 rank 变量【已核：摘要明示 "two direct QUBO models of JSP"、"hybridizes a QUBO model with constraint programming"、"first approach ... can solve non-trivial JSPs using QUBO hardware, albeit as part of a hybrid algorithm"】【二手：rank 变量细节、单机序列合并产生环、环交由 CP 处理】 | 未使用（LNS 邻域为 CP 侧大邻域，非预生成整机候选）【二手】 | 未使用【二手；未见摘要提及】 |
| A2 | Schmid, Braun, Sollacher & Hartmann, [arXiv:2401.16381](https://arxiv.org/abs/2401.16381)，期刊版 [Quantum Sci. Technol. 10, 015051 (2025)](https://doi.org/10.1088/2058-9565/ad9cba) | JSP/FJSP；**紧凑顺序编码**，bit-string 数比时间索引编码至少降 `N/log2(N)` 倍；VQA 求解【已核：摘要逐句】 | 未使用（编码列全部工序序列，非"每机候选子集"）【已核：摘要口径】 | 未使用；作者走 **black-box 目标**而非 QUBO，避免罚项【二手：转述自本仓库既有调研 [literature_survey.md](literature_survey.md) 对原文措辞的引用】 |
| A3 | Coupvent des Graviers, Kobrosly, Guettier & Cazenave, [arXiv:2504.16106](https://arxiv.org/abs/2504.16106) | JSP/FJSP **上下界更新**；用 OR-Tools 求解器组合（portfolio）计算开放基准的新界；闭合 ta33，改进 ta26/ta45/05a/06a 上界，给出 ta45、car5 最优解【已核：摘要逐句】 | 不适用（纯经典阈值/界计算） | 不适用 |
| A4 | Zhao, Fan & Han, [arXiv:2112.07109](https://arxiv.org/abs/2112.07109)（WCNC 2022） | MILP；**Benders 分解**，master problem 转 QUBO 交量子退火，子问题给出经典 Benders 割【已核：摘要】 | 未使用（master 变量即原 MILP 变量） | 未使用（割仍在经典侧生成并回填 master，非量子相位）【已核：摘要口径】 |
| A5 | Doucet, Mzaouali, Robertson, Gardas, Deffner & Domino, [arXiv:2601.04402](https://arxiv.org/abs/2601.04402)，期刊版 [New J. Phys. 28, 054512 (2026)](https://doi.org/10.1088/1367-2630/ae6e98) | **单个 JSP 实例**；QUBO 双参数罚项族 `(p_sum, p_pair)`（one-hot/求和约束、precedence 约束）；D-Wave Advantage + 绝热主方程模拟 + 循环逆退火【已核：摘要】 | 未使用（时间索引/one-hot 罚项结构）【已核：摘要口径】 | 未使用（罚项即能量项，非条件相位成员标志）【已核：摘要口径】 |
| A6 | Bourreau, Fleury & Lacomme, [arXiv:2402.18280](https://arxiv.org/abs/2402.18280)，期刊版 *J. Heuristics* 32(2)【二手，见 [ACM 页](https://dl.acm.org/doi/abs/10.1007/s10732-026-09593-6)】 | JSP；**Bierwirth "vector by repetition" 间接 rank 编码**，无环由编码性质保证；应用于 QAOA（IQAOA）【已核：摘要】 | 未使用 | 未使用（无环不进入约束，故无见证相位）【已核：摘要口径】 |
| A7 | Lopez-Ruiz, Tucker, Arnold, Epifanovsky, Kaushik & Roetteler, [arXiv:2510.26859](https://arxiv.org/abs/2510.26859) | Just-in-Time JSP；**Iterative-QAOA**（固定参数 schedule + 迭代 warm-start），IonQ Forte 真机，最大 97 qubit 张量网络仿真【已核：摘要】 | 未使用（时间索引/罚项型目标）【已核：摘要口径；罚项形式见 ar5iv 正文片段 `c(x)+λΣ(g−1)^2`，本次仅见片段，标【二手】】 | 未使用 |
| A8 | Casati, Giudici, Torta 等，*Iterative Quantum Annealing for Job Shop Scheduling with a Directed Graph Representation*，[IEEE Xplore 11250300](https://ieeexplore.ieee.org/abstract/document/11250300/) | JSP；**有向图表示 + 迭代量子退火**【未核实：IEEE 正文页需登录，只取到标题与会议类型 "IEEE Conference Publication"；作者名来自 Semantic Scholar 命中链接文本，未复核】 | **未核实** | **未核实** |
| A9 | Ciacco, Di Puglia Pugliese & Guerriero, [arXiv:2604.20321](https://arxiv.org/abs/2604.20321) | TSP（非 JSP）；**割平面框架**：子回路消除约束动态生成 + 预处理裁减候选弧；量子退火（QPU 直算与 hybrid）与经典并列比较【已核：摘要】 | 有"候选弧裁剪"，是**弧级**候选裁减，不是整机工序顺序候选【已核：摘要措辞 "preprocessing phase to reduce the number of candidate arcs"】 | 未使用（割动态生成在经典侧）【已核：摘要口径】 |
| A10 | Mendes de Araujo & de Abreu Faria, [arXiv:2606.29647](https://arxiv.org/abs/2606.29647) | 通用 QUBO（MDSSP 基准，N≤1000）；**Hybrid Quantum Neighborhood Selection**：每阶段只激活 F≪N 个变量构成 frontier，其余冻结并入 reduced QUBO 系数，多阶段轮换【已核：摘要】 | 有"变量前沿选择"，是**变量级邻域**而非问题结构化的整机候选【已核：摘要】 | 未使用（reduced Hamiltonian + QAOA，非谓词相位）【已核：摘要口径】 |
| A11 | Perron, Bérubé-Lauzière & Drouin-Touchette, [arXiv:2603.28933](https://arxiv.org/abs/2603.28933) | MWIS；**Lp-Quts**：中性原子采样器嵌入经典**割平面**算法；RLP 给出界，样本指导 odd-cycle 割选择；t-perfect 图上继承多项式收敛保证【已核：摘要】 | 无候选池，用 RLP 诱导的 reduced graph【已核：摘要】 | 未使用（采样器+割选择，非相位） |
| A12 | Moosavi & Farooq, [arXiv:2607.07550](https://arxiv.org/abs/2607.07550)（IEEE Quantum Week 2026） | 带时间窗取送货 VRP；**RL-Guided Quantum-ALNS**：浅层量子采样器嵌入 ALNS 的 repair 阶段，DQN 决定该 repair 用经典还是量子【已核：摘要】 | 无候选池；"reduced repair subproblem" 由 ALNS 破坏算子产生【已核：摘要】 | 未使用 |
| A13 | Plyashechnik, Zhukov, Lebedev & Pogosov, [arXiv:2606.03380](https://arxiv.org/abs/2606.03380) | 抽象搜索问题；**对角 Ising 哈密顿量的连续相位 oracle** 作为 Grover 型迭代 `W_T=D_ξ exp(−iTH)` 的 oracle；能量选择性共振的谱理论；高密度尾共振 `Θ(√(2^n/M))` 次 oracle 调用达常数成功概率【已核：摘要】 | 不适用 | **使用对角相位 oracle**（但为能量选择，不是"见证谓词成员标志"）【已核：摘要】 |
| A14 | Li, Wang 等, *Quantum Assisted Combinatorial Benders' Algorithm ...*，[IEEE Xplore 10841900](https://ieeexplore.ieee.org/document/10841900) / [NSF PAR 10600327](https://par.nsf.gov/biblio/10600327) | 氢-电-移动储能协同规划；量子辅助 **Combinatorial Benders** 算法【未核实：只取到标题与摘要级条目，正文未读】 | **未核实** | **未核实** |
| A15 | Brassard, Høyer, Mosca & Tapp, [arXiv:quant-ph/0005055](https://arxiv.org/abs/quant-ph/0005055)（AMS Contemp. Math. 305:53–74, 2002） | 振幅放大与振幅估计；对已知经典启发式的一大类搜索问题也给二次加速；无需预知 `a`【已核：摘要】 | 不适用（通用算法） | 不适用（`χ` 为布尔判定，非物理条件相位电路） |
| A16 | Montanaro, [arXiv:1509.02374](https://arxiv.org/abs/1509.02374) | 回溯树上的量子行走加速：`O(√T · n^{3/2} log n)` 次测试；可用于 DPLL/SAT【已核：摘要】 | 不适用 | 不适用 |
| A17 | Grover Adaptive Search-Based Hybrid Benders Decomposition for MILP，*IEEE Trans. Quantum Engineering*，[tqe.ieee.org 文章页](https://tqe.ieee.org/2026/04/06/grover-adaptive-search-based-hybrid-benders-decomposition-for-mixed-integer-linear-programs/) / [IEEE Xplore 11475201](https://ieeexplore.ieee.org/document/11475201) | MILP；GAS + Benders 混合；**完整 oracle 校验候选解是否同时满足目标条件与约束**【二手：仅从检索片段读到此句，正文与作者名单未取得】 | **未核实** | 有"oracle 校验"，但是否为**相位型**见证约束 **未核实** |

### 表 B：下界/认证、候选生成、与我们可验证的差异

| # | 下界/认证处理 | 候选生成方式 | 与我们可验证的差异（可被实验检验的表述） |
|---|---|---|---|
| A1 | 无量子侧下界；用 CP 提供可行性与界，LNS 迭代改进【二手】 | CP 大邻域 + 硬件采样，无预生成候选池【二手】 | ① 我们的见证（环/路径）**编译进量子主问题本身**（线性割/QUBO/条件相位三态等价），其环约束在 A1 中由 CP 承担、**未进入 QUBO**；② 我们的界始终标注"池内"作用域，A1 的 LNS 直接报 makespan 改进，二者认证口径不同——须在实验中分别给出池内最优 MILP 证据与全局 BKS 对照 |
| A2 | 无；VQA 期望值优化，无证书【已核：摘要口径】 | 编码覆盖全部序列，无采样候选池 | ① 变量语义不同：A2 是"全序列压缩编码"，我们是"每机 K 选 1 候选"；② 可验证差异 = 在**同等 bit/量子比特预算**下，我们的池内覆盖缺口（T01 已量化）与 A2 的信息论下界口径的对比；③ A2 明确绕开罚项，我们则要证明**条件相位路线**在"有见证约束但无罚权调参"这一维度上的资源账 |
| A3 | **本文献的核心就是上下界**：OR-Tools portfolio 更新开放实例 LB/UB，ta33 闭合，ta26/ta45/05a/06a 上界改进【已核：摘要】 | 不适用 | 我们的"目标 T 阈值不可行性"必须与该文的界保持一致的实例/命名口径（如 taXXjs 与 taXX flow-shop 不可混用，见既有调研 [literature_survey.md §2.1](literature_survey.md)）；差异点在于我们报告的是**池内 T 阈值可行性**，不含全局下界声明 |
| A4 | Benders 收敛性由经典分解理论保证；量子侧只当 master 求解器【已核：摘要】 | 无候选池 | ① 割的方向相反：A4 的割在**经典子问题**产生后回填 master；我们的割是**见证（图评价结果）**投影到候选池后编译成量子约束/相位；② 我们额外要求"换池即重投影标签掩码"，A4 无对应概念——可实验检验：池改变后不重投影会出现的恒真/恒假见证计数 |
| A5 | 无；关注罚项对可行性流形与热力学代价的影响【已核：摘要】 | 无候选池 | ① A5 用**两类罚项**（one-hot、precedence）刻画能量景观；我们的路径/环见证是**目标阈值 T 相关**的谓词，A5 无 `L_W>T` 型约束；② A5 的结论（罚项过弱→低能不可行流形；过强→压低有效能量尺度）可作为我们"采样失败不得抬高下界"纪律的**外部支持证据**，但两者**机制不同**（罚项 vs 条件相位），不可互相替代 |
| A6 | 无下界；靠编码性质保证无环【已核：摘要】 | 无候选池 | ① A6 用编码**规避**环约束，我们用见证**编码**环约束——两者可以做成**同实例对照实验**（同池、同 T，IQAOA 式编码 vs 见证割/相位）；② A6 的解质量来自解码器/oracle 评估【二手】，我们的认证要求"证书只从合法松弛/完整模型取得"，评价口径不同 |
| A7 | 无；报告与 VarQITE、Linear-Ramp QAOA 的相对表现【已核：摘要】 | warm-start 由迭代过程产生，无候选池 | ① A7 属"时间/罚项型编码 + 变分参数"，我们是"候选选择 + 见证相位"；② A7 提供了**同规模真机对照**的外部参照点（97 qubit 上限），我们的资源账应与之对齐可比的比特/深度指标 |
| A8 | **未核实** | **未核实** | 无法给出差异；需人工获取 IEEE 正文后补 |
| A9 | 割平面迭代的收敛性来自经典 OR 理论；量子侧为子问题求解器【已核：摘要】 | **弧级候选裁剪**（预处理）【已核：摘要】 | ① 结构同构点：A9 = "候选裁减 + 动态割生成 + 量子求解"，我们 = "整机候选池 + 见证割 + 量子主问题"——这是**最需要正面处理的结构邻接**；② 差异在候选语义（弧 vs 整机工序顺序）与割来源（子回路消除 vs 图评价见证）；③ A9 未涉及条件相位资源账 |
| A10 | 无；强调 bounded circuit width 与 QPU 时间稳定【已核：摘要】 | stochastic frontier 变量选择 + 多阶段轮换【已核：摘要】 | ① A10 的"前沿选择"是**变量级**的，我们是**结构级（整机候选）**的；② A10 的 warm-start/CVaR 过滤/随机轮换消融设计，可作为我们"受控消融"方法学的对照模板；③ A10 未做证书/界的作用域声明 |
| A11 | **有部分收敛保证**：RLP 给出界；t-perfect 图上继承多项式时间收敛【已核：摘要】 | RLP 诱导 reduced graph【已核：摘要】 | ① 这是"量子采样器 + 经典割平面 + 显式界"最接近我们**作用域安全**纪律的已发表工作，必须在 related work 中正面引用；② 差异：A11 的界来自松弛 LP 且对象是 MWIS，我们的界严格限定在候选池作用域且对象是 JSP 阈值 T 可行性；③ A11 未使用相位/条件相位电路 |
| A12 | 无（报告 gap 相对 ALNS 的改进）【已核：摘要】 | ALNS 破坏算子产生 reduced repair subproblem【已核：摘要】 | ① 同构点：量子采样器只做**局部修复/局部候选组合**——与我们"量子侧参与有价值的局部候选组合搜索"定位一致；② 差异：A12 用 DQN 决定**是否**用量子，我们用见证谓词决定**约束什么**；③ A12 的结论（量子仅在约 16% 状态可用、平均不占优，但匹配预算下 36 组中 29 组改善）是我们做"同池同见证同信息受控消融"时必须对照的期望模式 |
| A13 | 无界；给出 oracle 调用复杂度 `Θ(√(2^n/M))` 与成功概率 `Θ(1)`【已核：摘要】 | 不适用 | ① A13 与我们的"原生条件相位"最接近之处在于**用对角 Ising 相位承载 oracle**；差异是我们承载的是**见证谓词（成员标志 + 多控相位 + uncompute）**而非能量选择；② 这是最可能被审稿人拿来质疑"条件相位并不新"的一条，必须在方法学中明确区分：**相位作为约束谓词 vs 相位作为能量选择**，并给出各自的编译资源账 |
| A14 | **未核实** | **未核实** | 无法给出差异；需人工获取正文后补 |
| A15 | 无（算法复杂度结果） | 不适用 | 我们若主张"见证约束下采样加速"，须与 A15 的 `1/√a` 语义对齐，并说明把 `χ` 编译为**条件相位电路**的额外代价（标志位、多控门、uncompute）——A15 不含此代价 |
| A16 | 无（算法复杂度结果） | 不适用 | 回溯树加速与我们"环/路径见证枚举"有概念相邻性；差异：A16 加速的是**搜索树遍历**，我们加速的是**固定见证集下的候选组合采样**；不可直接引用为我们的复杂度依据 |
| A17 | 有 oracle 级的"目标条件 + 约束"校验【二手】 | **未核实** | 若其 oracle 确为 Grover 型约束校验，则与"量子侧做可行性判定"命题相邻；**在未取得正文前不得用于对照** |

**表格使用提醒**：表 A/B 中任意标【未核实】的单元不得进入论文的对照论证；标【二手】的单元需在投稿前取得原文逐条核对。

---

## 2. 研究假设 vs 已证结论

本节严格三分：**(H) 我们的假设**、**(F) 文献已证/已报告**、**(G) 本次检索未发现先例**。

### 2.1 (F) 文献已证 / 已报告（可直接引用，附核实级别）

| 编号 | 结论 | 出处 | 级别 |
|---|---|---|---|
| F1 | QUBO 硬件 + CP 的 LNS 可解非平凡 JSP（20×20 级），但**仍是混合算法**；作者自称是首个用 QUBO 硬件解非平凡 JSP 的工作 | [ICAPS 2022](https://ojs.aaai.org/index.php/ICAPS/article/view/19826) | 【已核：摘要原文措辞】 |
| F2 | 紧凑顺序编码可把表示全部调度所需的 bit-string 数降低至少 `N/log2(N)` 倍（相对时间索引编码），并在 VQA 上显著改善表现 | [arXiv:2401.16381](https://arxiv.org/abs/2401.16381) | 【已核：摘要】 |
| F3 | 用 OR-Tools 求解器组合可更新开放 JSP 基准的上下界（闭合 ta33；改进 ta26/ta45/05a/06a 上界） | [arXiv:2504.16106](https://arxiv.org/abs/2504.16106) | 【已核：摘要】 |
| F4 | Benders master 转 QUBO 交量子退火、子问题给经典割的混合分解可行，并能保证 MILP 解质量 | [arXiv:2112.07109](https://arxiv.org/abs/2112.07109) | 【已核：摘要】 |
| F5 | 对单个 JSP 实例，QUBO 罚项 `(p_sum,p_pair)` 存在**尖锐的可行性与求解成功转变**；罚项过弱产生低能不可行流形，过强压低有效问题能量尺度并增加不可逆性 | [arXiv:2601.04402](https://arxiv.org/abs/2601.04402) / [NJP 28 054512](https://doi.org/10.1088/1367-2630/ae6e98) | 【已核：摘要】 |
| F6 | Bierwirth "vector by repetition" 编码可保证解码图无环，从而**无需在 QUBO/QAOA 中写环约束** | [arXiv:2402.18280](https://arxiv.org/abs/2402.18280) | 【已核：摘要】 |
| F7 | 非变分 Iterative-QAOA 在真机（IonQ Forte）上对 JIT-JSSP 稳健收敛，规模至 97 qubit（仿真） | [arXiv:2510.26859](https://arxiv.org/abs/2510.26859) | 【已核：摘要】 |
| F8 | TSP 上"动态割生成 + 候选弧裁剪 + 量子退火（QPU/hybrid/经典并列）"可显著缩小模型规模并改善计算表现 | [arXiv:2604.20321](https://arxiv.org/abs/2604.20321) | 【已核：摘要】 |
| F9 | 变量前沿分解（只激活 F≪N 变量、其余冻结进 reduced QUBO 系数）可把电路负担从 `O(N^2)` 降到 `O(F^2)` 每层每阶段，并使密 QUBO 在 NISQ 上可执行 | [arXiv:2606.29647](https://arxiv.org/abs/2606.29647) | 【已核：摘要】 |
| F10 | 中性原子采样器嵌入经典割平面（RLP 给界 + 样本指导 odd-cycle 割选择）可在 t-perfect 图上**继承多项式收敛保证**；同采样预算下优于直接模拟量子协议与贪心基线，最优性差距 5–10% | [arXiv:2603.28933](https://arxiv.org/abs/2603.28933) | 【已核：摘要】 |
| F11 | 量子采样器只嵌入 ALNS **repair 阶段**时：仅约 16% reduced repair 状态可用、平均不占优，但在匹配修复预算下 36 组中 29 组改善最终 gap；结论是"近term 量子采样最有用的角色是**选择性局部修复机制**" | [arXiv:2607.07550](https://arxiv.org/abs/2607.07550) | 【已核：摘要】 |
| F12 | 对角 Ising 哈密顿量演化可直接作连续相位 oracle；能量选择性高密度尾共振需 `Θ(√(2^n/M))` 次 oracle 调用、成功概率 `Θ(1)`，相对"独立均匀采样 + 经典能量评估"有二次查询改进 | [arXiv:2606.03380](https://arxiv.org/abs/2606.03380) | 【已核：摘要】 |
| F13 | 振幅放大可把"跑 A、测量、用 `χ` 校验"的 `1/a` 期望重复压到 `∝1/√a`，且对一大类**已有经典启发式**的搜索问题同样给二次加速 | [quant-ph/0005055](https://arxiv.org/abs/quant-ph/0005055) | 【已核：摘要】 |
| F14 | 若经典回溯树含 T 个顶点，则存在用量 `O(√T·n^{3/2}log n)` 次测试的有界误差量子算法；可用于加速 DPLL | [arXiv:1509.02374](https://arxiv.org/abs/1509.02374) | 【已核：摘要】 |

### 2.2 (H) 我们的假设（**均为待验证**，不得写成结论）

| 编号 | 假设 | 目前支撑 | 需要什么才算验证 |
|---|---|---|---|
| H1 | 在候选池 Π 上，把环/路径见证编译为**线性割、平方松弛 QUBO、AND 辅助精确二次化、原生条件相位**四种形式，对**冻结见证集**等价 | 本仓库 T05 内部实验（等价编码与资源） | 形式化等价性证明 + 跨实现的资源对照（比特/深度/T 门计数）；**注意**：T05 只证"对冻结谓词集合等价"，**不等价于 JSP 可行性** |
| H2 | **原生条件相位**（compute 成员标志 → 多控相位 → uncompute）比罚项路线有更低的见证编译资源 | 本仓库门级离线实现 | 与 A5 的罚项路线、A2 的无罚项编码路线做**同实例、同池、同见证**的资源账对照 |
| H3 | 量子侧在"局部候选组合搜索"上有价值，但仅在**特定结构**上可被有效采样 | 本仓库 T01–T05 + 受控消融 | 预注册的消融协议（同池、同见证、同信息），并报告负结果；须对照 F11 的模式（局部修复偶有改善、平均不占优） |
| H4 | **作用域安全**（池内界 ≠ 全局界）+ **见证重投影**（换池必须重算标签掩码）是必要纪律，缺失会导致恒真/恒假见证与假的界声明 | 本仓库 P0/T03 计数（`infeasible_by_witness`、`graph_infeasible_but_predicates_satisfied`） | 需要给出**反例实验**：不重投影时界声明会怎样失效；并与 F10（RLP 给界的显式作用域）对比论述 |
| H5 | 采样失败不得抬高下界；证书只能来自合法松弛/完整模型 | 本仓库纪律条款 | 需要"失败即失败"的可复现日志与失败模式分类；F5 可作为外部支持但**不是同一机制** |

### 2.3 (G) 本次检索未发现先例的条目（**"未检索到 ≠ 新颖性证明"**）

> 以下每条都只是"在本次检索的范围、关键词与时间内未命中对应工作"。它们全部是**待验证假设**，投稿前须做正式的新颖性检索（含 Scopus/WoS/Google Scholar 引文追踪）。

| 编号 | 未发现先例的具体命题 | 本次检索覆盖情况（为什么不敢说"不存在"） |
|---|---|---|
| G1 | 把**跨机器环见证**与**超目标工期路径见证**编译成量子侧约束（线性割 / QUBO / **条件相位**），且**基态/相位语义恰对应"存在 ≤T 的池内调度"** | 命中的三篇最接近工作分别"交给 CP"（A1）、"靠编码规避"（A6）、"用 one-hot+precedence 罚项"（A5），均无 `L_W>T` 型路径阈值谓词；但检索未穷尽 2024–2026 的预印本，且 A8/A14/A17 正文未取得 |
| G2 | **整机候选池（每机 K 条完整工序顺序）+ one-hot 选择**作为量子变量，配合见证投影标签掩码 | A9 是**弧级**候选裁剪、A10 是**变量级**前沿选择，语义均不同；但"候选/邻域裁剪"这一上位概念已广泛存在，需在写作中严格区分层级 |
| G3 | **见证重投影纪律**（换池即重算 `A_m(W)` 掩码）与**池哈希绑定反馈**作为可审计机制 | 未见对应物；A4/A9/A11 的割/界机制中无"换池重投影"这一步（因它们不换候选池语义）；但这类"工程纪律"类贡献通常不被视为新颖性来源，宜作为方法学/可复现性贡献陈述 |
| G4 | 在**固定目标 T** 下把"阈值不可行性"作为量子主问题求解目标（而非最小化 makespan） | A1/A2/A6/A7 都是最小化 makespan；A3 做的是**经典**上下界更新。阈值/可行性形式的量子化在本次检索中未命中，但"decision version of optimization"本身是经典常识，新颖性只可能在**编译与资源**层面 |
| G5 | 对量子机制做**同池、同见证、同信息**的受控消融（含负结果报告） | A10 有 warm-start/CVaR/轮换的消融，A12 有 DQN 开关对照，但"候选池 + 见证"语境下的同信息消融未命中 |

**写作建议（据上述 (F)/(G) 得出的策略）**：不要把主张写成"整体未见发表"；应锁定 G1+G4 的**交集**——"在仅含整机候选变量与见证投影的紧凑模型上，把 `L_W>T` 编译为可施加相位的谓词，并证明池内语义与资源账"，同时把 A1/A5/A6/A9/A11 逐条写成差异（表 A/B 已给出可检验表述）。

---

## 3. 检索盲区

### 3.1 检索时间

- **2026-10-03（+08:00）**，单次会话内完成；无跨日复核。
- 检索工具：`web_search`（多组关键词）+ `web_fetch`（arXiv 摘要页 / AAAI OJS / QLab / IEEE Xplore / Semantic Scholar / par.nsf.gov）。

### 3.2 实际使用的关键词（原样记录，含失败项）

**第一轮（命中的关键查询）**
1. `Zhang Lo Bianco Beck 2022 ICAPS Solving Job-Shop Scheduling Problems with QUBO-Based Specialized Hardware`
2. `arXiv 2401.16381 Highly Efficient Encoding for Job-Shop Scheduling Problems quantum`
3. `arXiv 2504.16106 Updating Lower and Upper Bounds for the JSP Test Instances`
4. `arXiv 2112.07109 Hybrid Quantum Benders Decomposition MILP`
5. `Doucet arXiv 2601.04402 QUBO penalty weights thermalization`
6. `quantum constraint generation cutting plane phase oracle optimization`
7. `arXiv 1509.02374 Montanaro quantum backtracking`
8. `candidate pool machine sequence quantum sampling job shop local search`

**第二轮及以后（补充邻接工作）**
9. `graph neural network candidate generation job shop scheduling neighborhood quantum`
10. `quantum computing cutting plane constraint generation optimization QAOA`
11. `job shop scheduling quantum annealing candidate solution pool local search`
12. `scheduling quantum amplitude amplification Brassard quant-ph/0005055`
13. `quantum computing lower bound proof infeasibility certificate scheduling optimization 2025`
14. `quantum local search warm start neighborhood selection QAOA combinatorial optimization large neighborhood search`
15. `quantum annealing penalty weight constraint generation cutting plane QUBO 2025`
16. `"Grover Adaptive Search" Benders Decomposition mixed-integer linear programs arXiv`
17. `"A Non-Variational Quantum Approach to the Job Shop Scheduling Problem"`
18. `quantum job shop scheduling candidate pool machine permutation selection QAOA qubit reduction 2026`
19. `learning to generate candidate moves job shop scheduling graph neural network local search`
20. `quantum optimization conditional phase oracle constraint enforcement Grover feasibility certificate`
21. `hybrid quantum constraint generation cuts Benders scheduling lower bound 2025 2026`
22. `iterative quantum annealing job shop scheduling directed graph representation 2025 arxiv`
23. `quantum algorithms for scheduling problems: a survey EPJ Quantum Technology 2026`
24. `Casati Giudici Torta "Iterative Quantum Annealing" job shop 2025 IEEE conference`
25. `quantum computing scheduling one-hot machine sequence encoding candidate selection qubit`
26. `amplitude amplification uses heuristic solutions quantum search quadratically speedup good classical heuristics constrained optimization`
27. `quantum phase oracle constraint "compute" "uncompute" ancilla optimization problem encoding 2025`
28. `scoped bound subproblem quantum optimization claim certificate lower bound sampling failure safety`
29. `"witness" constraint generation quantum optimization cutting plane local search scheduling`

**失败/无效查询（记录以便复现）**
- `quantum computing job shop scheduling candidate pool ... 2026`（多组变体）— 未命中"整机候选池量子选择"的直接对应工作；返回多为不相关命中（如 LLM+CIM 的 [arXiv:2605.23934](https://arxiv.org/abs/2605.23934)，已核其摘要为 agentic 建模主题，**与本方向无关**）。
- Semantic Scholar Graph API 直连查询 — HTTP 429/202，未能取得结构化元数据。
- 中文关键词组合（含"证书 量子 调度"）— 未命中可用中文文献，返回为无关 PDF 片段。

### 3.3 未能获取全文 / 内容层未读的条目（**不得用于对照**）

| 条目 | 状态 | 建议的人工复核动作 |
|---|---|---|
| A8 Casati/Giudici/Torta 等，*Iterative Quantum Annealing for JSP with a Directed Graph Representation* | 【未核实】IEEE Xplore 正文页需账号；只取到标题、文献类型"IEEE Conference Publication"与 Semantic Scholar 命中链接。作者名、年份、会议名**均未逐字复核** | 通过机构订阅取 PDF；确认作者名单/年份/会议，并判断其"有向图表示"是否等价于我们的候选池语义 |
| A14 Li/Wang 等，*Quantum Assisted Combinatorial Benders' Algorithm ...* | 【未核实】只取到标题 + 摘要级条目（IEEE 10841900 / NSF PAR 10600327） | 取正文；确认其量子侧是否含约束 oracle 相位 |
| A17 *Grover Adaptive Search-Based Hybrid Benders Decomposition for MILP*（IEEE TQE） | 【二手：仅片段】作者名单、卷期、页码 **未能核实**；tqe.ieee.org 文章页 fetch 失败 | 直接取 IEEE TQE 页面确认题录与 oracle 结构 |
| A2 期刊版 *Quantum Sci. Technol.* 10, 015051 (2025) 全文 | 只读过摘要 + 本仓库既有调研对其原文措辞的转述（"forgo a QUBO formulation…"、"avoids the need for any penalty terms"） | 取 IOP 全文核对这三句措辞与公式，再写入论文 |
| A6 期刊版 *J. Heuristics* 32(2)（DOI 10.1007/s10732-026-09593-6） | 【未核实】ACM DL 需订阅 | 取期刊版确认 qubit 数与最大算例规模（arXiv v1 亦未读全文） |
| A1 的 rank QUBO 细节（"环由 CP 处理、未进入 QUBO"） | 【二手】来自本仓库既有调研，本次未读 ICAPS 全文 PDF（`ojs.aaai.org/.../download/19826/19585` fetch 失败） | 取 ICAPS PDF 逐条核对 QUBO 项与 LNS 邻域定义 |
| 「witness / constraint generation 作为量子相位或 QUBO 项」的直接对应工作 | **未命中**（G1 的检索面） | 需在 Scopus/WoS 做引文追踪：以 A1、A5、A6 为种子向前引用追踪；并检索量子约束满足（QCSP）与"cutting plane + QAOA"社群 |
| 「GNN/学习型 JSP 候选生成 + 量子」 | **未命中量子侧组合**；命中大量"GNN/RL + 经典 JSP"工作（如 [COR 2025 GNN+DRL FJSP](https://dl.acm.org/doi/10.1016/j.cor.2025.107155)） | 若论文要主张该组合新颖，须做一轮专门检索（学习型候选生成 × 量子求解器） |

### 3.4 需要人工复核的引用（题录级）

1. **A8 的作者与年份**：检索命中链接写作 `Casati-Giudici`，另有 `Rebecca Casati`、`Thomas Giudici`、`Pietro Torta` 的人物页命中，但**未取得题录页**。→ 全部标【未核实】。
2. **A14 的作者与出处**：命中 "Li-Wang" 与 IEEE 10841900，**未取得题录**。
3. **A17 的作者/卷期/页码**：**未能核实**。
4. **A6 期刊版卷期页码**：命中 "Journal of Heuristics: Vol 32, No 2"，**未取得题录页确认**。
5. **A9 是否已有期刊版**：仅见 [arXiv:2604.20321](https://arxiv.org/abs/2604.20321)（2026-04-22 提交），是否有期刊版 **未能核实**。
6. **A13 是否已正式发表**：仅见 arXiv v2（2026-09-05 修订），**未能核实**发表信息。
7. **本文件所有 URL**：均来自本次 `web_search`/`web_fetch` 实际返回；**未编造任何 URL、页码或卷期**。凡"未能核实"者已在表中显式标注。

---

## 4. 一句话收口

与我们的方法**结构最邻接**的是 A9（候选裁减 + 动态割 + 量子求解）、A11（量子采样 + 割平面 + 显式界）、A13（对角相位 oracle）；**要素层面**的"每机候选顺序""环约束""量子相位"分别都有已发表对应物（A1/A5/A6/A13），因此论文主张必须收窄到 **G1+G4 的交集**（见证投影下的 `L_W>T` 谓词编译及其资源账），并逐条对齐上表 B 的可检验表述；而 A8/A14/A17 三条**未取得正文**，在补齐前不得进入对照论证。
