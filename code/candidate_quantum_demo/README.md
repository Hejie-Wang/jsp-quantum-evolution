# 机器候选量子搜索：最小数学 demo

这是 [数学设计](../../docs/quantum_candidate_design.md) 的 CPU 验证参考。后续工程工作见 [Codex 任务清单](../../docs/quantum_candidate_codex_tasks.md)。

## 验证内容

- 固定 3×3 JSP，每机器两个候选，共 8 个组合；包括可行组合和跨机器有向环。
- 从完整图提取环或最长路径，将先后关系转换为每机器候选集合。
- 构造 `H = T + (U+1) * (active_cycles + active_overlong_paths)`，验证候选池内基态能量等于真实最优工期。
- 演示精确小主问题与图分离器的约束生成循环。
- 比较普通混合与由见证提出的联合混合，执行有限时间理想演化和能隙网格计算。

**没有 CUDA、门级 Qiskit 电路、硬件调用或量子优势。** 候选池是人工选择的验证 fixture，包含优质排程不意味着算法搜索发现了它。

## 运行

Python 3.10+：

```bash
cd code/candidate_quantum_demo
python -m pip install -r requirements.txt
python -m unittest -v test_demo
python demo.py --output results/local_result.json
```

`--tau` 是归一化 Hamiltonian 下的无量纲总演化时间，`--steps` 为分段常值中点演化步数；不是硬件秒数或门层数。

```bash
python demo.py --tau 20 --steps 160 --output results/refined_result.json
```

## 基底与编码

生产设计采用 one-hot 机器候选；demo 直接在**合法候选子空间**工作，不分配非法 one-hot 状态。3 个两候选寄存器在该子空间等价于 3 个候选标签位，但实际 one-hot 数据位数为 6。

T 使用 5 位，取值 0..31；全部工时之和 U=22，罚系数为 23。超过 U 的 T 不会成为基态，因为已有可行状态能量不超过 U。

候选组合按 `itertools.product` 的字典序，线性索引为 `choice_index * 32 + T`。模拟维度 256；不是已经编译的 11 位物理电路，也不是大规模模拟方法。

## 默认结果

运行快照：[results/demo_result.json](results/demo_result.json)。

| 检查 | 结果 |
|---|---:|
| 候选组合 / 可行组合 | 8 / 5 |
| 候选池内最佳工期 | 11 |
| 目标 Hamiltonian 基态能量 | 11 |
| 完整覆盖见证数 | 6 |
| 从见证生成的联合转移数 | 4 |
| 普通驱动基态概率，tau=20、steps=80 | 约 0.01909 |
| 联合驱动基态概率，同预算 | 约 0.01198 |

本次联合驱动更差。保留它是为了区分模型正确与动力学性能，不据此宣称联合驱动普遍无效，也不筛选有利参数后宣称优势。

两个变体均采用可证明的 `||H(s)|| <= 1` 归一化，联合项在两个端点为零。归一化控制能量尺度，但不能代替实际门数、噪声和时间公平性。

## 函数索引与限制

| 函数 | 用途 |
|---|---|
| `decode` / `verify_starts` | 图评价 / 独立排程验证 |
| `separate` / `Witness.active` | 见证到候选集合投影 |
| `energy_table` | 小规模目标谱的经典诊断表 |
| `exact_cut_loop` | 使用精确经典 master 的有限割循环 |
| `covering_witnesses` | 枚举 8 个组合构造覆盖证书，仅验证用 |
| `joint_specs` | 仅从见证集合提出联合跃迁，不读取最优标签 |
| `drivers` / `evolve` | 合法子空间驱动、理想离散演化和谱网格 |

`energy_table`、完整候选枚举、稠密矩阵全部是测试参考，生产实现禁止依赖这些结构。每个可行组合的一条最长路径、每个有环组合的一个环已足以覆盖本 fixture；并未声称枚举了所有图路径。

六项测试覆盖独立可行性、见证有效性、阈值边界、基态对应、未完整模型及割闭合、驱动厄米性和演化范数。能隙网格不是连续最小能隙证书，有限时间演化不是绝热保证。

本次验证环境：Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0。时间步数从 80 增至 160 后，两种驱动的目标概率变化均小于 4.4e-6，默认比较方向不变；这是一次步长敏感性检查，不是严格误差界。
