# JSP 析取图分解研究原型

这是数学建模与证书机制的正确性原型，不是已经实现量子加速的大规模求解器。

- `jsp_core.py`：析取图解码、独立排程验证、机器子集下界、环/路径约束分离、机器 QUBO 构造、小实例精确约束生成、整数拉格朗日不可行证书。
- `run_validation.py`：离线重跑小实例和三个上传实例的验证。
- `tabu_baseline.cpp`：独立的经典禁忌搜索参考程序，与量子分解算法没有性能等价关系。
- `data/`：上传数据的原样副本。
- `reference/`：公开机器顺序，文件内明确标注来源；不是本研究算法求得的解。
- `results/`：计算结果、排程、整数下界证书及运行日志。
- `JSP_数学推导与验证报告.md`：完整模型、证明、限制及研究建议。

## 运行

依赖 Python 3.10+、NumPy、SciPy（本次为 SciPy 1.17.0），经典参考程序需要 C++17。

```bash
python run_validation.py

g++ -O3 -std=c++17 tabu_baseline.cpp -o tabu_baseline
./tabu_baseline data/tai50_20_02.txt 60 20260929 2869 results/ta62_baseline_order.txt > results/ta62_baseline_run.json 2> results/ta62_baseline_progress.log
./tabu_baseline data/tai50_20_02.txt 120 20260929 2869 results/ta62_insertion_order.txt insert > results/ta62_insertion_run.json 2> results/ta62_insertion_progress.log
python run_validation.py
```

运行时间预算受硬件影响，迭代数及最终结果不保证跨机器完全一致。固定种子仅保证相同实现、执行路径与迭代预算下的随机序列。

`exact_cut_decision` 和 `certify_cut_infeasibility` 显式枚举机器排列，只用于小实例。不要把它们直接用于 50×20，也不要把小实例的完备性证明理解为多项式时间保证。

内部工序 ID、机器 ID 和公开机器顺序采用从 0 开始的编号。交付的排程 CSV 中工件、工序、机器编号均从 1 开始；时间为原始数据的时间单位，不能擅自解释为秒。

未连接或运行量子退火硬件。未对 50×20 实例运行完整 QUBO 分解优化。提供的两份最优排程来自公开资料，并在上传数据上重新验证；另一实例的参考启发式结果未达到已知最优值。不能用这些结果主张量子优势。
