# P0：致命崩溃

## B1：GA_VMD.Fun() 引用未定义的全局变量 `low` / `ub`

**严重性：** P0 — 管道崩溃

**问题：** `dataprecossing/GA_VMD.py` 中 `Fun()` 函数（第 24-31 行）引用了 `low[0]`、`low[1]`、`ub[0]`、`ub[1]` 作为模块级全局变量。但这些变量在重构后只在 `run()` 函数内部（第 229-230 行）作为局部变量定义，不在模块的全局作用域中。

调用链：`run()` → `GeneticAlgorithm.solve()` → `evaluate()` → `calculateFitness()` → `Fun()` → **NameError**

**复现条件：** `python main.py --pipeline ga-vmd --dataset SSE`

**现象：** `NameError: name 'low' is not defined`

**修复思路：** 将 `low` 和 `ub` 作为参数传入 `Fun()` 和 `GAIndividual.calculateFitness()`，或者在 `run()` 内设置为模块全局变量。

**涉及文件：** `dataprecossing/GA_VMD.py:24-31`, `dataprecossing/GA_VMD.py:75`

---

## B2：main.py 缺少 `import numpy`

**严重性：** P0 — 管道崩溃

**问题：** `main.py` 第 69 行调用 `np.load()`，但整个文件没有任何 `import numpy` 语句（也没有 `from numpy import`）。`main.py` 仅导入了 `argparse`、`subprocess`、`sys`、`os`。

**复现条件：** `python main.py --pipeline train --dataset DJIA`

**现象：** `NameError: name 'np' is not defined`

**涉及文件：** `main.py:69`（缺失 `import numpy as np`）

---

## B3：adjprocessing.py 加载不存在的 `_fea.npy` 文件

**严重性：** P0 — 管道崩溃

**问题：** `dataprecossing/adjprocessing.py` 第 12 行 `np.load(data_addr + dataset + '_fea.npy', ...)` 加载 `{dataset}_fea.npy` 文件，但 `dataprecossing/data.py` 的输出是 `{dataset}.npy`（不带 `_fea` 后缀）。`data/data/` 目录中不存在任何 `_fea` 文件。

**复现条件：** `python main.py --pipeline adj --dataset SSE`

**现象：** `FileNotFoundError: ../data/data/SSE_fea.npy`

**修复思路：** 将 `_fea.npy` 改为 `.npy`，或确认文件来源并在管道中新增生成步骤。

**涉及文件：** `dataprecossing/adjprocessing.py:12`
