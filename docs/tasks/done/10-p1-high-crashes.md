# P1 — 高：其他进程内崩溃

## 问题 3：VMD CSV 缺少反归一化所需元数据

**涉及文件：** `REGCN/REGCN.py` L133-136，`dataprecossing/data_VMD.py`

**根源：** `main()` 期望 VMD CSV 末尾有两行额外的元数据（第 `time_len` 行为最小值，第 `time_len+1` 行为最大值），用于 `unautoNorm()` 反归一化。但 `data_VMD.py` 只输出了 `time_len` 行数据，没有追加元数据行。

```python
# REGCN.py:135-136
mins = ndata[time_len][3]     # IndexError: 超出 CSV 行数
maxs = ndata[time_len + 1][3]
```

**后果：** 任何对 `main()` 的调用都因 `IndexError: index out of bounds` 崩溃，无法执行评估和绘图。

**修复方案：** 两种选择：
1. 在 `data_VMD.py` 中追加 min/max 行至 CSV
2. 或修改 `REGCN.py` 从原始数据计算 min/max（更健壮，不依赖 CSV 格式）

---

## 问题 4：`result[:, -1]` 扁平化后下游函数索引崩溃

**涉及文件：** `REGCN/REGCN.py` L140-156，`REGCN/utils.py` L73-78（`get_trend()`）

**根源：** `result = result[:, -1]` 将 `(n, 1)` 变为 `(n,)` 1D 数组。但 `get_trend()` 和 `avg_relative_error()` 内部使用 `cur[i, 0]` 和 `pred[i, 0]` 二维索引。

```python
# utils.py:73  — 访问 1D 数组的 [i, 0] 索引
if cur[i, 0] - pre[i, 0] > 0:

# utils.py:98
total += abs(pred[i, 0] - actual[i, 0]) / actual[i, 0]
```

**后果：** 在第 144 行 `get_trend(pre_y_test, result)` 处 `IndexError: too many indices for array`。

**修复方案：** 将 `result[:, -1]` 改为保留 2D 形状，或修改 `get_trend()` 和 `avg_relative_error()` 支持 1D 输入。

---

## 问题 5：VMD 文件名格式不匹配

**涉及文件：** `dataprecossing/data_VMD.py` L61，`REGCN/REGCN.py` L123

**根源：** `data_VMD.py` 生成的文件名格式为 `03-1.csv`（股票 0，K=3，IMF 1），但 `REGCN.py` 的 glob 模式是 `0_*.csv`，两者不匹配。

```python
# data_VMD.py:61 — 无下划线分隔
str(j) + str(K) + "-" + str(i + 1) + ".csv"  # 例如 "03-1.csv"

# REGCN.py:123 — 期望下划线
glob.glob(".../%s_*.csv" % s_index)           # 例如 ".../0_*.csv"
```

**后果：** `glob` 返回空列表，`result = []`，`np.sum([], axis=0)` 返回标量，`result[:, -1]` 触发 TypeError。

**修复方案：** 在 `data_VMD.py` 的 `str(j)` 后添加下划线，或修改 glob 模式去掉下划线。

---

## 问题 6：`main.py run_training()` 忽略 `--dataset` 参数

**涉及文件：** `main.py` L61-71

**根源：** `run_training(dataset, stock_index)` 从未使用 `dataset` 参数。`REGCN/REGCN.py` 在 import 时从 `config.ini` 读取 `datasets` 值。

```python
# main.py:65
from REGCN.REGCN import main, data  # REGCN.py 在此处读取 config.ini

# REGCN.py:37 — 永远用 config.ini 的值，而非 CLI
datasets = config["hyper"]["datasets"]
```

**后果：** `python main.py --dataset SSE --pipeline train` 打印 `SSE` 但实际加载 `../data/adj/DJIA/` 数据，产生静默错误的预测。

**修复方案：** 在 `run_training()` 中，import 后覆写 `REGCN.REGCN.datasets = dataset`，或通过函数参数传递 datasets 值。
