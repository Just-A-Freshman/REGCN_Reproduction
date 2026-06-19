# P3 — 低：清理和优化

## 问题 9：`main.py` 使用 `exec(open(...).read())` 反模式

**涉及文件：** `main.py` L20, L29, L37, L45

**问题：** 4 个管道运行函数使用 `exec(open("dataprecossing/XXX.py").read())` 执行脚本，绕过 Python 的 import 机制。

```python
# main.py:20 — 反模式
exec(open("dataprecossing/data.py").read())
```

**后果：** 文件路径变化产生不透明错误、命名空间污染、静态分析和断点调试失效。

**修复：** 改用 `importlib` 或直接函数调用。

---

## 问题 10：TF1 compat 与 TF2 Keras API 冲突

**涉及文件：** `REGCN/utils.py` L5-6, `REGCN/dgcgru.py` L7-11

**问题：** `utils.py` 使用 `tf.compat.v1.disable_v2_behavior()` 和 `tf.Session()`（TF1 模式），但 `dgcgru.py` 继承 `tf.keras.layers.AbstractRNNCell`，`REGCN.py` 使用 `tf.keras.optimizers.Adam`（TF2 API）。两者在 TF 2.x 中不兼容。

**修复：** 将 `utils.py` 中的 TF1 代码移植到 TF2（用 `@tf.function` 替代 `Session.run`），或统一使用 TF2 API。

---

## 问题 11：邻接计算硬编码了 0.8 训练比例

**涉及文件：** `dataprecossing/adjprocessing.py` L17

**问题：** 图构建使用 `train_size = int(tdata.shape[0] * 0.8)`，但 `config.ini` 配置了 `train_rate=0.7`。图反映了 80% 数据窗口的相关性，而模型仅用 70% 训练。

**修复：** 从 `config.ini` 读取 `train_rate`，或作为参数传入。

---

## 问题 12：弃用的 `DataFrame.append()`

**涉及文件：** `dataprecossing/normalization.py` L24

**问题：** `normalized_df.append([min_vals, max_vals], ignore_index=True)` 自 pandas 1.4.0 (2022) 弃用。

**修复：** 改用 `pd.concat([normalized_df, pd.DataFrame([min_vals, max_vals])], ignore_index=True)`。

---

## 问题 13：`calculate_laplacian()` 可批量调用

**涉及文件：** `REGCN/REGCN.py` L71-74

**问题：** 对 Pearson / Spearman / DTW 三张图分别调用 `calculate_laplacian()` + `tf.sparse.to_dense()`，共 3 次独立操作。每次做相同的 COO 转换、行求和、逆平方根、点积。

**修复：** 堆叠后一次调用，或向量化处理 `adj` 的第 0 轴。

---

## 问题 14：死代码与未使用导入

| 文件 | 行 | 内容 |
|------|-----|------|
| `input_data.py` | 5-6 | `load_price_data()` 函数体为 `pass`，从未调用 |
| `utils.py` | 57-62 | `weight_variable_glorot()` 从未调用 |
| `utils.py` | 80-93 | `get_vague_trend()` 从未调用 |
| `utils.py` | 103-107 | `get_total_relative_error()` 从未调用 |
| `REGCN.py` | 52 | `np.zeros()` 立即被下行覆盖 |
| `adjprocessing.py` | 38 | 第一次 `np.save()` 保存未阈值化数据，从未被加载 |
| `GA_VMD.py` | 3,5,8,9,11,12,15,17 | 8 个未使用导入 |
| `dgcgru.py` | 11-12 | 10 个未使用导入 |
| `data_VMD.py` | 1-2 | 重复 `import numpy` |
