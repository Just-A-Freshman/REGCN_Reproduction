# P1：结果错误 / 严重逻辑缺陷

## B4：unautoNorm 对 VMD 分量应用收盘价缩放，导致预测值偏差约 100 倍

**严重性：** P1 — 预测结果完全错误

**问题：** `REGCN/REGCN.py` 第 140 行的 `unautoNorm(result1, close_min, close_max)` 使用原始收盘价的范围（`close_max - close_min`）对**单个 VMD 分量**的预测值进行反归一化。

但 VMD 各分量本就在原始数据的量纲内（VMD 分解保持原始数据的量纲）。对单个分量应用收盘价缩放会将其放大 100 倍左右，然后再累加，导致集成预测结果出错。

**示例：** 以 DJIA 为例，`close_min=100`, `close_max=200`, `range=100`。VMD 高频残差分量预测值 +5.0，`unautoNorm` 后变为 5×100+100=600。正确做法是不做缩放直接累加各分量。

**修复思路：** 移除 `unautoNorm` 调用，直接累加各分量的原始预测值；或者确认 VMD 数据已被归一化（见 B8 注释）后再使用反归一化。

**涉及文件：** `REGCN/REGCN.py:50-54`（unautoNorm 函数）, `REGCN/REGCN.py:140`（调用处）

---

## B5：邻接矩阵阈值化丢弃强负相关性

**严重性：** P1 — 图结构丢失关键关系

**问题：** `dataprecossing/adjprocessing.py` 第 41 行 `adj[adj < threshold] = 0` 使用 `threshold=0.9`。该条件对**所有**小于 0.9 的值置零，包括 -0.95、-0.98 等**强负相关**值（因为 -0.95 < 0.9 为真）。

Pearson 和 Spearman 相关系数的取值范围是 [-1, 1]，负相关表示强反向关系，与正相关同等重要。正确的做法是保留绝对值大的值：`adj[np.abs(adj) < threshold] = 0`。

**示例：** "开盘价"和"收盘价"的 Pearson 为 -0.95（强负相关），被阈值化后该边消失，GCN 消息传递遗漏了这一关键反向关系。

**涉及文件：** `dataprecossing/adjprocessing.py:41`

---

## B6：GA_VMD 硬编码训练比例 0.8，与主训练管道的 0.7 不一致

**严重性：** P1 — 超参数优化目标偏移

**问题：** `dataprecossing/GA_VMD.py` `run()` 第 226 行使用 `train_size = int(tdata.shape[0] * 0.8)`（硬编码 0.8），但 `REGCN/REGCN.py` 从 `config.ini` 读取 `train_rate=0.7`。

VMD 参数（K, α）在 `tdata[:0.8×len]` 上优化，但实际训练只在 `tdata[:0.7×len]` 上进行。GA 看到的 70%-80% 区间在实际训练时不存在，导致优化目标偏移。

**修复思路：** 将 `GA_VMD.py` 中的 `0.8` 改为引用 `train_rate` 参数，或从 `config.ini` 读取。

**涉及文件：** `dataprecossing/GA_VMD.py:226`
