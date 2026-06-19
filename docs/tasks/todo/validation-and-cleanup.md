# 输入验证与代码清理

## 1. [ low ] `train_rate + val_rate > 1.0` 无校验

- **文件**: [REGCN.py](../../REGCN/REGCN.py#L49-L50)
- **问题**: `train_rate` 和 `val_rate` 从配置读取后直接使用，没有检查二者之和是否 ≤ 1.0。
- **触发条件**: 配置文件误填，例如 `train_rate=0.7, val_rate=0.4` → 和 = 1.1 > 1.0。
- **影响**: `val_size = int(time_len * 1.1) > time_len`，测试集切片 `data[val_size:time_len]` 为空。
  所有评估指标（ACC、R²、RMSE、MAE）接收空数组，输出无意义结果且无任何警告。
- **修复**: 在 `trainmodel()` 或 `__main__` 入口处添加校验：
  ```python
  if train_rate + val_rate >= 1.0:
      raise ValueError(f"train_rate + val_rate ({train_rate + val_rate}) must be < 1.0")
  ```

## 2. [ low ] 数据长度不足时滑窗为空，无任何错误提示

- **文件**: [input_data.py](../../REGCN/input_data.py#L22), [input_data.py](../../REGCN/input_data.py#L29), [input_data.py](../../REGCN/input_data.py#L36)
- **问题**: 滑窗迭代器 `range(len(train_data) - seq_len - pre_len + 1)` 在数据长度不足时为负数，
  Python 将负 `range` 视为空迭代器，`trainX`/`trainY` 等变为空列表。
- **触发条件**: 短时序数据，或 `seq_len` 设置过大。
- **影响**: 
  - 空数组传入 `model.fit()` 导致形状不匹配或静默无训练
  - 无任何警告或错误提示开发人员数据不足
- **修复**: 添加校验：
  ```python
  if len(train_data) < seq_len + pre_len:
      raise ValueError(f"train_data length ({len(train_data)}) < seq_len + pre_len ({seq_len + pre_len})")
  ```

## 3. [ low ] gcgru 细胞中 `s_index` 硬编码为 3

- **文件**: [REGCN.py](../../REGCN/REGCN.py#L93)
- **问题**: `cell = gcgru(..., s_index=3)` 将 GCN 输出的第 3 列（0-based，即第 4 个特征）作为预测目标。
  对 SSE（9 特征）和 DJIA（6 特征）而言列 3 恰好是收盘价，但参数名 `s_index` 易与股票索引混淆，
  且未来更换数据集时可能静默选错特征列。
- **影响**: 
  - 若数据集收盘价不在第 3 列，GCN 输出会提取错误特征进行 GRU 建模
  - `s_index` 命名与 `REGCN.py` 中 `s_index` 变量含义（股票索引）不一致，造成维护混淆
- **修复**: 将 `s_index=3` 改为从配置读取，或增加注释说明其含义为"收盘价在特征中的列索引"。

## 4. [ low ] `n_off` 配置参数和部分 import 未使用

- **文件**: [REGCN.py](../../REGCN/REGCN.py#L18-L21), [REGCN.py](../../REGCN/REGCN.py#L44)
- **问题**: 
  - `from tensorflow.keras.layers import Input` — 未使用
  - `from tensorflow.keras.models import Model` — 未使用
  - `from tensorflow.keras.layers import TimeDistributed` — 未使用
  - `n_off = int(config["hyper"]["n_off"])` — 读取后未使用
- **影响**: 代码维护负担，阅读者可能误以为这些是重要的导入/配置项。
- **修复**: 删除未使用的 import 行和 `n_off` 配置读取（如确需保留应加注释说明用途）。
