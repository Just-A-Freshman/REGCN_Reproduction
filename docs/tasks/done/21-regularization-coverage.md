# 正则化覆盖率修复：L1 应作用于全部可训练参数

## 来源

来自 `17-code-bugs.md` 的 B5，经审查验证属实。

## 涉及文件

- `REGCN/REGCN.py`
- `REGCN/dgcgru.py`（新增 L1 参数或 setter）
- `tests/test_trainmodel.py`

## 问题描述

论文 Eq.15 定义损失函数为：

```
L = (1/n) Σ(y - ŷ)² + λ₁·‖W‖₁ + λ₂·ACC(W)
```

其中 `λ₁·‖W‖₁` 应对**全部可训练参数**施加 L1 正则化。但当前代码仅在输出层的 Dense(1) kernel 上应用：

```python
model.add(Dense(1, kernel_regularizer=tf.keras.regularizers.l1(r_mse)))
```

GCGRU 内部的以下 11 个权重张量均未加任何正则化：

| 张量 | shape | 用途 |
|------|-------|------|
| `self.wz` | `(units, units)` | 更新门-输入 |
| `self.wr` | `(units, units)` | 重置门-输入 |
| `self.wh` | `(units, units)` | 候选状态-输入 |
| `self.uz` | `(units, units)` | 更新门-循环 |
| `self.ur` | `(units, units)` | 重置门-循环 |
| `self.uh` | `(units, units)` | 候选状态-循环 |
| `self.w0` | `(1, units)` | GCN 投影 |
| `self.wa` | `(3, N, N)` | 图融合权重 |
| `self.bz`, `self.br`, `self.bh` | `(units,)` | 偏置 |

## 修复要求

有两种方案可选：

### 方案 A：逐层添加 kernel_regularizer

在 `dgcgru.py` 的每个 `add_weight()` 调用中加入 `regularizer=tf.keras.regularizers.l1(r_mse)` 参数。这需要在 `gcgru.__init__` 中接收 `r_mse` 参数。

### 方案 B：在 trainmodel() 中统一收集并添加正则化损失

在 `REGCN.py` 的 `trainmodel()` 中，编译后通过 `model.add_loss()` 对全部可训练权重施加 L1 正则化：

```python
l1_reg = r_mse * sum(tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights)
model.add_loss(l1_reg)
```

## 验收标准

1. 训练时损失值比原始代码更高（因正则化项增大）
2. `model.trainable_weights` 中每个权重张量的正则化项均被计入总损失
3. 现有测试全部通过
