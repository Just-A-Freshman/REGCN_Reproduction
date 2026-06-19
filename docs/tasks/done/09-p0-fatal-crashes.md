# P0 — 致命：模型训练崩溃

## 问题 1：模型输出 shape 与训练目标不匹配

**涉及文件：** `REGCN/REGCN.py` L79-81，`REGCN/losses.py` L15-16

**根源：** 模型输出 `(batch, 1)`，但训练目标 `y_train` 是 `(batch, seq_len, 1)`。因为 `preprocess_data()` 构建了长度 `seq_len` 的标签窗口。

```python
# REGCN.py:79-81 — 当前
model.add(RNN(cell, return_sequences=False))  # → (batch, n_neurons)
model.add(Dense(1))                            # → (batch, 1)

# input_data.py:31 — 每个样本包含 seq_len 个标签
trainY.append(b[pre_len: seq_len + pre_len])   # → 长度 seq_len
```

**后果：** `losses.py:15` 将两者 reshape 为 `[-1]`：y_true → `(batch*seq_len,)`，y_pred → `(batch,)`。减法 `y_true - y_pred` 长度不匹配，`model.fit()` 直接崩溃。

---

### 修复方案（选用方案 A）

**Step 1：修改 REGCN.py L80 — 输出完整序列**

```python
# 修改前:
model.add(RNN(cell, return_sequences=False))

# 修改后:
model.add(RNN(cell, return_sequences=True))
```

改动后 RNN 输出 `(batch, seq_len, n_neurons)`，经 Dense(1) 后为 `(batch, seq_len, 1)`，与 y_train 的 `(n, seq_len, 1)` 匹配。

**Step 2：修改 losses.py — 趋势差分为逐序列内计算**

当前损失函数将所有样本扁平化后做全局差分，存在跨样本边界问题（样本 i 的最后一天与样本 i+1 的第一天做 sign 比较，无意义）。

```python
# losses.py 修改后:
def loss(y_true, y_pred):
    # MSE 分量
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # 趋势方向分量 — 在 seq_len 维度内逐序列计算差分
    # y_true/y_pred 形状: (batch, seq_len, 1)
    true_diff = y_true[:, 1:, :] - y_true[:, :-1, :]  # (batch, seq_len-1, 1)
    pred_diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]  # (batch, seq_len-1, 1)

    true_sign = tf.sign(true_diff)
    pred_sign = tf.sign(pred_diff)

    trend_penalty = tf.reduce_mean(
        tf.cast(tf.not_equal(true_sign, pred_sign), tf.float32)
    )

    return mse + r_acc * trend_penalty
```

关键改动：`y_true[1:]` → `y_true[:, 1:, :]`，差分局限在每个序列内部，不会跨样本边界比较。

---

## 问题 2：GRU 门控使用当前输入替代历史状态 ✅ 已修复

**涉及文件：** `REGCN/dgcgru.py` L86-89

**根源：** 更新门 `z` 和重置门 `r` 的递归权重计算使用了当前输入 `x`，而不是上一步的隐状态 `state`。

```python
# 错误的代码 (L86, L88):
z = K.dot(x, self.wz) + K.dot(x, self.uz) + self.bz  # 第二项应为 state
r = K.dot(x, self.wr) + K.dot(x, self.ur) + self.br  # 第二项应为 state
```

**当前状态：** ✅ 已修复

```python
# dgcgru.py:83,85 — 当前代码
z = K.dot(x, self.wz) + K.dot(state, self.uz) + self.bz
r = K.dot(x, self.wr) + K.dot(state, self.ur) + self.br
```
