# 最严重问题：训练流程断裂 + 核心损失函数缺失

## 问题定位

### 1. 训练函数 `trainmodel()` 无法执行

**文件：** REGCN/REGCN.py `trainmodel()` 函数（L76-93）

**现象：** 调用时报 `NameError`，`Ge` 未定义即用于 `fit()` 和 `predict()`。

**根因分析：** 函数体只有三行有效代码，但彼此断裂：

```python
# L76: gcgru cell 创建成功
cell = gcgru(n_neurons, Madj, n_gcn_nodes, 3)

# L78: Sequential 模型创建成功
model = Sequential()

# 缺失：RNN 包装 cell、Dense 输出层、compile
# L81-88: Ge.fit(...)   ← Ge 未定义，且 lr_scheduler 未定义
```

完整流程缺了：`model.add(RNN(cell))` → `model.add(Dense(1))` → `model.compile(loss=..., optimizer=...)`，跳到了根本不存在的 `Ge.fit()`。

### 2. 趋势正则化损失函数未实现

**论文公式 14-15：** 这是论文区别于基线模型的核心创新。

```
L_total = MSE(y, ŷ) + λ₁·‖w‖₁ + λ₂·(1/n) Σ 1[sign(Δy) ≠ sign(Δŷ)]
```

代码仅用默认 MSE，缺少：
- L1 权重稀疏约束（λ₁=0.01）
- 趋势方向惩罚项（λ₂=0.1）

---

## 实现步骤

### Step 1：修复训练函数

打开 `REGCN/REGCN.py`，对 `trainmodel()` 做以下改动：

1. 将 `model = Sequential()` 改为模型组装 + 编译：

```python
model = Sequential()
# cell 包装进 RNN 层，输出 hidden_state
model.add(RNN(cell, return_sequences=False))
# 投影到 1 维输出（收盘价）
model.add(Dense(1))
# 编译
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=custom_loss
)
```

2. 删除 `lr_scheduler` 的引用（或替换为 `ReduceLROnPlateau`）
3. 将所有 `Ge` 替换为 `model`
4. 删除无效回调引用或替换为有效回调

### Step 2：实现趋势正则化损失函数

新增 `REGCN/losses.py`：

```python
def trend_regularized_loss(y_true, y_pred):
    # 1. MSE 分量
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # 2. L1 正则化（从 model.losses 自动收集，或手动加）
    l1_reg = tf.reduce_sum([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])  # 近似
    
    # 3. 趋势方向惩罚
    y_true_diff = y_true[1:] - y_true[:-1]
    y_pred_diff = y_pred[1:] - y_pred[:-1]
    trend_err = tf.reduce_mean(
        tf.cast(tf.sign(y_true_diff) != tf.sign(y_pred_diff), tf.float32)
    )
    
    lambda_1 = 0.01  # 论文 Sec 4.2.4
    lambda_2 = 0.1
    
    return mse + lambda_1 * l1_reg + lambda_2 * trend_err
```

> 注意：上述 L1 计算方式为示意，实际应使用 `model.add_loss()` 或将 L1 加入 `model.losses`。

### 验证方法

1. 运行 `python -c "from REGCN.REGCN import trainmodel; print('OK')"` —— 不报 `NameError` 即通过
2. 在真实数据上运行一次训练，观察 loss 下降 —— loss 应稳定下降，且包含三部分分量
