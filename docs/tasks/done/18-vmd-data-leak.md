# VMD 分解修复：消除测试数据影响

## 来源

来自 `17-code-bugs.md` 的 B1，经审查验证属实。

## 涉及文件

- `dataprecossing/data_VMD.py`（主要修改）
- `tests/test_preprocessing.py`（新增测试）

## 问题描述

`data_VMD.py:31-35` 将训练集、验证集、测试集的信号拼接后一次性送入 VMD：

```python
combined = np.concatenate([
    data[:train_size, i],
    data[train_size:val_size, i],
    data[val_size:row, i]
])
u_all, _, _ = VMD(combined, alpha, tau, K, DC, init, tol)
```

VMD 是一种全局信号分解方法：在时刻 t 的模态值依赖于整个输入信号。当训练集和测试集拼接后分解，训练段 t 时刻的分解结果会受到测试段 t+Δt 数据的影响，导致测试集信息泄漏到训练特征中。

## 预期行为

训练集和测试集应当分别、独立地进行 VMD 分解，使用相同的超参数（K 和 α），但不得拼接数据。具体来说：

1. 分别对 `data[:train_size]`（训练段）和 `data[train_size:]`（测试+验证段）调用 VMD
2. 使用相同的 K 和 α 参数
3. 各自拼接后保存

如果采用分别分解的方案，需要注意模态编号不对应的问题：两段独立 VMD 的第 k 个模态可能代表不同频率成分。建议优先采用"仅在训练集上拟合 VMD，再用同一组 VMD 分解器（通过显式传递分解参数）作用于测试集"的方案。

## 验收标准

1. VMD CSV 文件中训练段对应的行只依赖于训练集数据
2. 评估指标不再因测试数据混入而虚假乐观
3. 现有测试全部通过
