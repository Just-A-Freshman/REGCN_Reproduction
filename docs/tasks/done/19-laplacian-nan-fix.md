# Laplacian NaN 传播修复

## 来源

来自 `17-code-bugs.md` 的 B2，经审查验证属实。

## 涉及文件

- `REGCN/utils.py`（主要修改）
- `tests/test_trainmodel.py`（新增测试）

## 问题描述

`utils.py:7-16` 中的 `normalized_adj()` 对行和求 `-0.5` 次幂时，若行和为负数（经 B5 修复后强负相关边被保留），`np.power(负数, -0.5)` 返回 NaN：

```python
def normalized_adj(adj):
    adj = sp.coo_matrix(adj, dtype=np.float32)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.    # ← 只捕获 inf，不捕获 NaN
```

`np.isinf(NaN)` 为 `False`，NaN 未被清除，最终通过 `sp.diags(d_inv_sqrt)` 传播到整个 Laplacian 矩阵，使所有损失和预测坍缩为 NaN。

## 触发条件

当 Pearson/Spearman 邻接矩阵中某节点的强负相关边（权重 ≈ -0.95）的数量超过正相关边与自环之和（`1 + Σ正相关 - Σ负相关 < 0`）时触发。

## 修复要求

1. 在 `d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.` 之后增加 NaN 处理：`d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.`
2. 或者对行和取绝对值后再求幂，但需确认对称归一化的数学正确性
3. 为 `normalized_adj()` 添加单元测试，验证负行和情况下的输出不含 NaN

## 验收标准

1. 构造一个含强负相关的邻接矩阵（如 `[[1, -0.95], [-0.95, 1]]`），`calculate_laplacian()` 输出不含 NaN
2. 现有测试全部通过

## 测试代码更新

`tests/test_trainmodel.py` 中的 `test_calculate_laplacian_returns_sparse_tensor` 仅验证输出是否为 `SparseTensor`，未覆盖负行和情形。需新增一个测试用例：

```python
def test_normalized_adj_negative_rowsum(self):
    """负行和不应产生 NaN。"""
    from utils import normalized_adj
    import numpy as np
    # 2×2 邻接矩阵，行和 = 1 + (-0.95) = 0.05 > 0 通过
    # 需要更极端的：至少 4 个节点，其中 2 个强负相关
    adj = np.array([
        [0,  0.95, -0.95, -0.95],
        [0.95, 0,   0.95,  0.95],
        [-0.95, 0.95, 0,   0.95],
        [-0.95, 0.95, 0.95, 0]
    ], dtype=np.float32)
    result = normalized_adj(adj + np.eye(4))
    self.assertFalse(np.any(np.isnan(result.toarray())),
                     "负行和不应产生 NaN")
```

提交前验证该测试能通过。
