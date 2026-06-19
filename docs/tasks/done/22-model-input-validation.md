# 模型输入校验与防御性编程

## 来源

来自 `17-code-bugs.md` 的 B9 和 B10，经审查验证属实。

---

## B9：收盘价列索引硬编码

### 涉及文件

- `REGCN/REGCN.py`
- `tests/test_trainmodel.py`

### 问题

`REGCN.py:62` 硬编码收盘价列为索引 3：

```python
labels = data[:, 3]
```

SSE（9 列）和 DJIA（6 列）的收盘价均在第 3 列，因此当前正常工作。但若未来使用列数不同的数据集，此处会静默地使用错误的数据列，或少于 4 列时抛出 `IndexError`。

### 修复要求

将收盘价列索引作为可配置参数，从 `config.ini` 读取，或根据数据维度自动推断。

---

## B10：build() 未校验输入特征维度

### 涉及文件

- `REGCN/dgcgru.py`
- `tests/test_trainmodel.py`

### 问题

`dgcgru.py:29-72` 的 `build()` 方法忽略 `input_shape` 参数，未校验输入特征数与邻接矩阵维度 `n_gcn_nodes` 是否匹配：

```python
def build(self, input_shape):
    self.wz = self.add_weight(shape=(self.units, self.units), ...)
    # 未检查 input_shape[-1] == self._gcn_nodes
```

若 `X_train.shape[2]`（特征数）与邻接矩阵的维度不匹配，`K.dot(inputs, adj)` 会抛出难以调试的 TF 图级运行时错误。

### 修复要求

1. 在 `build()` 中添加显式断言：`assert input_shape[-1] == self._gcn_nodes, ...`
2. 提供清晰的错误信息，指出期望的维度

---

## 验收标准

1. 传入错误的特征维度时，模型在构建期（而非图执行期）抛出清晰的错误
2. 现有测试全部通过
