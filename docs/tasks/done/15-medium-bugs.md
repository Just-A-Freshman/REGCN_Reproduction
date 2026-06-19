# P2：逻辑缺陷 / 代码质量问题

## B7：模型权重文件被 VMD 分量依次覆盖

**严重性：** P2 — 权重保存不可用

**问题：** `REGCN/REGCN.py` 第 104 行 `model.save_weights()` 的路径只包含股票索引 `s_index`，不包含 VMD 分量索引 `j`。训练循环中 K 个 VMD 分量依次将权重写入同一文件，只有最后一个分量的权重被保留。

```python
# 当前：
model_weights_path = './model/model_' + datasets + '-' + str(s_index) + '-weights.h5'
# 应改为：
model_weights_path = './model/model_' + datasets + '-' + str(s_index) + f'-comp{j}-weights.h5'
```

**涉及文件：** `REGCN/REGCN.py:104`

---

## B8：normalization.py 存在数据泄露且是死代码

**严重性：** P2 — 数据泄露隐患 + 废弃代码

**问题：** `dataprecossing/normalization.py` 中存在两个问题：

1. **数据泄露**：第 19-20 行 `df.min()` / `df.max()` 对**全量数据**（含测试集）计算最小最大值，归一化时测试集统计量污染了训练数据。
2. **死代码**：该脚本输出到 `../data/data/VMDnor/`，但管道中没有任何文件从该目录读取。`REGCN.py` 和 `adjprocessing.py` 均从 `../data/data/VMDdata/` 读取原始 VMD 数据。

**影响：** 泄露问题无实际影响（因为是死代码），但 `VMDnor/` 目录占用磁盘空间且容易误导后续维护者。

**涉及文件：** `dataprecossing/normalization.py:19-25`

---

## B9：data_VMD.py 分段独立拟合 VMD，模态编号可能不对应

**严重性：** P2 — 理论风险，实际影响不确定

**问题：** `dataprecossing/data_VMD.py` 第 33-35 行对训练集、验证集、测试集分别独立调用 VMD。VMD 是无监督分解算法，第 k 个模态在不同数据段上可能代表不同频率成分。

训练段的高频模态（索引 1）可能与验证段的另一频率被编号为索引 1，拼接后在分段边界处存在频率不连续，GRU 无法有效学习。

**实际影响：** VMD 在连续信号上较稳定，该问题在实际中可能不显著。但论文明确要求"先对训练集分解，相同参数再应用于测试集"以防止数据泄露——当前的三段独立分解做法与其不完全一致。

**涉及文件：** `dataprecossing/data_VMD.py:33-39`
