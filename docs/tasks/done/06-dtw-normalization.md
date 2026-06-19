# DTW 归一化公式修复

## B8：DTW 相似度计算错误

**文件：** `dataprecossing/adjprocessing.py` L33

**问题：** 运算符优先级导致公式计算错误。

**当前代码：**
```python
d = 1 - p / 10 * train_data.shape[0]
# 实际: 1 - (p / 10) * len   →  1 - p·len / 10
```

**论文公式：**
$$
\text{similarity} = 1 - \frac{\text{DTW距离}}{10 \times \text{序列长度}}
$$

**修复：** 给分母加括号：
```python
d = 1 - p / (10 * train_data.shape[0])
```

**验证：** 对比修复前后的 DTW 邻接矩阵数值。修复前相似度偏小，修复后应更接近理论值。
