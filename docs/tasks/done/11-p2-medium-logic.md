# P2 — 中：结果偏差（逻辑错误）

## 问题 7：GA 交叉操作覆盖染色体后读取修改值

**涉及文件：** `dataprecossing/GA_VMD.py` L196-203

**根源：** `crossoverOperation()` 先计算 `newpop[i].chrom[j]` 并直接覆盖原数组，随后读取该已修改值计算第二个后代，导致 `child2` 偏向第二个父本。

```python
# L199: newpop[i].chrom[j] 被覆盖为 child1
newpop[i].chrom[j] = newpop[i].chrom[j] * alpha + (1-alpha) * newpop[i+1].chrom[j]

# L202: 此时 newpop[i].chrom[j] 已是 child1，不是原始父本
newpop[i+1].chrom[j] = newpop[i+1].chrom[j] * alpha + (1-alpha) * newpop[i].chrom[j]
# 实际结果: child2 = 0.25·p1 + 0.75·p2 (alpha=0.5时)
# 期望结果: child2 = 0.5·p2 + 0.5·p1
```

**修复方案：** 在计算前暂存原始值：
```python
temp = newpop[i].chrom[j]
newpop[i].chrom[j] = temp * alpha + (1-alpha) * newpop[i+1].chrom[j]
newpop[i+1].chrom[j] = newpop[i+1].chrom[j] * alpha + (1-alpha) * temp
```

---

## 问题 8：GA 适应度函数未取 Spearman 绝对值

**涉及文件：** `dataprecossing/GA_VMD.py` L52

**根源：** `Fun()` 对 Spearman 相关性直接求和，正值和负值会互相抵消。GA 最小化适应度，因此会偏向产生负相关残差的参数组合。

```python
# GA_VMD.py:52
s += df.corr()[0][1]     # 直接累加，未取绝对值
```

**后果：** 一个特征 +0.50 另一个 -0.50，平均 0.00 看似最优，但两个特征分解都不好。GA 收敛到次优的 VMD 参数 (K, α)。

**修复方案：** 改为累加绝对值 `s += abs(df.corr()[0][1])`，或等效地将最小化目标转为最大化绝对相关性。
