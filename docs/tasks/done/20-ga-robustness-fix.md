# GA 优化鲁棒性修复

## 来源

来自 `17-code-bugs.md` 的 B4 和 B12，经审查验证属实。

## 涉及文件

- `dataprecossing/GA_VMD.py`
- `tests/test_preprocessing.py`

---

## B4：轮盘赌选择偏向第一个个体

### 问题

`GA_VMD.py:142-152` 的轮盘赌选择实现中，当 `random()` 返回的值大于 `accuFitness` 数组所有元素时（浮点舍入导致 `accuFitness[sizepop-1] < 1.0`），循环遍历完所有候选者后无任何分支命中，`idx` 保持默认值 0：

```python
idx = 0
for j in range(0, self.sizepop - 1):
    if j == 0 and r < accuFitness[j]:
        idx = 0
        break
    elif r >= accuFitness[j] and r < accuFitness[j + 1]:
        idx = j + 1
        break
# 若 r >= accuFitness[sizepop-2] 且 r < accuFitness[sizepop-1] 不成立
# 且 r < accuFitness[0] 不成立 → idx 保持 0
newpop.append(self.population[idx])  # idx=0 → 偏向第一个个体
```

### 修复要求

1. 将循环逻辑改为能兜底处理越界情况，或使用 `np.random.choice` 配合归一化的 fitness 权重
2. 确保当 `r ≈ 1.0` 时落在最后一个区间

---

## B12：VMD 收敛失败导致 GA 适应度 NaN

### 问题

`GA_VMD.py:29` 中 VMD 对某些 (K, α) 组合可能不收敛，返回包含 NaN 的模态输出。NaN 通过以下路径传播：

```
VMD() → u 含 NaN → Fun() → fitness[i]=NaN → totalFitness=NaN → accuFitness=NaN → 轮盘赌退化
```

### 修复要求

1. 在 `Fun()` 中检测 VMD 输出是否含 NaN
2. 若含 NaN，跳过该样本的适应度计算或返回一个惩罚值（如 1.0）
3. 在 `evaluate()` 中处理 fitness 为 NaN 的情况

---

## 验收标准

1. GA 连续运行 100 代，不同种群的 idx 分布均匀（不偏向第 0 个个体）
2. VMD 不收敛时不会使整个 GA 进程崩溃
3. 现有测试全部通过

## 相关测试更新

`tests/test_preprocessing.py` 中的 `test_fun_uses_absolute_correlation` 使用字符串匹配检查 `GA_VMD.py` 中是否存在 `abs(df.corr()[0][1])`。但 Fun() 经过 B12 修复重构后使用了中间变量：

```python
corr_val = df.corr()[0][1]
s += abs(corr_val) if not np.isnan(corr_val) else 0
```

该测试不再命中，需将其断言更新为匹配新的代码模式，例如改为检查 `abs(corr_val` 字符串的存在。
