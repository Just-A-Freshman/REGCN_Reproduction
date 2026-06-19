# 训练管道边界情况处理

## 来源

来自 `17-code-bugs.md` 的 B14 和 B15，经审查验证属实。

## 涉及文件

- `REGCN/REGCN.py`
- `tests/test_trainmodel.py`

---

## B14：空 glob 结果导致 np.sum 崩溃

### 问题

`REGCN.py:122-136` 中，若 VMD CSV 文件缺失（如未运行 data_VMD.py），`glob.glob` 返回空列表：

```python
file = glob.glob(os.path.join("%s%s/%s_*.csv" % (VMD_addr, datasets, s_index)))
VMD = []  # ← 若 file 为空，VMD 保持空列表
for f in file:
    VMD.append(pd.read_csv(f, header=None).values)

result = []
j = 0
for ndata in VMD:  # ← 跳过循环
    ...
    result.append(result1)
# result = []  ← 保持空

result = np.sum(result, axis=0)  # ← 崩溃：ValueError
```

### 修复要求

在 `np.sum` 之前检查 `result` 是否为空，若为空则打印错误信息并 `return`，而非崩溃。

---

## B15：VMD 分量预测窗口数不一致导致形状不匹配

### 问题

`REGCN.py:136` 将各 VMD 分量的预测结果按元素求和，隐式假设所有分量输出的测试窗口数相同：

```python
result = np.sum(result, axis=0)
```

若某个 VMD CSV 文件行数少于 `time_len`（如数据预处理异常），`trainmodel()` 会生成不同数量的测试窗口，`np.sum` 在形状不同的数组上抛出 `ValueError`。

### 修复要求

1. 在 `main()` 中校验各 VMD 分量的行数是否一致
2. 不一致时打印警告并跳过或截断至最短长度

---

## 验收标准

1. VMD CSV 缺失时，管道打印清晰错误信息而非崩溃
2. VMD 分量长度不一致时，管道优雅处理而非抛出未处理异常
3. 正常运行时不影响现有行为
4. 现有测试全部通过
