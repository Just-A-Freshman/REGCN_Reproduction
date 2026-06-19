# 邻接矩阵构建修复：文件排序与目录创建

## 来源

来自 `17-code-bugs.md` 的 B11 和 B13，经审查验证属实。

## 涉及文件

- `dataprecossing/adjprocessing.py`
- `tests/test_preprocessing.py`

---

## B11：glob.glob 顺序不确定导致图与 VMD 分量错配

### 问题

`adjprocessing.py:15` 使用 `glob.glob()` 遍历 VMD CSV 文件，但 glob 不保证返回顺序：

```python
file = glob.glob(os.path.join("../data/data/VMDdata/%s/%s*.csv" % (dataset, j)))
```

同样地，`REGCN.py:122` 也使用 `glob.glob()` 遍历同一目录。两处代码依赖隐式的文件系统返回顺序一致，但不同操作系统或文件系统下顺序可能不同。

若 `0_3-1.csv` 和 `0_3-2.csv` 的返回顺序在两处代码中不一致，第一个 VMD 分量的预测数据将与第二个分量的邻接矩阵配对。

### 修复要求

在 glob 结果上显式调用 `.sort()`，使文件按名称排序：

```python
file = sorted(glob.glob(...))
```

两处（`adjprocessing.py` 和 `REGCN.py`）均需修复。

---

## B13：未创建保存目录导致 np.save 崩溃

### 问题

`adjprocessing.py:40` 在保存邻接矩阵前未确保目标目录存在：

```python
base = '../data/adj/' + dataset + '/'
adj = np.array(adj)
np.save(base + dataset + '_VMD_' + str(j) + '.npy', adj)
```

当 `../data/adj/SSE/` 目录不存在时（如新克隆的环境），`np.save` 抛出 `FileNotFoundError`。

### 修复要求

在首次 `np.save` 前添加：

```python
os.makedirs(base, exist_ok=True)
```

---

## 验收标准

1. 不同文件返回顺序下，图与 VMD 分量的配对一致
2. 空目录环境下运行不会因目录不存在而崩溃
3. 现有测试全部通过
