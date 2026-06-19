# 代码小问题审查

## 1. [ medium ] GA_VMD.py 中 `Fun()` 误用全局变量 `data`

- **文件**: [GA_VMD.py](../../dataprecossing/GA_VMD.py)
- **行号**: ~L45, L52
- **问题**: `Fun(x, data1)` 的定义参数是 `data1`，但函数体内却用了全局变量 `data`：
  - `for i in range(data.shape[2])` → 应使用 `data1.shape[1]`
  - `s /= data.shape[1]` → 应使用 `data1.shape[1]`
- **后果**:
  - 循环次数：`data.shape[2]`（特征数）恰好等于 `data1.shape[1]`（也是特征数），所以碰巧没出 bug
  - 除数：`data.shape[1]` 是总时间长度（SSE=726, DJIA=752），而正确应为特征数（SSE=9, DJIA=6）。不过 GA 的适应度是相对排序，等比例缩放不影响优化结果
- **修复**: 将函数体内 `data` 替换为 `data1`



