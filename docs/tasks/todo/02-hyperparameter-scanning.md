# 超参数敏感性分析（Table 7 + Fig. 4）

## 目标

自动化两个超参数扫描实验：
1. **Table 7** — 滑动窗口长度 `seq_len` 对性能的影响
2. **Fig. 4** — 正则化系数 `r_mse` (λ₁) 和 `r_acc` (λ₂) 组合对性能的影响

## 论文参考

### Table 7：滑动窗口长度
测试 `seq_len ∈ {7, 15, 30, 45, 60}`，分别在 SSE 和 DJIA 上评估 5 个指标。

### Fig. 4：正则化系数敏感性
遍历 `r_mse` 和 `r_acc` 的组合，观察 ACC 和 RMSE 的变化趋势（论文为热力图/折线图）。

## 现有状态
当前只能手动修改 `config.ini` 的 `seq_len` 或 `r_mse`/`r_acc`，每次改完手动运行，无自动化脚本。

## 需求

### 1. 滑动窗口扫描

```bash
python main.py --pipeline scan-window --dataset SSE
```

内部逻辑：
- 遍历 `seq_len ∈ [7, 15, 30, 45, 60]`
- 每次修改配置后完整运行训练/评估
- 产出 `result/<dataset>/window_sensitivity.csv`：

```
seq_len,dataset,acc,r2,rmse,mae,mape
7,SSE,0.6802,0.9398,0.1600,0.1214,0.888
15,SSE,0.6901,0.9289,0.1562,0.1175,0.865
30,SSE,0.6940,0.9226,0.1500,0.1137,0.843
45,SSE,0.6797,0.9299,0.1531,0.1161,0.844
60,SSE,0.6733,0.9145,0.1550,0.1144,0.867
```

### 2. 正则化系数扫描

```bash
python main.py --pipeline scan-reg --dataset SSE
```

扫描范围（论文 Fig. 4 使用的范围）：

| 参数 | 取值范围 |
|---|---|
| `r_mse` | 0, 1e-4, 1e-3, 0.01, 0.1, 1.0（论文 x 轴） |
| `r_acc` | 0, 0.01, 0.05, 0.1, 0.2, 0.5（论文 y 轴） |

产出 `result/<dataset>/reg_sensitivity.csv`，格式为全组合：

```
r_mse,r_acc,dataset,acc,r2,rmse,mae,mape
0.01,0.0,SSE,0.6795,0.9225,0.1595,0.1179,0.873
0.01,0.1,SSE,0.6940,0.9226,0.1500,0.1137,0.843
...
```

### 3. 可视化输出

`scan-reg` 模式自动产出热力图（`.png`），参考 Fig. 4 格式：

```bash
python main.py --pipeline scan-reg --dataset SSE --plot
# → 输出: result/SSE/reg_heatmap_acc.png
# → 输出: result/SSE/reg_heatmap_rmse.png
```

使用 `matplotlib` 的 `imshow` 或 `seaborn.heatmap` 绘制。

### 4. 实现要点

- **配置隔离**：每次扫描迭代时，修改 `config.ini` 的运行副本，不污染原始配置
- **只扫一支股票**：超参数扫描可只对 SSEx600000（或第一支股票）运行，减少总耗时
- **结果缓存**：若某组合已有结果则跳过，支持断点续扫
- **日志记录**：每次迭代记录当前参数组合和已完成状态

### 5. 执行效率优化

扫描组合数：
- Window scan: 5 种 × 每支股票（建议限 1 支）= 5 次训练
- Reg scan: 6×6 = 36 种 × 每支股票（建议限 1 支）= 36 次训练

建议在 `scan` 模式下增加 `--stock` 参数指定用于扫描的股票索引。
