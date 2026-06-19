# 消融实验自动化（Table 6）

## 目标
自动化运行论文中 9 种消融变体，输出统一性能对比表，验证各模块贡献。

## 论文参考
Table 6 定义了 10 种方法（含本方法自身），通过逐模块替换/移除来验证各组件贡献：

| 方法 | 改变 | 验证目标 |
|---|---|---|
| Method 1 | VMD → EMD | 分解算法选择的影响 |
| Method 2 | VMD → EEMD | 同上 |
| Method 3 | VMD → CEEMDAN | 同上 |
| Method 4 | VMD → SSA | 同上 |
| Method 5 | 去掉序列分解 | 分解模块整体贡献 |
| Method 6 | Multi-GCN → GCN（仅 Pearson） | 多图融合贡献 |
| Method 7 | Multi-GCN → GCN（仅 Spearman） | 多图融合贡献 |
| Method 8 | Multi-GCN → GCN（仅 DTW） | 多图融合贡献 |
| Method 9 | 去掉 GCN（仅 GRU） | GCN 模块贡献 |
| Method 10 | 去掉趋势正则化（仅 MSE loss） | 正则化项贡献 |
| Proposed | 完整模型 | 基准 |

## 现有状态
当前代码只支持完整 REGCN 模型，无消融模式开关。

## 需求

### 1. 消融模式注册机制

在 `main.py` 中增加消融模式参数，示例：

```bash
# 运行特定消融变体
python main.py --pipeline train --ablation no-vmd --dataset SSE

python main.py --pipeline train --ablation single-graph pearson --dataset SSE

python main.py --pipeline train --ablation no-gcn --dataset SSE

python main.py --pipeline train --ablation no-trend-loss --dataset SSE
```

### 2. 各变体实现方案

| 变体 | 修改位置 | 具体改动 |
|---|---|---|
| `no-decomp` | 管线跳过 VMD，直接使用原始序列 | 从 `main.py` 跳过分解步骤 |
| `decomp-{emd,eemd,ceemdan,ssa}` | 替换 VMD 为对应算法 | 新增分解函数接口，替换 `data_VMD.py` |
| `single-graph-{pearson,spearman,dtw}` | 构图时只出一张图 | `adjprocessing.py` 添加模式参数 |
| `no-gcn` | 去掉 GCN 层，GRU 直接处理特征 | `dgcgru.py` 或 `REGCN.py` 中添加纯 GRU 分支 |
| `no-trend-loss` | loss 退化为纯 MSE | `losses.py` 添加 `mse_only` 参数 |

### 3. 结果汇总

运行完成后追加写入 `result/<dataset>/ablation_results.csv`：

```
method,dataset,acc,r2,rmse,mae,mape
Proposed,SSE,0.6940,0.9226,0.1500,0.1137,0.843
Method_1_EMD,SSE,0.6861,0.9290,0.1542,0.1170,0.848
...
```

### 4. 批量运行

支持一键全消融运行：

```bash
python main.py --pipeline ablation-all --dataset SSE
```

自动依次运行所有 10 种变体并汇总结果。

### 5. 注意事项

- 所有变体使用 **相同的超参数**（同 `config.ini`），差异仅限于目标模块
- 各变体的运行次数应当一致（每支股票 1 次）
- 确保 `no-decomp` 变体仍使用相同的数据长度和滑窗参数
