# 预测效果可视化增强（Fig. 3）

## 目标
将当前单条预测曲线图升级为论文 Fig. 3 风格的基线对比图。

## 论文参考
Fig. 3 展示本方法与若干基线（至少 GRU、GCGRU、VGC-GAN）在同一时间段的预测曲线叠加对比，直观展示不同模型的拟合精度差异。

## 现有状态
`REGCN.py` 已产出 `REGCN_<dataset>-<index>.png`，包含：
- 红色折线：真实收盘价
- 蓝色折线：REGCN 预测值

但**缺少基线对比**，无法直观展示 REGCN 相对于其他方法的优势。

## 需求

### 1. 预测值存盘

基线对比的前提：需要保存各基线的预测值。

在 `baseline-comparison.md` 实现后，所有模型预测值（反归一化后的原始尺度）保存到：

```
result/<dataset>/predictions/<stock_index>_<method>.npy
```

每文件 shape `(n_windows, 1)`，与 REGCN 的预测输出格式一致。

### 2. 对比图生成脚本

```bash
python main.py --pipeline plot-compare --dataset SSE --stock 0
```

产出 `result/SSE/compare_stock_0.png`，风格要求：

- 横轴：时间（交易日序号）
- 纵轴：收盘价（原始价格尺度）
- 黑色实线：真实值
- 红色线：REGCN（本方法）
- 2~3 条灰色/蓝色虚线：选择的基线（默认取 GRU、GCGRU、VGC-GAN）
- 图例标注各线条对应的方法
- 标题包含股票代码和数据集名称

### 3. 批量对比

```bash
python main.py --pipeline plot-compare-all --dataset SSE
```

对数据集中每支股票各产出一张对比图，存入 `result/SSE/compare/` 目录。

### 4. 对比方法选择

默认选取论文 Table 5 中排名前 3 的基线（按 ACC 排序）与本方法对比。

可通过 `--methods` 参数自定义：

```bash
python main.py --pipeline plot-compare --dataset SSE --stock 0 --methods GRU,GCGRU,VGC-GAN
```

## 注意事项
- 所有方法需在相同的测试集时间段上绘制，确保时间轴对齐
- 若未运行基线对比，应提示"请先运行 baseline 模式"而非静默失败
- 避免一张图上线条过多（≤ 5 条），防止视觉混乱
