# ADF 检验自动化（Table 1）

## 目标
将论文 Table 1 的 ADF（Augmented Dickey-Fuller）平稳性检验自动化，输出格式化结果。

## 论文参考
Table 1 对比了原始序列与 VMD 分解后各 IMF 分量的 ADF 统计量和 p-value，证明 VMD 有效降低了序列的非平稳性。

## 现有状态
`dataprecossing/ADF.py` 存在，但存在以下问题：
- 硬编码了 `../data/data/50sh.csv` 单文件路径，无法适配 SSE/DJIA 数据集
- 硬编码了 `../data/VMDdata_without precessing/` 目录（该目录可能已不存在或用不同命名）
- 只针对单支股票，未设计为批量运行
- 无格式化输出

## 需求

### 1. 通用 ADF 运行脚本

- 输入：numpy 数据文件路径、VMD 分解结果路径、股票索引
- 对指定股票逐维执行 ADF 检验
- 输出：原始序列和各 IMF 分量的 ADF 统计量、p-value、临界值

### 2. 覆盖场景

- SSE 数据集所有股票
- DJIA 数据集所有股票
- 至少输出原始序列和 IMF1~IMF3 的检验结果（论文表格格式）

### 3. 输出格式

产出 CSV 文件，格式示例：

```
stock,series,adf_stat,p_value,critical_1,critical_5,critical_10
600000,original,-0.8875,0.7921,-3.432,-2.862,-2.567
600000,IMF1,-1.6647,0.4494,-3.432,-2.862,-2.567
600000,IMF2,-7.3048,0.0000,-3.432,-2.862,-2.567
```

### 4. 集成方式

- 注册到 `main.py` 管线，支持 `python main.py --pipeline adf --dataset SSE`
- 结果写入 `result/<dataset>/adf_results.csv`

## 可参考代码
- `dataprecossing/ADF.py`：现有 ADF 调用逻辑可复用
- `statsmodels.tsa.stattools.adfuller`：论文使用的检验方法
