# normalization.py 未接入训练流水线

## 背景

经过代码审计确认，项目中的归一化步骤 (`dataprecossing/normalization.py`) **没有被集成到 `main.py` 的自动流水线中**。

## 问题

### 1. main.py pipeline 缺失 normalization 步骤

`main.py` 的 pipeline 为：
```
data → ga-vmd → vmd → adj → train
```
没有 normalization。`normalization.py` 从未被导入或调用，只能作为独立脚本手动运行。

### 2. config.ini 已指向 VMDnor/ 但从未生成

`REGCN/config.ini` 中 `VMD_addr = ../data/data/VMDdata/` 是正确的（adjprocessing.py 需要原始量纲的 VMD 数据），但 `REGCN/REGCN.py` 中使用了独立的 `VMD_addr` 读取。

当前磁盘上的 `VMDdata/` 文件是旧版代码生成的（已归一化 + 附加 min/max 行），和当前 `data_VMD.py` 的输出不一致。

### 3. 从零运行 pipeline 会崩溃

执行 `python main.py --pipeline all` 时：
- `data_VMD.py` 生成非归一化的 VMD 分量（原始价格量级）
- `REGCN.py` 尝试读取 `VMDnor/`，但该目录不存在或内容陈旧
- 训练步骤因 `FileNotFoundError` 崩溃

## 修复方案

### 方案 A：将 normalization 加入 pipeline（推荐）

在 `main.py` 的 pipeline 中 `vmd` 步骤之后添加 `normalize` 步骤：

```python
def run_normalize(dataset):
    from dataprecossing.normalization import run
    run(dataset)

# pipeline 改为:
data → ga-vmd → vmd → normalize → adj → train
```

同时确保 `REGCN/REGCN.py` 中读取的 VMD 路径指向 `VMDnor/`（已配好）或使用归一化后的数据。

### 方案 B：将归一化移回 data_VMD.py

如果 `normalization.py` 的功能简单，可以直接合并回 `data_VMD.py`，减少一个 pipeline 步骤。

## 前置修复

修 `normalization.py` 中的两个 bug（见 `28-normalization-pipeline-fixes.md`）：
- `pd.read_csv()` 缺少 `header=None`
- 常数列除零保护
