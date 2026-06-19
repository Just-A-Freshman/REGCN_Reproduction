# 训练流程完善：验证集 + 统一入口

## B5：添加验证集划分

**问题：** 代码使用 8:2 划分训练/测试，论文使用 7:1:2 训练/验证/测试。

**涉及文件：**
- `REGCN/input_data.py` `preprocess_data()` — 需新增 validation 分支
- `REGCN/REGCN.py` `trainmodel()` — 需传入 `validation_data`
- VMD 和图构建需对验证集单独处理（同测试集流程）

**改动：**
1. 划分比例从 8:2 改为 7:1:2
2. VMD 分解对验证集单独调用（同参数 K/α）
3. 图构建基于训练集计算，验证集和测试集复用
4. 训练时添加 `validation_data` 参数
5. 可考虑添加 `EarlyStopping(monitor='val_loss')` 回调

**验证：** 打印各集合样本数，确认符合 7:1:2（例如 DJIA：526 训练 / 75 验证 / 151 测试）。

---

## F2：统一入口与配置

**问题：** 预处理（VMD、图构建）和模型训练分散在独立脚本中，需要手动依次执行。所有路径硬编码。

**涉及文件：**
- 预处理脚本：`dataprecossing/data.py`、`data_VMD.py`、`GA_VMD.py`、`adjprocessing.py`
- 训练脚本：`REGCN/REGCN.py`
- 核心配置：已部分迁移到 `REGCN/config.ini`

**建议改动（可选）：**
1. 在 `REGCN/config.ini` 中补全所有路径和参数
2. 新增 `main.py` 串联全流程（数据预处理 → VMD → 图构建 → 训练 → 评估）
3. 支持命令行参数：`python main.py --dataset DJIA --train`

**注意：** 此项为工程优化，不修复不会影响模型训练本身。优先级低于其他修复项。
