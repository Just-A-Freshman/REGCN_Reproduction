# REGCN.py 运行时崩溃类 bug

## 1. [ medium ] `model.save_weights()` 时 `./model/` 目录未创建

- **文件**: [REGCN.py](../../REGCN/REGCN.py#L122)
- **问题**: `model.save_weights('./model/model_...h5')` 假设 `./model/` 目录已存在，
  但代码中没有 `os.makedirs('./model/', exist_ok=True)` 的创建逻辑。
- **触发条件**: 首次运行，或删除过 `./model/` 目录后。
- **影响**: 训练完成后在保存权重时抛出 `FileNotFoundError` / `OSError`，
  整个股票的训练结果丢失。
- **修复**: 在 `save_weights` 前添加 `os.makedirs('./model/', exist_ok=True)`。

## 2. [ medium ] `plt.show()` 在无头环境（SSH/CI）崩溃

- **文件**: [REGCN.py](../../REGCN/REGCN.py#L203)
- **问题**: `plt.show()` 在没有 GUI 显示器的环境中（SSH、CI 流水线、WSL 无 X 转发的场景）
  会抛出 `tk.TclError: no display name and no $DISPLAY environment variable`。
- **影响**: 训练完成后无法保存结果图，进程崩溃，该股票的所有指标丢失。
- **修复**: 
  - 无头模式下改用 `Agg` 后端：`import matplotlib; matplotlib.use('Agg')`
  - 或 `plt.show()` 前加 try/except 降级为 `plt.close()`
  - 始终在 `savefig` 后调用 `plt.close()` 防止句柄泄漏

## 3. [ medium ] 所有文件路径依赖当前工作目录

- **文件**: [REGCN.py](../../REGCN/REGCN.py#L32-L56)
- **问题**: `config.ini`、数据文件路径、邻接矩阵路径、VMD CSV 路径均使用相对路径，
  解析结果取决于运行脚本时的 `cwd`，而非脚本自身位置。
- **触发条件**: 从项目根目录执行 `python REGCN/REGCN.py`，而非先 `cd REGCN`。
- **影响**: 
  - `config.ini` 找不到 → 配置读取失败
  - 数据路径多一层 `../` 前缀 → 所有文件均无法加载
  - 错误信息只有"FileNotFoundError"等，无提示指向工作目录问题
- **修复**: 用 `os.path.dirname(__file__)` 锚定所有路径：
  ```python
  import os
  BASE = os.path.dirname(os.path.abspath(__file__))
  config_file_addr = os.path.join(BASE, "config.ini")
  ```
