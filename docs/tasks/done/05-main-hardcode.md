# main.py 硬编码超参数

## B9：`run_training()` 超参数未读取 config.ini

**文件：** `main.py` L52-58

**问题：** `run_training()` 函数硬编码了超参数，未从 `config.ini` 读取：

```python
def run_training(dataset, stock_index=None):
    from REGCN.REGCN import main, data
    if stock_index is not None:
        main(data, stock_index,
             lr=8e-4, n_neurons=128, seq_len=30, n_epochs=100)
    else:
        for i in range(data.shape[0]):
            main(data, i, lr=8e-4, n_neurons=128, seq_len=30, n_epochs=100)
```

而 `REGCN/REGCN.py` 内部是从 `config.ini` 读取参数的。两边不一致。

**修复：** 改为从 `config.ini` 读取超参数：

```python
from configparser import ConfigParser
config = ConfigParser()
config.read("REGCN/config.ini")

lr = float(config["hyper"]["lr"])
n_neurons = int(config["hyper"]["n_neurons"])
seq_len = int(config["hyper"]["seq_len"])
n_epochs = int(config["hyper"]["n_epochs"])
```

**验证：** 修改 `config.ini` 中的 `lr` 后，通过 `main.py` 调用的训练应反映新学习率（观察 loss 收敛速度变化）。
