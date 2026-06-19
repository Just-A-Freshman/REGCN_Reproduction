from __future__ import print_function, division
import csv
import pandas as pd
import numpy as np
import glob, os

# ── TF 线程控制（优先用环境变量，默认 intra=16 适合多 worker） ──
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '16')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

import tensorflow as tf

# ── 设备自动检测 ────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'[GPU] Using: {[gpu.name for gpu in gpus]}')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print('[INFO] No GPU detected, using CPU')
# ───────────────────────────────────────────────────────────
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RNN, TimeDistributed
import sys, os
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from input_data import preprocess_data, preprocess_test_only
from utils import get_trend, avg_relative_error, calculate_laplacian
from dgcgru import gcgru
from losses import trend_regularized_loss
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
from configparser import ConfigParser
import matplotlib.pyplot as plt


config_file_addr = os.path.join(_script_dir, "config.ini")
config = ConfigParser()
config.read(config_file_addr)
data_addr = os.path.join(_script_dir, config["hyper"]["data_addr"])
adj_addr = os.path.join(_script_dir, config["hyper"]["adj_addr"])
adj2_addr = config["hyper"]["adj_type2"]
s_index = int(config["hyper"]["s_index"])
lr = float(config["hyper"]["lr"])
n_neurons = int(config["hyper"]["n_neurons"])
seq_len = int(config["hyper"]["seq_len"])
n_epochs = int(config["hyper"]["n_epochs"])
batch_size = int(config["hyper"]["batch_size"])
n_off = int(config["hyper"]["n_off"])
all_data = int(config["hyper"]["all_data"])
start_index = int(config["hyper"]["start_index"])
VMD_addr = os.path.join(_script_dir, config["hyper"]["VMD_addr"])
datasets = config["hyper"]["datasets"]
train_rate = float(config["hyper"].get("train_rate", "0.7"))
val_rate = float(config["hyper"].get("val_rate", "0.1"))
label_col = int(config["hyper"].get("label_col", "3"))
n_jobs = int(config["hyper"].get("n_jobs", "1"))


data_addr = os.path.join(data_addr, datasets + '.npy')
data = np.load(data_addr, allow_pickle=True)

r_mse = float(config["hyper"]["r_mse"])
r_acc = float(config["hyper"]["r_acc"])


def unautoNorm(data, mins, maxs):
    """Denormalize data from [0,1] back to original scale.

    Args:
        data: numpy array of predicted values in [0, 1] range.
        mins, maxs: scalar min/max of the target variable (closing price).
    Returns:
        Data restored to original scale.
    """
    return data * (maxs - mins) + mins


def trainmodel(tdata, tadj, s_index, lr, n_neurons,
               seq_len, n_epochs, j):

    data = tdata.astype(float)
    adj = tadj.astype(float)
    labels = data[:, label_col]
    pre_len = 1
    time_len = data.shape[0]
    n_gcn_nodes = data.shape[1]

    X_train, y_train, X_val, y_val, X_test, y_test, pre_y_test = preprocess_data(
        data, labels, time_len, train_rate, val_rate, seq_len, pre_len)
    y_train = np.expand_dims(y_train, -1)
    y_val = np.expand_dims(y_val, -1)
    p = tf.sparse.to_dense(calculate_laplacian(adj[0]), default_value=0)
    sp = tf.sparse.to_dense(calculate_laplacian(adj[1]), default_value=0)
    DTW = tf.sparse.to_dense(calculate_laplacian(adj[2]), default_value=0)
    Madj = tf.stack([p, sp, DTW], axis=0)

    cell = gcgru(n_neurons, Madj, n_gcn_nodes, 3)

    model = Sequential()
    model.add(RNN(cell, return_sequences=True))
    model.add(Dense(1, kernel_regularizer=tf.keras.regularizers.l1(r_mse)))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=trend_regularized_loss(r_acc)
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
    ]

    model.fit(
        x=X_train,
        y=y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    model_weights_path = 'model/model_' + datasets + '-' + str(s_index) + f'-comp{j}.weights.h5'
    model.save_weights(model_weights_path)
    result = model.predict(X_test, batch_size=batch_size, verbose=0)
    return result


def main(data, s_index, lr, n_neurons,
         seq_len, n_epochs):
    adj_addr1 = adj_addr + datasets + '/' + datasets + '_VMD_' + str(s_index) + adj2_addr
    adj = np.load(adj_addr1, allow_pickle=True)

    tdata = data[s_index]
    tdata = tdata.astype(float)
    labels = tdata[:, label_col]
    pre_len = 1
    time_len = tdata.shape[0]
    y_test, pre_y_test = preprocess_test_only(labels, time_len, train_rate, val_rate, seq_len, pre_len)
    y_test = np.expand_dims(y_test, -1)
    file = sorted(glob.glob(os.path.join("%s%s/%s_*.csv" % (VMD_addr, datasets, s_index))))
    if not file:
        print(f"Warning: no VMD CSV files for {datasets} stock {s_index}, skipping")
        return
    VMD = []
    for f in file:
        VMD.append(pd.read_csv(f, header=None).values)

    result = []
    j = 0
    for ndata in VMD:
        # VMDnor CSVs have min/max appended as last 2 rows
        mdata = ndata[0:time_len]
        close_min = ndata[-2, label_col]
        close_max = ndata[-1, label_col]
        result1 = trainmodel(mdata, adj[j], s_index, lr, n_neurons,
                             seq_len, n_epochs, j)
        # Denormalize each component's prediction from [0,1] to original price scale
        result1 = unautoNorm(result1, close_min, close_max)
        j += 1
        result.append(result1)

    if not result:
        print(f"Warning: no predictions for stock {s_index}, skipping")
        return

    # Align test window counts across VMD components before summing
    min_windows = min(r.shape[0] for r in result)
    aligned = [r[:min_windows] for r in result]
    result = np.sum(aligned, axis=0)
    result = result[:, -1, :]
    y_test = y_test[:, -1, :]

    actual_trend = get_trend(pre_y_test, y_test)
    predicted_trend = get_trend(pre_y_test, result)
    accuracy = accuracy_score(actual_trend, predicted_trend)

    print("***********************")
    print(j)
    print("accuracy: ", accuracy)
    r2 = r2_score(y_test, result)
    print("r2: ", r2)
    rmse = sqrt(mean_squared_error(y_test, result))
    print("rmse: ", rmse)
    mae = mean_absolute_error(y_test, result)
    print("mae: ", mae)
    re = avg_relative_error(y_test, result)
    print("re: ", re)
    print(f"pred_range: [{result.min():.4f}, {result.max():.4f}]  std={result.std():.4f}")
    print(f"actual_range: [{y_test.min():.4f}, {y_test.max():.4f}]")
    write_data = ["REGCN_" + str(seq_len), str(s_index), str(accuracy), str(r2), str(rmse), str(mae), str(re), str(r_mse), str(r_acc)]
    _result_dir = os.path.join('result', 'Table5', datasets)
    os.makedirs(_result_dir, exist_ok=True)
    stock_csv = os.path.join(_result_dir, f'result_REGCN_stock{s_index}.csv')
    with open(stock_csv, 'w', newline='', encoding='UTF8') as f:
        d = csv.writer(f)
        d.writerow(write_data)

    plt.figure()
    plt.plot(y_test, color='red', label='Real Stock Price')
    plt.plot(result, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    _fig_dir = 'result/Fig3'
    os.makedirs(_fig_dir, exist_ok=True)
    plt.savefig(os.path.join(_fig_dir, 'REGCN_' + datasets + '-' + str(s_index) + '.png'), dpi=200)
    if all_data != 1:
        plt.show()
    plt.close()



def _merge_results():
    import glob as _gl
    _rd = os.path.join('result', 'Table5', datasets)
    parts = sorted(_gl.glob(os.path.join(_rd, f'result_REGCN_stock*.csv')))
    if not parts:
        return
    with open(os.path.join(_rd, 'result_REGCN.csv'), 'w', newline='', encoding='UTF8') as out:
        w = csv.writer(out)
        for pf in parts:
            with open(pf, 'r') as f:
                for row in csv.reader(f):
                    w.writerow(row)
            os.remove(pf)
    print(f'[Merge] {len(parts)} stock results -> result_REGCN.csv')


def _main_serial(data, start_idx, lr, n_neurons, seq_len, n_epochs):
    if n_jobs > 1:
        from multiprocessing import get_context
        stocks = list(range(start_idx, data.shape[0]))
        pool_sz = min(n_jobs, len(stocks))
        print(f'[Parallel] Training {len(stocks)} stocks with {pool_sz} workers')
        with get_context('spawn').Pool(pool_sz) as pool:
            pool.starmap(main, [(data, i, lr, n_neurons, seq_len, n_epochs)
                                for i in stocks])
        _merge_results()
    else:
        for s_index in range(start_idx, data.shape[0]):
            main(data, s_index, lr, n_neurons, seq_len, n_epochs)


if __name__ == '__main__':
    if all_data == 1:
        _main_serial(data, start_index, lr, n_neurons, seq_len, n_epochs)
    else:
        main(data, s_index, lr, n_neurons, seq_len, n_epochs)
