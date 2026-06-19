"""
Generate Fig 3 prediction plots from saved model weights.
Usage: python fig3_prediction_plots.py [--datasets SSE] [--start 0] [--end 40]
"""
import os, sys, glob, csv
import numpy as np
import pandas as pd
import tensorflow as tf
from math import sqrt
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── TF threading ──
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
REGCN_DIR = os.path.join(BASE, 'REGCN')
sys.path.insert(0, BASE)
sys.path.insert(0, REGCN_DIR)

from input_data import preprocess_data, preprocess_test_only
from losses import trend_regularized_loss
from utils import calculate_laplacian, get_trend, avg_relative_error

# unautoNorm is defined in REGCN.py; copy here to avoid circular import
def unautoNorm(data, mins, maxs):
    return data * (maxs - mins) + mins
from dgcgru import gcgru

# ── Config ──
import configparser
config = configparser.ConfigParser()
config.read(os.path.join(REGCN_DIR, 'config.ini'))

SEQ_LEN = int(config['hyper'].get('seq_len', '30'))
R_MSE = float(config['hyper'].get('r_mse', '0.0'))
R_ACC = float(config['hyper'].get('r_acc', '0.1'))
TRAIN_RATE = float(config['hyper'].get('train_rate', '0.7'))
VAL_RATE = float(config['hyper'].get('val_rate', '0.1'))
LABEL_COL = int(config['hyper'].get('label_col', '3'))
LR = float(config['hyper'].get('lr', '8e-3'))
N_NEURONS = int(config['hyper'].get('n_neurons', '128'))
N_EPOCHS = int(config['hyper'].get('n_epochs', '100'))
BATCH_SIZE = int(config['hyper'].get('batch_size', '128'))
ADJ_ADDR = os.path.join(REGCN_DIR, config['hyper']['adj_addr'])
VMD_ADDR = os.path.join(REGCN_DIR, config['hyper']['vmd_addr'])
ADJ_TYPE2 = config['hyper']['adj_type2']
DATA_ADDR = os.path.join(REGCN_DIR, config['hyper']['data_addr'])

MODEL_DIR = r'C:\temp\stock_models'
FIG_DIR = os.path.join(BASE, 'result', 'Fig3')
FIG_DIR = os.path.join(BASE, 'result', 'Fig3')
os.makedirs(FIG_DIR, exist_ok=True)

def predict_with_weights(tdata, adj, s_index, dataset, j):
    """Build model, load weights, predict (no training)"""
    data_arr = tdata.astype(float)
    adj_arr = adj.astype(float)
    labels = data_arr[:, LABEL_COL]
    time_len = data_arr.shape[0]
    n_gcn_nodes = data_arr.shape[1]

    X_train, y_train, X_val, y_val, X_test, y_test, pre_y_test = preprocess_data(
        data_arr, labels, time_len, TRAIN_RATE, VAL_RATE, SEQ_LEN, 1)
    y_train = np.expand_dims(y_train, -1)
    y_val = np.expand_dims(y_val, -1)

    p = tf.sparse.to_dense(calculate_laplacian(adj_arr[0]), default_value=0)
    sp = tf.sparse.to_dense(calculate_laplacian(adj_arr[1]), default_value=0)
    dtw = tf.sparse.to_dense(calculate_laplacian(adj_arr[2]), default_value=0)
    madj = tf.stack([p, sp, dtw], axis=0)

    cell = gcgru(N_NEURONS, madj, n_gcn_nodes, 3)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.RNN(cell, return_sequences=True))
    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(R_MSE)))

    # Build model properly by fitting one batch
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss=trend_regularized_loss(R_ACC))
    model.fit(X_train[:2], y_train[:2], epochs=1, batch_size=2, verbose=0)

    # Load saved weights (overwrites the 1-epoch training)
    weight_path = os.path.join(MODEL_DIR, f'model_{dataset}-{s_index}-comp{j}.weights.h5')
    if not os.path.exists(weight_path):
        print(f"  WARNING: weights not found: {weight_path}")
        return None
    model.load_weights(weight_path)

    result = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    return result

def gen_figure(dataset, s_index):
    """Generate prediction plot for one stock using saved weights"""
    data_path = os.path.join(DATA_ADDR, f'{dataset}.npy')
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return False

    data = np.load(data_path, allow_pickle=True)
    if s_index >= data.shape[0]:
        print(f"Stock {s_index} out of range (max {data.shape[0]-1})")
        return False

    adj_path = os.path.join(ADJ_ADDR, dataset, f'{dataset}_VMD_{s_index}{ADJ_TYPE2}')
    if not os.path.exists(adj_path):
        print(f"Adj file not found: {adj_path}")
        return False
    adj = np.load(adj_path, allow_pickle=True)

    tdata = data[s_index]
    labels = tdata[:, LABEL_COL]
    time_len = tdata.shape[0]
    pre_len = 1

    y_test, pre_y_test = preprocess_test_only(labels, time_len, TRAIN_RATE, VAL_RATE, SEQ_LEN, pre_len)

    # Load VMD CSV files
    vmd_pattern = os.path.join(VMD_ADDR, dataset, f'{s_index}_*.csv')
    files = sorted(glob.glob(vmd_pattern))
    if not files:
        print(f"  No VMD files for {dataset} stock {s_index}")
        return False

    vmd_data = []
    for f in files:
        vmd_data.append(pd.read_csv(f, header=None).values)

    predictions = []
    j = 0
    for ndata in vmd_data:
        mdata = ndata[0:time_len]
        close_min = ndata[-2, LABEL_COL]
        close_max = ndata[-1, LABEL_COL]
        result = predict_with_weights(mdata, adj[j], s_index, dataset, j)
        if result is None:
            return False
        result = unautoNorm(result, close_min, close_max)
        predictions.append(result)
        j += 1

    if not predictions:
        return False

    # Align and sum components
    min_windows = min(r.shape[0] for r in predictions)
    aligned = [r[:min_windows] for r in predictions]
    result = np.sum(aligned, axis=0)
    result = result[:, -1, :]
    y_test_plot = np.expand_dims(y_test, -1)[:min_windows, -1, :]

    # Metrics
    r2 = r2_score(y_test_plot, result)
    rmse = sqrt(mean_squared_error(y_test_plot, result))
    actual_trend = get_trend(pre_y_test, y_test_plot)
    predicted_trend = get_trend(pre_y_test, result)
    acc = accuracy_score(actual_trend, predicted_trend)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_plot, color='red', label='Real')
    plt.plot(result, color='blue', label='Predicted')
    plt.title(f'{dataset} Stock {s_index}  (R2={r2:.4f}, ACC={acc:.4f})')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(alpha=0.3)
    fig_path = os.path.join(FIG_DIR, f'REGCN_{dataset}-{s_index}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [{dataset}] Stock {s_index}: R2={r2:.4f}, ACC={acc:.4f} -> {fig_path}")
    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['SSE'])
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    for ds in args.datasets:
        data_path = os.path.join(DATA_ADDR, f'{ds}.npy')
        data = np.load(data_path, allow_pickle=True)
        total = data.shape[0]
        end = args.end if args.end is not None else total
        print(f"\n{'='*50}")
        print(f"{ds}: stocks {args.start} to {end-1} ({end-args.start}/{total})")
        print(f"{'='*50}")
        for idx in range(args.start, end):
            gen_figure(ds, idx)
    print("\nAll done!")
