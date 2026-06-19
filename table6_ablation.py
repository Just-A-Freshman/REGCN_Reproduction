"""
REGCN Table 6 Ablation Study — external runner, no original code modified.
Usage:
  python table6_ablation.py --method 5 --datasets SSE --start 0 --end 1  (local test)
  python table6_ablation.py --method 6 --datasets SSE DJIA              (full run)
"""
import os, sys, glob, csv, time, argparse, shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from math import sqrt
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error

os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'REGCN'))

import configparser
config = configparser.ConfigParser()
config.read(os.path.join(BASE, 'REGCN', 'config.ini'))

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

from input_data import preprocess_data, preprocess_test_only
from utils import calculate_laplacian, get_trend, avg_relative_error
from losses import trend_regularized_loss
from dgcgru import gcgru

def unautoNorm(data, mins, maxs):
    return data * (maxs - mins) + mins

# ─── Shared training function (non-invasive, copied for Methods 9, 10) ───

def _trainmodel_vanilla(tdata, tadj, s_index, lr, n_neurons, seq_len, n_epochs, j):
    """Original trainmodel from REGCN.py — unmodified logic."""
    data_arr = tdata.astype(float)
    adj_arr = tadj.astype(float)
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
    cell = gcgru(n_neurons, madj, n_gcn_nodes, 3)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.RNN(cell, return_sequences=True))
    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(R_MSE)))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=trend_regularized_loss(R_ACC))
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=BATCH_SIZE, verbose=0,
              validation_data=(X_val, y_val),
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    result = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    return result

def _trainmodel_mse(tdata, tadj, s_index, lr, n_neurons, seq_len, n_epochs, j):
    """Method 10: MSE loss instead of trend_regularized_loss."""
    data_arr = tdata.astype(float)
    adj_arr = tadj.astype(float)
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
    cell = gcgru(n_neurons, madj, n_gcn_nodes, 3)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.RNN(cell, return_sequences=True))
    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(R_MSE)))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse')  # ← Method 10: plain MSE
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=BATCH_SIZE, verbose=0,
              validation_data=(X_val, y_val),
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    result = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    return result

def _trainmodel_nogcn(tdata, tadj, s_index, lr, n_neurons, seq_len, n_epochs, j):
    """Method 9: No GCN — replace gcgru cell with plain GRUCell."""
    data_arr = tdata.astype(float)
    labels = data_arr[:, LABEL_COL]
    time_len = data_arr.shape[0]
    X_train, y_train, X_val, y_val, X_test, y_test, pre_y_test = preprocess_data(
        data_arr, labels, time_len, TRAIN_RATE, VAL_RATE, SEQ_LEN, 1)
    y_train = np.expand_dims(y_train, -1)
    y_val = np.expand_dims(y_val, -1)
    # Plain GRU — no GCN, no adjacency
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(n_neurons, return_sequences=True))
    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(R_MSE)))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=trend_regularized_loss(R_ACC))
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=BATCH_SIZE, verbose=0,
              validation_data=(X_val, y_val),
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    result = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    return result

# ─── Method data preparation ───

def _load_vmd_data(dataset, s_index):
    """Load VMD components for a stock. Returns list of (mdata, close_min, close_max)."""
    data_path = os.path.join(BASE, 'data', 'data', '%s.npy' % dataset)
    data = np.load(data_path, allow_pickle=True)
    tdata = data[s_index]
    time_len = tdata.shape[0]
    vmd_pattern = os.path.join(BASE, 'data', 'data', 'VMDdata', dataset, '%d_*.csv' % s_index)
    files = sorted(glob.glob(vmd_pattern))
    if not files:
        return None, None
    components = []
    for f in files:
        ndata = pd.read_csv(f, header=None).values
        mdata = ndata[0:time_len]
        cmin = ndata[-2, LABEL_COL]
        cmax = ndata[-1, LABEL_COL]
        components.append((mdata, cmin, cmax))
    return components, data

def _load_adj(dataset, s_index):
    """Load adjacency matrices for a stock."""
    adj_path = os.path.join(BASE, 'data', 'adj', dataset,
                            '%s_VMD_%d_90.npy' % (dataset, s_index))
    return np.load(adj_path, allow_pickle=True)

# ─── Main runner ───

METHOD_NAMES = {
    5: 'method5_nodecomp', 6: 'method6_pearson', 7: 'method7_spearman',
    8: 'method8_dtw', 9: 'method9_nogcn', 10: 'method10_mse'
}

def run_method(method, dataset, s_index):
    """Run one stock for one ablation method. Returns metrics dict or None."""
    components, data = _load_vmd_data(dataset, s_index)
    if components is None:
        print("  No VMD data for stock %d" % s_index)
        return None

    # Method-specific preparation
    if method == 5:
        # ── No decomposition: skip VMD, use raw normalized data ──
        tdata = data[s_index]
        adj = _load_adj(dataset, s_index)
        # Normalize raw data to [0, 1]
        data_min = tdata.min(axis=0, keepdims=True)
        data_max = tdata.max(axis=0, keepdims=True)
        denom = data_max - data_min
        denom[denom == 0] = 1.0
        mdata = (tdata - data_min) / denom
        cmin = float(tdata[:, LABEL_COL].min())
        cmax = float(tdata[:, LABEL_COL].max())
        # Single component: use adj[0]
        result_raw = _trainmodel_vanilla(mdata, adj[0], s_index, LR, N_NEURONS, SEQ_LEN, N_EPOCHS, 0)
        result = unautoNorm(result_raw, cmin, cmax)
        result_arr = [result]
    elif method in (6, 7, 8):
        # ── Single graph type ──
        adj_all = _load_adj(dataset, s_index)
        graph_idx = {6: 0, 7: 1, 8: 2}[method]
        result_arr = []
        for j, (mdata, cmin, cmax) in enumerate(components):
            g = adj_all[j][graph_idx]  # single graph matrix
            single_adj = np.stack([g, g, g], axis=0)  # repeat 3x
            result_raw = _trainmodel_vanilla(mdata, single_adj, s_index, LR, N_NEURONS, SEQ_LEN, N_EPOCHS, j)
            result_arr.append(unautoNorm(result_raw, cmin, cmax))
    elif method == 9:
        # ── No GCN ──
        adj_all = _load_adj(dataset, s_index)
        result_arr = []
        for j, (mdata, cmin, cmax) in enumerate(components):
            # Use vanilla adj (not needed for no-GCN, but pass for interface compat)
            result_raw = _trainmodel_nogcn(mdata, adj_all[j], s_index, LR, N_NEURONS, SEQ_LEN, N_EPOCHS, j)
            result_arr.append(unautoNorm(result_raw, cmin, cmax))
    elif method == 10:
        # ── MSE loss ──
        adj_all = _load_adj(dataset, s_index)
        result_arr = []
        for j, (mdata, cmin, cmax) in enumerate(components):
            result_raw = _trainmodel_mse(mdata, adj_all[j], s_index, LR, N_NEURONS, SEQ_LEN, N_EPOCHS, j)
            result_arr.append(unautoNorm(result_raw, cmin, cmax))
    else:
        raise ValueError("Method %d not yet implemented" % method)

    # ── Common evaluation ──
    tdata = data[s_index]
    labels = tdata[:, LABEL_COL]
    time_len = tdata.shape[0]
    y_test, pre_y_test = preprocess_test_only(labels, time_len, TRAIN_RATE, VAL_RATE, SEQ_LEN, 1)

    min_windows = min(r.shape[0] for r in result_arr)
    aligned = [r[:min_windows] for r in result_arr]
    pred = np.sum(aligned, axis=0)[:, -1, :]
    yt = np.expand_dims(y_test, -1)[:min_windows, -1, :]

    acc = accuracy_score(get_trend(pre_y_test[:min_windows], yt), get_trend(pre_y_test[:min_windows], pred))
    r2 = r2_score(yt, pred)
    rmse = sqrt(mean_squared_error(yt, pred))
    mae = mean_absolute_error(yt, pred)
    mape = avg_relative_error(yt, pred)

    return {'s_index': s_index, 'ACC': acc, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
            'r_mse': R_MSE, 'r_acc': R_ACC, 'seq_len': SEQ_LEN}

def save_metrics(method, dataset, metrics):
    """Save metrics to result/Table6/{method_name}/{dataset}_{method}.csv"""
    method_name = METHOD_NAMES[method]
    out_dir = os.path.join(BASE, 'result', 'Table6', method_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, '%s_%s.csv' % (dataset, method_name))
    header = False if os.path.exists(out_file) else True
    with open(out_file, 'a', newline='') as f:
        w = csv.writer(f)
        if header:
            w.writerow(['Method', 'Stock', 'ACC', 'R2', 'RMSE', 'MAE', 'MAPE', 'r_mse', 'r_acc'])
        w.writerow([metrics['seq_len'], metrics['s_index'], metrics['ACC'], metrics['R2'],
                    metrics['RMSE'], metrics['MAE'], metrics['MAPE'],
                    metrics['r_mse'], metrics['r_acc']])

def _run_one_stock(args):
    """Module-level wrapper for multiprocessing. Args: (method, dataset, idx)."""
    method, dataset, idx = args
    t0 = time.time()
    metrics = run_method(method, dataset, idx)
    if metrics:
        save_metrics(method, dataset, metrics)
        dt = time.time() - t0
        return "  [%s] Stock %d: R2=%.4f ACC=%.4f (%.0fs)" % (dataset, idx, metrics['R2'], metrics['ACC'], dt)
    return "  [%s] Stock %d: SKIP" % (dataset, idx)

def summarize(method, dataset):
    """Print summary of results."""
    method_name = METHOD_NAMES[method]
    fn = os.path.join(BASE, 'result', 'Table6', method_name, '%s_%s.csv' % (dataset, method_name))
    if not os.path.exists(fn):
        print("  No results found")
        return
    with open(fn) as f:
        reader = csv.reader(f)
        header = next(reader)
        r2s = [float(r[3]) for r in reader if r and r[0].startswith(str(SEQ_LEN))]
    if not r2s:
        return
    import statistics
    print("  %s %s: %d stocks, R2 mean=%.4f, med=%.4f, pos=%d/%d" % (
        method_name, dataset, len(r2s), sum(r2s)/len(r2s),
        sorted(r2s)[len(r2s)//2], sum(1 for r in r2s if r > 0), len(r2s)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=int, required=True, choices=[5, 6, 7, 8, 9, 10])
    parser.add_argument('--datasets', nargs='+', default=['SSE'])
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers (default=1 serial). Use 16 for cloud.')
    args = parser.parse_args()

    method_name = METHOD_NAMES[args.method]
    print("=" * 50, flush=True)
    print("Ablation Method %d: %s (workers=%d)" % (args.method, method_name, args.workers), flush=True)
    print("=" * 50, flush=True)

    for ds in args.datasets:
        data_path = os.path.join(BASE, 'data', 'data', '%s.npy' % ds)
        data = np.load(data_path, allow_pickle=True)
        total = data.shape[0]
        end = args.end if args.end is not None else total
        indices = list(range(args.start, end))
        print("\n%s: stocks %d-%d (%d/%d)" % (ds, args.start, end-1, end-args.start, total), flush=True)

        if args.workers > 1:
            from multiprocessing import get_context
            pool_sz = min(args.workers, len(indices))
            print("  Parallel: %d workers for %d stocks (fork)" % (pool_sz, len(indices)), flush=True)
            work = [(args.method, ds, idx) for idx in indices]
            ctx = get_context('fork') if os.name != 'nt' else get_context('spawn')
            with ctx.Pool(pool_sz) as pool:
                for msg in pool.imap_unordered(_run_one_stock, work):
                    print(msg, flush=True)
        else:
            for idx in indices:
                print(_run_one_stock((args.method, ds, idx)), flush=True)

        summarize(args.method, ds)

    print("\nDone! Results in result/Table6/%s/" % method_name, flush=True)
