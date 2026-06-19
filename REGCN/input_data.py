import numpy as np


def preprocess_data(data, labels, time_len, train_rate, val_rate, seq_len, pre_len):
    """Split time series into train / val / test with sliding windows.

    Ratios from paper: train=0.7, val=0.1, test=0.2.
    """
    train_size = int(time_len * train_rate)
    val_size = int(time_len * (train_rate + val_rate))

    train_data = data[0:train_size]
    val_data = data[train_size:val_size]
    test_data = data[val_size:time_len]

    train_label = labels[0:train_size]
    val_label = labels[train_size:val_size]
    test_label = labels[val_size:time_len]
    pre_test_label = labels[val_size - 1:time_len - 1]

    trainX, trainY = [], []
    for i in range(len(train_data) - seq_len - pre_len + 1):
        a = train_data[i: i + seq_len + pre_len]
        b = train_label[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(b[pre_len: seq_len + pre_len])

    valX, valY = [], []
    for i in range(len(val_data) - seq_len - pre_len + 1):
        a = val_data[i: i + seq_len + pre_len]
        b = val_label[i: i + seq_len + pre_len]
        valX.append(a[0: seq_len])
        valY.append(b[pre_len: seq_len + pre_len])

    testX, testY, pre_testY = [], [], []
    for i in range(len(test_data) - seq_len - pre_len + 1):
        a = test_data[i: i + seq_len + pre_len]
        b = test_label[i: i + seq_len + pre_len]
        c = pre_test_label[i: i + seq_len + pre_len]
        testX.append(a[0: seq_len])
        testY.append(b[pre_len: seq_len + pre_len])
        pre_testY.append(c[seq_len: seq_len + pre_len])

    return (np.array(trainX), np.array(trainY),
            np.array(valX), np.array(valY),
            np.array(testX), np.array(testY),
            np.array(pre_testY))


def per_window_normalize(X, y=None):
    """Per-window min-max normalization to [0, 1].

    Each window (sample) is normalized independently using its own min/max,
    ensuring the model always sees values in [0, 1] regardless of global
    distribution shift (OOD).

    Args:
        X: (n_windows, seq_len, n_feat) — globally-normalized VMD component.
        y: (n_windows, seq_len, 1) or None — closing price target.

    Returns:
        X_norm, y_norm (or None), win_mins, win_maxs
    """
    n = X.shape[0]
    win_mins = np.zeros(n)
    win_maxs = np.zeros(n)
    X_norm = np.zeros_like(X, dtype=np.float64)

    for i in range(n):
        lo = X[i].min()
        hi = X[i].max()
        win_mins[i] = lo
        win_maxs[i] = hi
        if hi > lo:
            X_norm[i] = (X[i] - lo) / (hi - lo)

    if y is not None:
        y_norm = np.zeros_like(y, dtype=np.float64)
        for i in range(n):
            lo, hi = win_mins[i], win_maxs[i]
            if hi > lo:
                y_norm[i] = (y[i] - lo) / (hi - lo)
        return X_norm, y_norm, win_mins, win_maxs

    return X_norm, None, win_mins, win_maxs


def per_window_denormalize(pred, win_mins, win_maxs):
    """Reverse per-window normalization — back to global VMD-normalised scale.

    Args:
        pred: (n_windows, seq_len, 1)
        win_mins, win_maxs: (n_windows,)

    Returns:
        Denormalised predictions in the global [0,1] VMD-component scale.
    """
    denom = win_maxs - win_mins
    denom[denom == 0] = 1.0
    return pred * denom[:, np.newaxis, np.newaxis] + win_mins[:, np.newaxis, np.newaxis]


def preprocess_test_only(labels, time_len, train_rate, val_rate, seq_len, pre_len):
    """Build test/pre-test label windows (no X data). Used in main() for evaluation."""
    val_size = int(time_len * (train_rate + val_rate))
    test_label = labels[val_size:time_len]
    pre_test_label = labels[val_size - 1:time_len - 1]

    testY, pre_testY = [], []
    for i in range(len(test_label) - seq_len - pre_len + 1):
        b = test_label[i: i + seq_len + pre_len]
        c = pre_test_label[i: i + seq_len + pre_len]
        testY.append(b[pre_len: seq_len + pre_len])
        pre_testY.append(c[seq_len: seq_len + pre_len])
    return np.array(testY), np.array(pre_testY)
