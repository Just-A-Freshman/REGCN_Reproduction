import numpy as np
from vmdpy import VMD
import pandas as pd
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..'))


def run(dataset):
    """Apply VMD decomposition for all stocks in dataset."""
    data_addr = os.path.join(_project_root, 'data/data/', dataset + '.npy')
    canshu = pd.read_csv(os.path.join(_project_root, 'result/Table4/', dataset + '_GA.csv'), header=None).values
    canshu = canshu.astype(int)
    tdata = np.load(data_addr)
    print(tdata.shape)
    tau = 0.
    DC = 0
    init = 1
    tol = 1e-7
    train_rate = 0.7
    val_rate = 0.1

    for j in range(tdata.shape[0]):
        data = tdata[j]
        K = canshu[j][0]
        alpha = canshu[j][1]

        row = data.shape[0]
        train_size = int(row * train_rate)
        val_size = int(row * (train_rate + val_rate))
        u2 = []
        for i in range(data.shape[1]):
            # Decompose each split separately to prevent test data leakage
            train_u, _, _ = VMD(data[:train_size, i], alpha, tau, K, DC, init, tol)
            val_u, _, _ = VMD(data[train_size:val_size, i], alpha, tau, K, DC, init, tol)
            test_u, _, _ = VMD(data[val_size:row, i], alpha, tau, K, DC, init, tol)
            # train_u shape: (K, train_size); concatenate along time axis
            u_all = np.concatenate([train_u, val_u, test_u], axis=1)
            # u_all shape: (K, total_time); transpose to (total_time, K)
            u2.append(list(map(list, zip(*u_all))))

        # u2: list of n_features arrays, each (total_time, K)
        # Rearrange to (K, total_time, n_features)
        u2 = np.array(list(map(list, zip(*u2))))
        out_dir = os.path.join(_project_root, "data/data/VMDdata/", dataset, "")
        os.makedirs(out_dir, exist_ok=True)
        for i in range(K):
            np.savetxt(
                out_dir + str(j) + "_" + str(K) + "-" + str(i + 1) + ".csv",
                u2[:, :, i], delimiter=",")
    print("Finish\n")


if __name__ == "__main__":
    datasets = "SSE"
    run(datasets)



