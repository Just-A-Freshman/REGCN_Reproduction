import glob, os
import numpy as np
from fastdtw import fastdtw
import pandas as pd
from scipy.spatial.distance import euclidean

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..'))


def run(dataset, train_rate=0.7):
    """Construct adjacency graphs (Pearson / Spearman / DTW)."""
    threshold = 0.9
    data_addr = os.path.join(_project_root, 'data/data/')
    data = np.load(os.path.join(data_addr, dataset + '.npy'), allow_pickle=True)
    for j in range(data.shape[0]):
        adj = []
        file = sorted(glob.glob(os.path.join(
            _project_root, "data/data/VMDdata/%s/%s*.csv" % (dataset, j))))
        for f in file:
            tdata = pd.read_csv(f, header=None).values
            train_size = int(tdata.shape[0] * train_rate)
            train_data = tdata[:train_size]
            pf = pd.DataFrame(train_data)
            na = pf.corr(method='spearman')
            nb = pf.corr(method='pearson')
            na = np.array(na)
            row, col = np.diag_indices_from(na)
            na[row, col] = 0
            nb = np.array(nb)
            row, col = np.diag_indices_from(nb)
            nb[row, col] = 0
            ndtw_data = np.zeros(shape=(tdata.shape[1], tdata.shape[1]))
            for m in range(tdata.shape[1]):
                for n in range(m + 1, tdata.shape[1]):
                    p, _ = fastdtw(train_data[:, m].reshape(-1, 1),
                                   train_data[:, n].reshape(-1, 1), dist=euclidean)
                    d = 1 - p / (10 * train_data.shape[0])
                    ndtw_data[m][n] = d
                    ndtw_data[n][m] = d
            adj.append([nb, na, ndtw_data])
        base = os.path.join(_project_root, 'data/adj/', dataset, '')
        os.makedirs(base, exist_ok=True)
        adj = np.array(adj)
        np.save(base + dataset + '_VMD_' + str(j) + '.npy', adj)
        adj[np.abs(adj) < threshold] = 0
        np.save(base + dataset + '_VMD_' + str(j) + '_90.npy', adj)


if __name__ == "__main__":
    datasets = "SSE"
    run(datasets)
