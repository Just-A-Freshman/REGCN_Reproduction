import glob
import numpy as np
import pandas as pd
import os


def run(dataset):
    """Read raw CSV files and save as numpy array."""
    dtw_addr = '../data/data/raw data/' + dataset + '/'
    file = glob.glob(os.path.join('%s*.csv') % (dtw_addr))
    result = []
    n = 0
    for f in file:
        filename, extension = os.path.splitext(f)
        idata = pd.read_csv(f, header=None).values
        if dataset == 'DJIA':
            data = idata[1:-1, 1:].astype(float)
        else:
            data = idata[1:-1, 3:].astype(float)
        result.append(data)
        n += 1
    print(np.array(result).shape)
    np.save('../data/data/' + dataset + '.npy', result)


if __name__ == "__main__":
    datasets = "SSE"
    run(datasets)
