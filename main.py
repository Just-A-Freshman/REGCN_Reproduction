"""Unified pipeline: data → GA-VMD → VMD → normalize → graph → train.

Usage:
    python main.py --dataset SSE           # single dataset
    python main.py --dataset DJIA --stock 3  # single stock
    python main.py --pipeline all           # run everything end-to-end
"""

import argparse
import subprocess
import sys
import os
import numpy as np


def run_data_preprocessing(dataset):
    """Raw CSV → numpy arrays."""
    print(f"[Step 1/5] Loading raw data for {dataset}...")
    from dataprecossing.data import run
    run(dataset)


def run_ga_vmd(dataset):
    """GA optimisation of VMD parameters (K, α)."""
    print(f"[Step 2/5] GA-VMD optimisation for {dataset}...")
    from dataprecossing.GA_VMD import run
    run(dataset)


def run_vmd(dataset):
    """VMD decomposition using optimised params."""
    print(f"[Step 3/5] VMD decomposition for {dataset}...")
    from dataprecossing.data_VMD import run
    run(dataset)


def run_normalize(dataset):
    """Normalise VMD-decomposed data to [0,1] using training-set stats."""
    print(f"[Step 4/5] Normalising VMD data for {dataset}...")
    from dataprecossing.normalization import run
    run(dataset)


def run_adj(dataset):
    """Construct adjacency graphs (Pearson / Spearman / DTW)."""
    print(f"[Step 5/5] Building adjacency graphs for {dataset}...")
    from dataprecossing.adjprocessing import run
    run(dataset)


def _load_config():
    """Load training hyperparameters from config.ini."""
    from configparser import ConfigParser
    config = ConfigParser()
    config.read("REGCN/config.ini")
    return {
        "lr": float(config["hyper"]["lr"]),
        "n_neurons": int(config["hyper"]["n_neurons"]),
        "seq_len": int(config["hyper"]["seq_len"]),
        "n_epochs": int(config["hyper"]["n_epochs"]),
    }


def run_training(dataset, stock_index=None):
    """Train REGCN model (hyperparameters read from config.ini)."""
    print(f"[Training] REGCN on {dataset}" +
          (f" stock {stock_index}" if stock_index is not None else ""))
    import REGCN.REGCN as regcn
    from configparser import ConfigParser
    cfg = ConfigParser()
    cfg.read("REGCN/config.ini")
    _regcn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "REGCN")
    base_data_addr = os.path.join(_regcn_dir, cfg["hyper"]["data_addr"])

    # Override datasets and reload data so --dataset takes effect
    regcn.datasets = dataset
    regcn.data_addr = os.path.join(base_data_addr, dataset + ".npy")
    regcn.data = np.load(regcn.data_addr, allow_pickle=True)

    hp = _load_config()
    if stock_index is not None:
        regcn.main(regcn.data, stock_index, **hp)
    else:
        for i in range(regcn.data.shape[0]):
            regcn.main(regcn.data, i, **hp)


def parse_args():
    parser = argparse.ArgumentParser(
        description="REGCN: Regularized Ensemble GCN for stock prediction"
    )
    parser.add_argument("--dataset", default="SSE", choices=["SSE", "DJIA"],
                        help="Stock dataset")
    parser.add_argument("--stock", type=int, default=None,
                        help="Stock index (omit for all)")
    parser.add_argument("--pipeline", default="train",
                        choices=["all", "data", "ga-vmd", "vmd", "normalize", "adj", "train"],
                        help="Pipeline step to run")
    return parser.parse_args()


def main():
    args = parse_args()
    ds = args.dataset
    steps = {
        "data": lambda: run_data_preprocessing(ds),
        "ga-vmd": lambda: run_ga_vmd(ds),
        "vmd": lambda: run_vmd(ds),
        "normalize": lambda: run_normalize(ds),
        "adj": lambda: run_adj(ds),
        "train": lambda: run_training(ds, args.stock),
    }

    if args.pipeline == "all":
        for name in ["data", "ga-vmd", "vmd", "normalize", "adj", "train"]:
            steps[name]()
    else:
        steps[args.pipeline]()

    print(f"[Done] {args.pipeline} for {ds} completed.")


if __name__ == "__main__":
    main()
