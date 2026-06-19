import os
import pandas as pd
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..'))


def run(dataset, train_rate=0.7):
    """Normalise VMD CSV files to [0,1] using training-set min/max only.

    Output goes to VMDnor/. Each CSV has the normalised data followed
    by two extra rows: the per-column min (row -2) and max (row -1)
    that were computed from the training set only.

    The training pipeline (REGCN.py) reads from VMDnor/ and uses the
    appended min/max rows to denormalise predictions back to original scale.
    """
    input_folder = os.path.join(_project_root, 'data/data/VMDdata/', dataset, '')
    output_folder = os.path.join(_project_root, 'data/data/VMDdata/', dataset, '')

    # When input == output, we're overwriting raw VMD CSVs with normalized ones
    # (the original dataset already had this format — normalized data + min/max rows)
    os.makedirs(output_folder, exist_ok=True)

    file_names = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    for file_name in file_names:
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path, header=None)

        n_train = int(df.shape[0] * train_rate)
        train_df = df.iloc[:n_train]

        min_vals = train_df.min()
        max_vals = train_df.max()
        # Avoid division by zero for constant columns
        denom = (max_vals - min_vals).replace(0, 1.0)
        normalized_df = (df - min_vals) / denom

        min_max_df = pd.DataFrame([min_vals, max_vals])
        normalized_df = pd.concat([normalized_df, min_max_df], ignore_index=True)

        output_path = os.path.join(output_folder, file_name)
        normalized_df.to_csv(output_path, index=False, header=False)

    print("Normalization completed using training-set statistics only.")


if __name__ == '__main__':
    datasets = 'SSE'
    run(datasets)
