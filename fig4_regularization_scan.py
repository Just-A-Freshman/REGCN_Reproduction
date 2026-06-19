"""
Figure 4: Systematic scan of λ₁ (r_mse) × λ₂ (r_acc) regularization coefficients.

Scans a grid of (r_mse, r_acc) values on a single representative stock,
then prints a summary table suitable for recreating Figure 4 from the paper.
"""
import os
import sys
import subprocess
import csv
import itertools
import shutil
from configparser import ConfigParser

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_INI = os.path.join(PROJECT_ROOT, 'REGCN', 'config.ini')
PYTHON = r'D:/Program_Files/Anaconda/python.exe'

# ── Config ──────────────────────────────────────────────────────
STOCK = 0          # representative stock
DATASET = 'SSE'

# Grid: λ₁ (L1 weight reg)  ×  λ₂ (trend-direction reg)
r_mse_vals = [0, 0.0001, 0.001, 0.01]
r_acc_vals = [0, 0.01, 0.1, 0.5, 1.0]
# ────────────────────────────────────────────────────────────────

results = []
os.makedirs(os.path.join(PROJECT_ROOT, 'result', 'Figure4'), exist_ok=True)

# Backup original config
shutil.copy2(CONFIG_INI, CONFIG_INI + '.bak')
print(f'Backup saved to {CONFIG_INI}.bak')

try:
    total = len(r_mse_vals) * len(r_acc_vals)
    for i, (rm, ra) in enumerate(itertools.product(r_mse_vals, r_acc_vals), 1):
        print(f'\n[{i}/{total}] r_mse={rm}, r_acc={ra}  (stock {STOCK})')
        print('-' * 50)

        # ── Write config ──
        cfg = ConfigParser()
        cfg.read(CONFIG_INI)
        cfg['hyper']['r_mse'] = str(rm)
        cfg['hyper']['r_acc'] = str(ra)
        cfg['hyper']['s_index'] = str(STOCK)
        cfg['hyper']['all_data'] = '0'
        with open(CONFIG_INI, 'w') as f:
            cfg.write(f)

        # ── Run REGCN ──
        env = os.environ.copy()
        env['TF_ENABLE_ONEDNN_OPTS'] = '0'
        # Limit CPU threads so local machine stays responsive
        env['TF_NUM_INTRAOP_THREADS'] = '4'
        env['TF_NUM_INTEROP_THREADS'] = '1'
        env['TF_CPP_MIN_LOG_LEVEL'] = '1'  # suppress info/warning logs

        r = subprocess.run(
            [PYTHON, 'REGCN/REGCN.py'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )

        # Print stdout (metrics lines)
        for line in r.stdout.splitlines():
            stripped = line.strip()
            if any(k in stripped for k in ('accuracy', 'r2', 'rmse', 'mae', 're', 'pred_range', 'actual_range', 'WARNING', 'ERROR')):
                print(f'  {stripped}')

        if r.returncode != 0:
            print(f'  [ERROR] return code {r.returncode}')
            for line in r.stderr.splitlines():
                print(f'  ERR: {line}')
            continue

        # ── Read result CSV ──
        result_csv = os.path.join(
            PROJECT_ROOT, 'result', 'Table5', DATASET,
            f'result_REGCN_stock{STOCK}.csv'
        )
        if os.path.exists(result_csv):
            with open(result_csv) as f:
                row = next(csv.reader(f))
            results.append({
                'r_mse': rm,
                'r_acc': ra,
                'accuracy': float(row[2]),
                'r2': float(row[3]),
                'rmse': float(row[4]),
                'mae': float(row[5]),
                're': float(row[6]),
            })
            print(f'  [OK] ACC={row[2]}  R2={row[3]}  RMSE={row[4]}')
        else:
            print(f'  [FAIL] No result file found')

finally:
    # Restore original config
    shutil.move(CONFIG_INI + '.bak', CONFIG_INI)
    print(f'\nConfig restored from backup.')


# -- Summary ----------------------------------------------------
print('\n' + '=' * 70)
print('Figure 4 - Regularization Coefficient Grid Scan')
print(f'Dataset: {DATASET}, Stock: #{STOCK}')
print('=' * 70)

# Table header
print(f'\n{"r_mse(L1)":>10} {"r_acc(L2)":>10} {"ACC":>8} {"R2":>10} {"RMSE":>10} {"MAE":>10}')
print('-' * 58)

for r in results:
    print(f'{r["r_mse"]:>10} {r["r_acc"]:>10} {r["accuracy"]:>8.4f} {r["r2"]:>10.4f} {r["rmse"]:>10.4f} {r["mae"]:>10.4f}')

# Best R2 combo
if results:
    best = max(results, key=lambda x: x['r2'])
    print(f'\nBest R2: r_mse={best["r_mse"]}, r_acc={best["r_acc"]}  '
          f'(ACC={best["accuracy"]:.4f}, R2={best["r2"]:.4f})')

# Save to CSV
summary_csv = os.path.join(PROJECT_ROOT, 'result', 'Figure4',
                           f'figure4_stock{STOCK}_grid.csv')
with open(summary_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['r_mse', 'r_acc', 'accuracy', 'r2', 'rmse', 'mae', 're'])
    for r in results:
        w.writerow([r['r_mse'], r['r_acc'], r['accuracy'],
                     r['r2'], r['rmse'], r['mae'], r['re']])

print(f'\nSummary saved to: {summary_csv}')
print('Done.')
