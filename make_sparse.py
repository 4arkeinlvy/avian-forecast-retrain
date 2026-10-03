"""Rebuild the activity-days ("sparse") files used as Croston input for v4/v5 (run_v2.sh, run_retrain.sh).

v3 ships its own sparse file (Dataset/model_ready_df_v3_pilot.csv). v4 and v5 define true_demand differently, so
their sparse files are rebuilt from each dataset: a row is kept if anything happened that day (transactions,
order intake, a change in open sales orders, or demand).
"""
import argparse, os
import pandas as pd

SRC = {'v4_pilot_clean': 'Dataset/another data/v4_pilot_clean.csv',
       'v5_pilot':       'Dataset/another data/v5_pilot.csv'}

def sparse(df):
    keep = ((df['obs_transaction_count'] > 0) | (df['obs_order_intake'] > 0) |
            (df['obs_delta_SO'] != 0) | (df['true_demand'] > 0))
    return df[keep]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/sparse')
    out = p.parse_args().out
    os.makedirs(out, exist_ok=True)
    for name, path in SRC.items():
        df = pd.read_csv(path, low_memory=False)
        s = sparse(df)
        s.to_csv(f'{out}/{name}_sparse.csv', index=False)
        print(f'{name}: {len(df):,} -> {len(s):,} rows -> {out}/{name}_sparse.csv')
