#!/usr/bin/env bash
# Leak-free retrain on every daily dataset (+ notebook-style leaky ablation), then the metrics.
# Resumes from checkpoints; add --fresh to a line to restart it.
set -euo pipefail
cd "$(dirname "$0")"
. ./common.sh
SPARSE=Dataset/model_ready_df_v3_pilot.csv
FILLED=Dataset/model_ready_df_v3_pilot_filled.csv
A="Dataset/another data"
run() { py "retrain.py run $*"; }
[ -f results/sparse/v5_pilot_sparse.csv ] || py make_sparse.py   # v4/v5 Croston input
run --name v3_filled       --data $FILLED                    --sparse $SPARSE
run --name v3_filled_leaky --data $FILLED                    --sparse $SPARSE --leaky
run --name v3_clean        --data "'$A/v3_pilot_clean.csv'"  --sparse $SPARSE
run --name v4_clean        --data "'$A/v4_pilot_clean.csv'"  --sparse results/sparse/v4_pilot_clean_sparse.csv
run --name v5              --data "'$A/v5_pilot.csv'"        --sparse results/sparse/v5_pilot_sparse.csv
py "metrics.py eval"
