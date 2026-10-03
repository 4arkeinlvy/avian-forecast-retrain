# Avian demand forecasting: leak-free retrain and honest evaluation

Retrain of the models in the Colab notebook `notebook/Copy_of_7sep_avian_forecasting_v3.ipynb` (the source of
truth), without the evaluation problems found in the notebook, scored with standard accuracy metrics at daily,
weekly and monthly level.

Data: 50 SKUs x 3 branches (11B, 11C, 12A), daily Monday to Saturday, February 2024 to May 2026.

## Repository layout

| Path | Content |
|---|---|
| `notebook/Copy_of_7sep_avian_forecasting_v3.ipynb` | Original Colab notebook (source of truth, unchanged) |
| `Dataset/model_ready_df_v3_pilot_filled.csv` | v3, every Monday to Saturday (zero-filled): the notebook's main input |
| `Dataset/model_ready_df_v3_pilot.csv` | v3, activity days only ("sparse"): the notebook's Croston input |
| `Dataset/series_classification_v3.csv` | Series routing: FULL / LIMITED / VERY_SPARSE / MISSING |
| `Dataset/final_model_selection_v3.csv` | Original Colab result, for reference |
| `Dataset/another data/` | `v3_pilot_clean`, `v4_pilot_clean`, `v5_pilot` (daily) and `v5_pilot_month` (monthly) |
| `wfv.py` | The notebook's building blocks: windows, features, FA% formula, spike layer, Croston, Theta |
| `retrain.py` | Leak-free retrain; saves the daily forecast of every candidate model |
| `metrics.py` | Daily / weekly / monthly metrics and honest (rolling-origin) model selection |
| `make_sparse.py` | Rebuilds the v4/v5 activity-day files used as Croston input |
| `run_retrain.sh`, `common.sh` | Runs everything |
| `shell.nix` | Pinned Nix environment |
| `pyproject.toml`, `uv.lock`, `.python-version` | Pinned uv environment |
| `results/retrain/` | Results of the retrain |

## Environment

Two options with the same pinned package versions. Both give identical forecasts.

| Python | pandas | numpy | scikit-learn | xgboost | lightgbm | quantile-forest | pyarrow |
|---|---|---|---|---|---|---|---|
| 3.13 | 2.3.3 | 2.4.4 | 1.8.0 | 3.0.5 | 4.6.0 | 1.4.1 | 23.0.0 |

**Option A: Nix** (Linux, macOS, Windows through WSL2). `shell.nix` pins nixpkgs to revision
`6713828a351efa628b025a1adf7f43cbf8597513`.

```bash
nix-shell --run "python wfv.py selftest"
```

**Option B: uv** (Windows, macOS, Linux). Install uv from https://docs.astral.sh/uv/, then:

```bash
uv sync                          # installs Python 3.13 and the locked packages into .venv
uv run python wfv.py selftest
```

On macOS, xgboost and lightgbm also need OpenMP: `brew install libomp`. On NixOS use Option A, or
`uv sync --python-preference only-managed`.

## How to run

Run all commands from the repository root.

**1. Checks** (a few minutes)

```bash
uv run python wfv.py selftest
uv run python metrics.py selftest
uv run python retrain.py leaktest --data Dataset/model_ready_df_v3_pilot_filled.csv --sparse Dataset/model_ready_df_v3_pilot.csv
```

With Nix, replace `uv run python ...` by `nix-shell --run "python ..."`. The leak test scrambles every actual
from the forecast origin onward: the leak-free forecasts must not change, the notebook-style (`--leaky`)
forecasts must change.

**2. Retrain and metrics** (about 45 minutes, about 1 GB of RAM)

On Linux, macOS, WSL or Git Bash:

```bash
./run_retrain.sh
```

The script uses Nix if it is installed, otherwise uv, otherwise `python` on the PATH (force one with
`RUNNER=nix`, `RUNNER=uv` or `RUNNER=python`). In Windows PowerShell, run the same steps directly:

```powershell
uv run python make_sparse.py
uv run python retrain.py run --name v3_filled --data Dataset/model_ready_df_v3_pilot_filled.csv --sparse Dataset/model_ready_df_v3_pilot.csv
uv run python retrain.py run --name v3_filled_leaky --data Dataset/model_ready_df_v3_pilot_filled.csv --sparse Dataset/model_ready_df_v3_pilot.csv --leaky
uv run python retrain.py run --name v3_clean --data "Dataset/another data/v3_pilot_clean.csv" --sparse Dataset/model_ready_df_v3_pilot.csv
uv run python retrain.py run --name v4_clean --data "Dataset/another data/v4_pilot_clean.csv" --sparse results/sparse/v4_pilot_clean_sparse.csv
uv run python retrain.py run --name v5 --data "Dataset/another data/v5_pilot.csv" --sparse results/sparse/v5_pilot_sparse.csv
uv run python metrics.py eval
```

Runs resume from their checkpoint; add `--fresh` to start a run again.

### Outputs (`results/retrain/<run>/`)

Runs: `v3_filled`, `v3_filled_leaky` (notebook-style lags, for the leakage comparison), `v3_clean`, `v4_clean`, `v5`.

| File | Content |
|---|---|
| `daily/*.parquet` | Forecast and actual of every candidate model, per day |
| `sw_metrics.parquet` | All metrics per series, window and candidate |
| `picks.csv` | Model chosen by each selection rule, per series and window |
| `eval.csv` | Summary per selection rule and bucket (FA, bias, MASE, RMSSE) |
| `eval_rows.parquet` | Metrics of every pick |
| `quantile_calibration.csv` | Daily coverage and scaled pinball loss of the quantile models |
| `horizon.csv` | Daily FA of the tree models by forecast day |
| `demand_class.csv` | Syntetos-Boylan demand class per series |

## Issues found in the notebook and what changed

### Metrics

- **FA% measured the window total, not daily accuracy.** `FA% = 100 * (1 - |sum F - sum A| / sum A)` over a whole
  one- or two-month window, so daily and weekly over- and under-forecasts cancel; it is a bias measure.
  Now: FA = 1 - WAPE, with errors formed per bucket (day, Monday-to-Saturday week, calendar month) before summing.
- **Two-month windows.** W2, W4, W5 and W8 are two months long, so their FA% mixed two months.
  Now: the monthly bucket scores each calendar month.
- **Plain average over series.** A 3-unit SKU weighed as much as a 3,000-unit SKU.
  Now: volume-weighted totals, plus the per-series median.
- **"Bias", "MAPE (approx)" and "WMAPE" were the same number.** All three came from window averages, and "WMAPE"
  was weighted by the actual squared. Now: bias = (sum F - sum A) / sum A, reported separately, and
  WAPE = sum |F - A| / sum A.
- **Quantile "coverage" from window averages.** Now: daily coverage (a band from share(A < F) to share(A <= F),
  because of ties on zero days) and scaled pinball loss.
- **MAPE_ns and the composite score.** They drop zero and spike days and add arbitrary constants (+500, +30).
  Now: not used.
- **No benchmark.** Now: seasonal naive (last Monday-to-Saturday week), MA14 and Croston on the daily calendar,
  with MASE, RMSSE and RMSE relative to the seasonal naive.

### Training and evaluation

- **Lag leakage.** Lags 7/14/28 of forecast days were read from future actuals. Now: the tree models forecast
  recursively from their own predictions, checked by `retrain.py leaktest`.
- **Spike threshold tuned on the test actuals.** Now: every threshold is its own candidate model.
- **Model chosen on the windows it is scored on.** Now: rolling origin; the model for window Wk is chosen on
  W1 to W(k-1) only.
- **Selection by composite score and FA%.** Now: daily RMSSE (squared error, so near-zero forecasts are not
  rewarded on intermittent days), weekly WAPE and monthly WAPE.
- **Bias limit on the signed mean.** +60% and -60% windows averaged to 0%. Now: mean of the absolute bias.
- **NaN rows silently disabled the tree models.** Now: missing feature values are filled.
- **Croston fitted on the activity-days file.** Its demand probability was per activity day but the forecast
  was spread over every calendar day. Now: also fitted on the zero-filled calendar (`CrostonTSB_daily`).

### Kept as in the notebook

So that the effect of each fix can be measured on its own, the retrain keeps the notebook's windows (W1 to W8),
series routing, model families and hyper-parameters, the hand-written Theta, lag 7 as 7 rows (not the same
weekday), one local model per series with constant branch/SKU codes, and the original tier table.
