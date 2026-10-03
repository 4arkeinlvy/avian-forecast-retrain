"""Leak-free retrain of the notebook WFV (Copy_of_7sep_avian_forecasting_v3).

Differences vs wfv.py (which stays notebook-faithful):
- lag_7/14/28 inside the forecast window are built recursively from the model's own forecasts.
  The notebook reads them from the file, i.e. from future actuals. --leaky restores that behaviour.
- every spike threshold is kept as its own candidate (no threshold picked on test actuals).
- adds SNaive6 (seasonal naive, last Mon-Sat week) and CrostonTSB_daily (TSB on the zero-filled calendar).
- saves day-level forecasts; all metrics and model selection happen afterwards in metrics.py.
"""
import argparse, glob, json, os, time, warnings
import numpy as np
import pandas as pd
from wfv import (WFV_WINDOWS, ALL_FEAT, LAG_COLS, QUANTILES, SPIKE_THRS, SPIKE_FEAT, MIN_TRAIN, CLASS_PATH,
                 load, rolling_fc, fit_croston_tsb, fit_theta, get_p90, build_spike_feats,
                 predict_spike_proba, apply_override, fmt_time)
warnings.filterwarnings('ignore')

OUT_ROOT = 'results/retrain'
LAG_K    = {c: int(c.rsplit('_', 1)[1]) for c in LAG_COLS}
BLOCK    = min(LAG_K.values())
CHUNK_N  = 20


def snaive(hist, n_fc, m=6):
    """Seasonal naive: repeat the last m rows (one Mon-Sat week, so weekdays line up)."""
    return np.resize(hist[-m:], n_fc).astype(float)

def fit_trees(X_tr, y_tr, errors, tag):
    """Same models and hyper-parameters as wfv.py CELL 6, fitted once, returned as predict functions."""
    out = {}
    try:
        from quantile_forest import RandomForestQuantileRegressor
        m = RandomForestQuantileRegressor(n_estimators=50, min_samples_leaf=5,
                                          random_state=42, n_jobs=-1).fit(X_tr, y_tr)
        m.n_jobs = 1
        for q in QUANTILES:
            out[f'QRF_q{int(q*100)}'] = lambda X, m=m, q=q: m.predict(X, quantiles=q)
    except Exception as e:
        errors.append(f'QRF {tag}: {e}')
    try:
        import xgboost as xgb
        for q in QUANTILES:
            m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=q,
                                 n_estimators=100, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, random_state=42, verbosity=0).fit(X_tr, y_tr)
            out[f'XGB_q{int(q*100)}'] = m.predict
    except Exception as e:
        errors.append(f'XGB {tag}: {e}')
    try:
        import lightgbm as lgb
        for q in QUANTILES:
            m = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=100,
                                  num_leaves=31, learning_rate=0.05, subsample=0.8,
                                  random_state=42, verbose=-1).fit(X_tr, y_tr)
            out[f'LGB_q{int(q*100)}'] = m.predict
    except Exception as e:
        errors.append(f'LGB {tag}: {e}')
    return out

def recursive_fc(predict, y_hist, F, lag_pos, y_future=None):
    """Forecast the window in blocks of BLOCK rows. Lag k of row i reads position i-k of
    [history, forecasts]; with k >= BLOCK that position is always before the current block.
    y_future (leaky mode) puts the window's actuals there instead, exactly as the notebook does."""
    n0, n = len(y_hist), len(F)
    path = np.concatenate([y_hist, np.zeros(n) if y_future is None else y_future]).astype(float)
    out = np.empty(n)
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        X = F[s:e].copy()
        for j, k in lag_pos.items():
            X[:, j] = path[n0 + s - k:n0 + e - k]
        out[s:e] = np.clip(predict(X), 0, None)
        if y_future is None:
            path[n0 + s:n0 + e] = out[s:e]
    return out


def run_series(srow, s_f, s_s, feat, leaky, errors):
    """Returns {window: (dates, actual, {model: daily forecast})}."""
    branch, sku, tier = srow['branch_id'], srow['sku_id'], srow['tier']
    s_f = s_f.sort_values('date').reset_index(drop=True)
    for c, k in LAG_K.items():
        s_f[c] = s_f['true_demand'].shift(k)
    lag_pos = {feat.index(c): k for c, k in LAG_K.items() if c in feat}
    out = {}
    for win in WFV_WINDOWS:
        train_end = pd.Timestamp(win['train_end'])
        fc_start, fc_end = pd.Timestamp(win['fc_start']), pd.Timestamp(win['fc_end'])
        tr = s_f[s_f['date'] <= train_end]
        te = s_f[(s_f['date'] >= fc_start) & (s_f['date'] <= fc_end)]
        if len(tr) < 30 or len(te) < 3:
            continue
        actual = te['true_demand'].values.astype(float)
        hist   = tr['true_demand'].values.astype(float)
        n_fc, p90, tag = len(actual), get_p90(tr), f'{branch}/{sku}/{win["name"]}'

        fcs = {'SNaive6': snaive(hist, n_fc),
               'MA7': rolling_fc(hist, n_fc, 7), 'MA14': rolling_fc(hist, n_fc, 14)}
        try:
            hist_s = s_s[s_s['date'] <= train_end]['true_demand'].values.astype(float)
            if len(hist_s) >= 5:
                fcs['CrostonTSB'] = fit_croston_tsb(hist_s, n_fc)
            fcs['CrostonTSB_daily'] = fit_croston_tsb(hist, n_fc)
        except Exception as e:
            errors.append(f'Croston {tag}: {e}')

        base = {}
        if tier == 'FULL' and len(tr) >= MIN_TRAIN:
            d = tr[feat + ['true_demand']].dropna()
            if len(d):
                F = np.nan_to_num(te[feat].values.astype(float))
                for name, pred in fit_trees(d[feat].values, d['true_demand'].values.astype(float),
                                            errors, tag).items():
                    try:
                        base[name] = recursive_fc(pred, hist, F, lag_pos, actual if leaky else None)
                    except Exception as e:
                        errors.append(f'{name} {tag}: {e}')
            try:
                base['Theta'] = fit_theta(hist, n_fc)
            except Exception as e:
                errors.append(f'Theta {tag}: {e}')
            try:
                s_same  = build_spike_feats(s_f[(s_f['date'].dt.month == fc_start.month) &
                                                (s_f['date'] < fc_start)].copy())
                s_fc_sp = build_spike_feats(te.copy())
                avail_sf = [c for c in SPIKE_FEAT if c in s_same.columns]
                if len(s_same) >= 5 and len(avail_sf) >= 5:
                    s_same['is_spike'] = (s_same['true_demand'] >= p90).astype(int)
                    spike_days = s_same[s_same['is_spike'] == 1]['true_demand']
                    spike_mag  = float(spike_days.mean()) if len(spike_days) > 0 else float(p90*1.3)
                    sp_prob = predict_spike_proba(s_fc_sp, s_same, avail_sf, s_same['is_spike'].mean())
                    fcs.update({f'{bk}+Spike{int(thr*100)}': apply_override(bfc, sp_prob, spike_mag, thr)
                                for bk, bfc in base.items() for thr in SPIKE_THRS})
            except Exception as e:
                errors.append(f'Spike {tag}: {e}')
        fcs.update(base)
        out[win['name']] = (te['date'].values, actual, fcs)
    return out

def to_frame(key, tier, res):
    parts = []
    for w, (dates, actual, fcs) in res.items():
        names, n = list(fcs), len(actual)
        parts.append(pd.DataFrame({
            'window': w, 'model': np.repeat(names, n), 'date': np.tile(dates, len(names)),
            'h': np.tile(np.arange(1, n + 1, dtype='int16'), len(names)),
            'fc': np.concatenate([np.asarray(fcs[m], float)[:n] for m in names]).astype('float32'),
            'actual': np.tile(actual, len(names)).astype('float32')}))
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, 'tier', tier); df.insert(0, 'sku_id', key[1]); df.insert(0, 'branch_id', key[0])
    return df


def prepare(args):
    df_f, feat = load(args.data, ALL_FEAT)
    df_s = df_f if not args.sparse else load(args.sparse, ALL_FEAT)[0]
    series = pd.read_csv(CLASS_PATH)
    series = series[series['tier'].isin(['FULL', 'LIMITED', 'VERY_SPARSE'])].reset_index(drop=True)
    gf = dict(tuple(df_f.groupby(['branch_id', 'sku_id'])))
    gs = gf if df_s is df_f else dict(tuple(df_s.groupby(['branch_id', 'sku_id'])))
    return df_f, feat, series, gf, gs

def cmd_run(args):
    out_dir = f'{OUT_ROOT}/{args.name}'
    os.makedirs(f'{out_dir}/daily', exist_ok=True)
    if args.fresh:
        for f in glob.glob(f'{out_dir}/daily/*') + glob.glob(f'{out_dir}/*.*'):
            os.remove(f)
    df_f, feat, series, gf, gs = prepare(args)
    if args.limit:
        series = series.head(args.limit)
    print(f'{args.name}: rows={len(df_f):,} series={len(series)} features={len(feat)} '
          f'{df_f.date.min().date()}..{df_f.date.max().date()} leaky={args.leaky}', flush=True)
    cp = f'{out_dir}/checkpoint.json'
    start = json.load(open(cp))['done'] if os.path.exists(cp) else 0
    if start:
        print(f'  resuming from series {start}')
    empty, errors, buf, times, t0 = df_f.iloc[:0], [], [], [], time.time()
    for si, srow in series.iterrows():
        if si < start:
            continue
        t_s, key = time.time(), (srow['branch_id'], srow['sku_id'])
        try:
            res = run_series(srow, gf.get(key, empty).copy(), gs.get(key, empty), feat, args.leaky, errors)
            if res:
                buf.append(to_frame(key, srow['tier'], res))
        except Exception as e:
            import traceback
            errors.append(f'{key}: {e}\n{traceback.format_exc()[:300]}')
        times.append(time.time() - t_s)
        eta = np.mean(times[-10:]) * (len(series) - si - 1)
        print(f'  [{si+1:>3}/{len(series)}] {key[0]}x{key[1]:<13} {srow["tier"]:<12} {times[-1]:>5.1f}s  '
              f'elapsed {fmt_time(time.time()-t0)}  ETA {fmt_time(eta)}', flush=True)
        if (si + 1) % CHUNK_N == 0 or si + 1 == len(series):
            if buf:
                part = pd.concat(buf, ignore_index=True)
                cats = ['branch_id', 'sku_id', 'tier', 'window', 'model']
                part[cats] = part[cats].astype('category')
                part.to_parquet(f'{out_dir}/daily/part-{si+1:03d}.parquet', index=False)
            buf = []
            json.dump({'done': si + 1}, open(cp, 'w'))
    if errors:
        with open(f'{out_dir}/errors.txt', 'a') as f:
            f.write('\n'.join(errors) + '\n')
    meta = {'name': args.name, 'data': args.data, 'sparse': args.sparse, 'leaky': args.leaky,
            'runtime': fmt_time(time.time() - t0), 'series': len(series), 'errors': len(errors)}
    json.dump(meta, open(f'{out_dir}/meta.json', 'w'), indent=1)
    print(f'done in {meta["runtime"]}, errors={len(errors)} -> {out_dir}/daily/', flush=True)

def cmd_leaktest(args):
    """Scramble every actual from a window's fc_start onward: a leak-free forecast for that
    window must not change. Run in both modes; the leaky mode must change (test has power)."""
    df_f, feat, series, gf, gs = prepare(args)
    rng = np.random.default_rng(0)
    full = series[series['tier'] == 'FULL'].head(args.n)
    for leaky in (False, True):
        changed, checked = set(), 0
        for _, srow in full.iterrows():
            key = (srow['branch_id'], srow['sku_id'])
            s = gf[key].copy()
            for win in WFV_WINDOWS[:3]:
                fut = s['date'] >= pd.Timestamp(win['fc_start'])
                s2 = s.copy()
                s2.loc[fut, 'true_demand'] = rng.integers(0, 500, fut.sum()).astype(float)
                a = run_series(srow, s.copy(), gs.get(key, s.iloc[:0]), feat, leaky, [])
                b = run_series(srow, s2, gs.get(key, s.iloc[:0]), feat, leaky, [])
                if win['name'] not in a:
                    continue
                fa, fb = a[win['name']][2], b[win['name']][2]
                for m in fa:
                    checked += 1
                    if not np.allclose(fa[m], fb[m]):
                        changed.add(m.split('+')[0])
        print(f'leaky={leaky}: {checked} forecasts checked, models that changed: {sorted(changed) or "none"}')
        if not leaky:
            assert not changed, 'LEAK: leak-free forecast depends on future actuals'
        else:
            assert changed, 'leak test has no power: leaky mode did not change anything'
    print('leaktest ok')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'leaktest'):
        r = sub.add_parser(name)
        r.add_argument('--data', required=True)
        r.add_argument('--sparse', help='Croston input (default: --data)')
        if name == 'run':
            r.add_argument('--name', required=True)
            r.add_argument('--leaky', action='store_true', help='notebook behaviour: window actuals in lags')
            r.add_argument('--limit', type=int, help='first N series only (smoke test)')
            r.add_argument('--fresh', action='store_true', help='delete previous output first')
        else:
            r.add_argument('--n', type=int, default=2, help='FULL series to test')
    a = p.parse_args()
    {'run': cmd_run, 'leaktest': cmd_leaktest}[a.cmd](a)
