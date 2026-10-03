"""Standard accuracy metrics at daily / weekly / monthly buckets for the retrain, and honest (rolling-origin)
model selection. Reads results/retrain/<run>/daily/*.parquet.

Per series x window x model and bucket b (day, calendar Mon-Sat week clipped to the window, calendar month):
  WAPE_b  = sum|F-A| / sum A           FA_b = 1 - WAPE_b      (errors first, then summed: nothing cancels)
  Bias    = (sum F - sum A) / sum A    (same for every bucket; this is what the notebook FA% measures)
  MASE_b  = mean|F-A| / in-sample naive MAE   (day: seasonal lag 6 = one Mon-Sat week; week/month: lag 1)
  RMSSE_b = sqrt(mean (F-A)^2 / in-sample one-step naive MSE)   (M5; scales start at first non-zero)
  old_fa  = notebook monthly_fa = max(0, 100 - |sum F - sum A| / sum A * 100)
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
from wfv import WFV_WINDOWS, ALL_FEAT, load, monthly_fa

ROOT      = 'results/retrain'
KEYS      = ['branch_id', 'sku_id']
SW        = KEYS + ['window', 'model']
BUCKETS   = ['day', 'week', 'month']
WEEKMASK  = '1111110'
TRAIN_END = {w['name']: pd.Timestamp(w['train_end']) for w in WFV_WINDOWS}
CLASS_END = pd.Timestamp('2025-05-31')
COMMON    = ['W3', 'W4', 'W5', 'W6']
BENCH     = ['SNaive6', 'MA14', 'CrostonTSB_daily']
TREE_BASE = [f'{f}_q{q}' for f in ('QRF', 'XGB', 'LGB') for q in (50, 60, 75)]
CRITERIA  = {
    'daily_rmsse':     ('daily_rmsse',  True,  True),
    'weekly_wape':     ('weekly_wape',  True,  True),
    'monthly_wape':    ('monthly_wape', True,  True),
    'daily_rmsse_ng':  ('daily_rmsse',  True,  False),
    'weekly_wape_ng':  ('weekly_wape',  True,  False),
    'monthly_wape_ng': ('monthly_wape', True,  False),
    'old_fa':          ('old_fa',       False, False),
}
LABEL = {'daily_rmsse': 'Daily RMSSE', 'weekly_wape': 'Weekly WAPE', 'monthly_wape': 'Monthly WAPE',
         'daily_rmsse_ng': 'Daily RMSSE (no guard)', 'weekly_wape_ng': 'Weekly WAPE (no guard)',
         'monthly_wape_ng': 'Monthly WAPE (no guard)', 'old_fa': 'Old FA% (honest)',
         'insample_old_fa': 'Old FA% picked on test windows', 'SNaive6': 'Seasonal naive',
         'MA14': 'MA14', 'CrostonTSB_daily': 'Croston TSB (daily)'}


def week_key(dates):
    return dates - pd.to_timedelta(dates.dt.dayofweek, unit='D')

def bucket_stats(d):
    """Per series x window x model: for each bucket n, sum|e|, sum e^2, sum A, sum F.
    'weekf' = full 6-day weeks only (sensitivity for partial edge weeks)."""
    d = d[SW + ['date', 'fc', 'actual']].copy()
    d['fc'], d['actual'] = d['fc'].astype(float), d['actual'].astype(float)
    d['week'], d['month'] = week_key(d['date']), d['date'].values.astype('datetime64[M]')
    out = []
    for b, key in [('day', None), ('week', 'week'), ('month', 'month')]:
        g = d if key is None else (d.groupby(SW + [key], observed=True, sort=False)
                                    .agg(fc=('fc', 'sum'), actual=('actual', 'sum'), k=('fc', 'size'))
                                    .reset_index())
        for name, gg in [(b, g)] + ([('weekf', g[g['k'] == 6])] if b == 'week' else []):
            e = gg['fc'] - gg['actual']
            s = (gg.assign(ae=e.abs(), se=e**2)
                   .groupby(SW, observed=True)
                   .agg(n=('ae', 'size'), sae=('ae', 'sum'), sse=('se', 'sum'),
                        sa=('actual', 'sum'), sf=('fc', 'sum')))
            out.append(s.add_prefix(f'{name}_'))
    return pd.concat(out, axis=1).reset_index()


def naive_scale(y, m=1, sq=False):
    """Mean |y_t - y_{t-m}| (or squared) from the first non-zero value; NaN if undefined or zero."""
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0 or len(y) - nz[0] <= m:
        return np.nan
    y = y[nz[0]:]
    d = y[m:] - y[:-m]
    v = float(np.mean(d**2) if sq else np.mean(np.abs(d)))
    return v if v > 0 else np.nan

def complete_buckets(tr):
    """Weekly and monthly totals of the training rows, complete buckets only."""
    wk = tr.groupby(week_key(tr['date']))['true_demand'].agg(['sum', 'size'])
    wk = wk[wk['size'] == 6]['sum'].values
    mo = tr.groupby(tr['date'].values.astype('datetime64[M]'))['true_demand'].agg(['sum', 'size'])
    mk = mo.index.values.astype('datetime64[M]')
    exp = np.busday_count(mk.astype('datetime64[D]'), (mk + 1).astype('datetime64[D]'), weekmask=WEEKMASK)
    return wk.astype(float), mo[mo['size'].values == exp]['sum'].values.astype(float)

def demand_class(y):
    """Syntetos-Boylan-Croston class from ADI (mean interval between demands) and CV^2 of demand sizes."""
    nz = np.flatnonzero(y > 0)
    if len(nz) < 2:
        return 'dead', np.nan, np.nan
    adi = (nz[-1] - nz[0]) / (len(nz) - 1) if len(nz) > 1 else np.nan
    adi = max(adi, 1.0)
    sz = y[nz]
    cv2 = float((sz.std() / sz.mean())**2)
    cls = ('smooth' if cv2 < 0.49 else 'erratic') if adi < 1.32 else ('intermittent' if cv2 < 0.49 else 'lumpy')
    return cls, adi, cv2

def scales(data_path):
    df, _ = load(data_path, ALL_FEAT)
    rows, cls = [], []
    for (b, s), g in df.groupby(KEYS):
        g = g.sort_values('date')
        c, adi, cv2 = demand_class(g[g['date'] <= CLASS_END]['true_demand'].values.astype(float))
        cls.append({'branch_id': b, 'sku_id': s, 'demand_class': c, 'adi': adi, 'cv2': cv2})
        for w, te in TRAIN_END.items():
            tr = g[g['date'] <= te]
            if len(tr) < 30:
                continue
            y = tr['true_demand'].values.astype(float)
            wk, mo = complete_buckets(tr)
            rows.append({'branch_id': b, 'sku_id': s, 'window': w,
                         'day_scale_mae': naive_scale(y, 6), 'day_scale_mse': naive_scale(y, 1, True),
                         'day_scale_mae1': naive_scale(y, 1),
                         'week_scale_mae': naive_scale(wk), 'week_scale_mse': naive_scale(wk, 1, True),
                         'month_scale_mae': naive_scale(mo), 'month_scale_mse': naive_scale(mo, 1, True)})
    return pd.DataFrame(rows), pd.DataFrame(cls)

def add_metrics(st, sc):
    m = st.merge(sc, on=['branch_id', 'sku_id', 'window'], how='left')
    for b in ['day', 'week', 'weekf', 'month']:
        sb = 'week' if b == 'weekf' else b
        sa = m[f'{b}_sa']
        m[f'{b}_wape']  = m[f'{b}_sae'] / sa.where(sa > 0)
        m[f'{b}_mase']  = m[f'{b}_sae'] / m[f'{b}_n'] / m[f'{sb}_scale_mae']
        m[f'{b}_rmsse'] = np.sqrt(m[f'{b}_sse'] / m[f'{b}_n'] / m[f'{sb}_scale_mse'])
    sa = m['day_sa']
    m['bias']   = (m['day_sf'] - sa) / sa.where(sa > 0)
    m['old_fa'] = np.where(sa > 0, np.clip(100 - (m['day_sf'] - sa).abs() / sa.where(sa > 0) * 100, 0, None), 0.0)
    return m


def _wkey(w):
    return int(str(w)[1:])

def honest_picks(m, min_prior=2):
    """Model for window Wk chosen on W1..W(k-1) only. Bias guard: mean |window bias| <= 40%, else 60%."""
    out = []
    for (b, s), g in m.groupby(KEYS, observed=True):
        wins = sorted(g['window'].unique(), key=_wkey)
        for i in range(min_prior, len(wins)):
            cur = set(g.loc[g['window'] == wins[i], 'model'])
            past = g[g['window'].isin(wins[:i]) & g['model'].isin(cur)]
            ps = (past.groupby('model', observed=True)
                      .agg(n=('window', 'nunique'), rmsse=('day_rmsse', 'mean'), sse=('day_sse', 'sum'),
                           wk=('week_sae', 'sum'), mo=('month_sae', 'sum'), sa=('day_sa', 'sum'),
                           absbias=('bias', lambda x: x.abs().mean()), old_fa=('old_fa', 'mean')))
            ps = ps[ps['n'] >= min_prior]
            if not len(ps):
                continue
            ps['daily_rmsse']  = ps['rmsse'] if ps['rmsse'].notna().any() else ps['sse']
            ps['weekly_wape']  = ps['wk'] / ps['sa'].clip(lower=1e-9)
            ps['monthly_wape'] = ps['mo'] / ps['sa'].clip(lower=1e-9)
            el = ps[ps['absbias'] <= .4]
            el = el if len(el) else ps[ps['absbias'] <= .6]
            el = el if len(el) else ps
            for crit, (col, low, guard) in CRITERIA.items():
                pool = el if guard else ps
                v = pool[col].dropna()
                if not len(v):
                    v = pool['sse']
                    low = True
                out.append({'branch_id': b, 'sku_id': s, 'window': wins[i], 'criterion': crit,
                            'model': v.idxmin() if low else v.idxmax()})
    return pd.DataFrame(out)

def reference_picks(m, scored):
    """Benchmarks (fixed model) and the notebook-style pick: best mean old FA% over the scored
    windows themselves (selection on test, as notebook cells 15-16)."""
    out = []
    ms = m[m['window'].isin(scored)]
    for (b, s), g in ms.groupby(KEYS, observed=True):
        for w in g['window'].unique():
            have = set(g.loc[g['window'] == w, 'model'])
            out += [{'branch_id': b, 'sku_id': s, 'window': w, 'criterion': bm, 'model': bm}
                    for bm in BENCH if bm in have]
        fa = g.groupby('model', observed=True).agg(fa=('old_fa', 'mean'), n=('window', 'nunique'))
        fa = fa[fa['n'] == fa['n'].max()]
        best = fa['fa'].idxmax()
        out += [{'branch_id': b, 'sku_id': s, 'window': w, 'criterion': 'insample_old_fa', 'model': best}
                for w in g.loc[g['model'] == best, 'window'].unique()]
    return pd.DataFrame(out)


def near_dead(m):
    a = m[m['model'] == 'MA14']
    am = (a['day_sa'] / a['day_n']).groupby([a['branch_id'], a['sku_id']], observed=True).max()
    return set(am[am < 0.5].index)

def summarize(e, buckets=('day', 'week', 'month')):
    """One row per bucket: pooled (volume-weighted) and per-series statistics."""
    rows = []
    for b in buckets:
        per = e.groupby(KEYS, observed=True).agg(sae=(f'{b}_sae', 'sum'), sa=(f'{b}_sa', 'sum'),
                                                 mase=(f'{b}_mase', 'mean'), rmsse=(f'{b}_rmsse', 'mean'),
                                                 old_fa=('old_fa', 'mean'))
        fa_s = 1 - per['sae'] / per['sa'].where(per['sa'] > 0)
        rows.append({'bucket': b, 'n_series': len(per), 'n_rows': len(e),
                     'fa_pooled': (1 - e[f'{b}_sae'].sum() / e[f'{b}_sa'].sum()) * 100,
                     'fa_series_median': fa_s.median() * 100, 'fa_series_mean': fa_s.clip(lower=0).mean() * 100,
                     'share_fa_ge70': (fa_s >= .7).mean() * 100,
                     'mase_median': per['mase'].median(), 'rmsse_median': per['rmsse'].median(),
                     'bias_pooled': (e['day_sf'].sum() - e['day_sa'].sum()) / e['day_sa'].sum() * 100,
                     'old_fa_mean': per['old_fa'].mean()})
    return pd.DataFrame(rows)

def evaluate(m, picks, windows, dead):
    e = picks[picks['window'].isin(windows)].merge(m, on=SW)
    e = e[~e.set_index(KEYS).index.isin(list(dead))]
    res = []
    for crit, g in e.groupby('criterion'):
        res.append(summarize(g).assign(criterion=crit))
    return pd.concat(res, ignore_index=True), e


def quantile_calibration(d, sc):
    t = d[d['model'].isin(TREE_BASE)].copy()
    t['q'] = t['model'].astype(str).str[-2:].astype(int) / 100
    e = t['actual'].astype(float) - t['fc'].astype(float)
    t['pin'] = np.where(e >= 0, t['q'] * e, (t['q'] - 1) * e)
    t['cov'] = (t['actual'] <= t['fc']).astype(float)
    t['cov_lo'] = (t['actual'] < t['fc']).astype(float)
    sw = (t.groupby(SW, observed=True)
           .agg(q=('q', 'first'), pin=('pin', 'mean'), cov=('cov', 'mean'), cov_lo=('cov_lo', 'mean'),
                fm=('fc', 'mean'), am=('actual', 'mean'))
           .reset_index().merge(sc[KEYS + ['window', 'day_scale_mae1']], on=KEYS + ['window'], how='left'))
    sw['spl'] = sw['pin'] / sw['day_scale_mae1']
    sw['proxy'] = (sw['fm'] >= sw['am']).astype(float)
    return (sw.groupby('model', observed=True)
              .agg(quantile=('q', 'first'), daily_coverage=('cov', 'mean'), daily_coverage_lo=('cov_lo', 'mean'),
                   notebook_proxy=('proxy', 'mean'),
                   scaled_pinball_median=('spl', 'median'))
              .reset_index())

def horizon_wape(d, models=TREE_BASE):
    t = d[d['model'].isin(models)]
    hb = pd.cut(t['h'], [0, 7, 14, 28, 99], labels=['1-7', '8-14', '15-28', '29+'])
    g = t.assign(ae=(t['fc'].astype(float) - t['actual'].astype(float)).abs(), hb=hb) \
         .groupby('hb', observed=True).agg(sae=('ae', 'sum'), sa=('actual', 'sum'))
    return (1 - g['sae'] / g['sa']) * 100


def run_dirs():
    return sorted(d for d in glob.glob(f'{ROOT}/*') if os.path.exists(f'{d}/meta.json'))

def cmd_eval(args):
    for d in run_dirs():
        name = os.path.basename(d)
        if args.run and name not in args.run:
            continue
        meta = json.load(open(f'{d}/meta.json'))
        print(f'[{name}] loading daily forecasts ...', flush=True)
        daily = pd.read_parquet(f'{d}/daily')
        sc, cls = scales(meta['data'])
        m = add_metrics(bucket_stats(daily), sc)
        tiers = daily.groupby(KEYS, observed=True)['tier'].first().reset_index()
        m = m.merge(tiers, on=KEYS).merge(cls, on=KEYS, how='left')
        m.to_parquet(f'{d}/sw_metrics.parquet', index=False)
        cls.to_csv(f'{d}/demand_class.csv', index=False)
        wins = sorted(m['window'].unique(), key=_wkey)
        scored = wins[2:]
        picks = pd.concat([honest_picks(m), reference_picks(m, scored)], ignore_index=True)
        picks.to_csv(f'{d}/picks.csv', index=False)
        dead = near_dead(m)
        ev, e = evaluate(m, picks, scored, dead)
        ev.insert(0, 'windows', f'{scored[0]}-{scored[-1]}')
        ev36, _ = evaluate(m, picks, COMMON, dead)
        ev36.insert(0, 'windows', 'W3-W6')
        pd.concat([ev, ev36]).assign(run=name).to_csv(f'{d}/eval.csv', index=False)
        e[KEYS + ['window', 'criterion', 'model', 'tier', 'demand_class'] +
          [c for c in e.columns if c.endswith(('_sae', '_sse', '_sa', '_sf', '_n', '_mase', '_rmsse', '_wape'))] +
          ['bias', 'old_fa']].to_parquet(f'{d}/eval_rows.parquet', index=False)
        quantile_calibration(daily, sc).assign(run=name).to_csv(f'{d}/quantile_calibration.csv', index=False)
        horizon_wape(daily).rename('daily_fa_pooled').reset_index().assign(run=name) \
            .to_csv(f'{d}/horizon.csv', index=False)
        main = ev[(ev['criterion'].isin(['daily_rmsse', 'weekly_wape', 'monthly_wape', 'old_fa', 'insample_old_fa',
                                         'SNaive6']))]
        print(main.pivot_table(index='criterion', columns='bucket', values='fa_pooled').round(1).to_string(), flush=True)
        del daily, m, e

def cmd_selftest(_):
    a = np.array([0, 0, 30, 0, 0, 30] * 4, float)
    f = np.full_like(a, 10.0)
    assert monthly_fa(f, a) == 100
    dates = pd.Series(pd.bdate_range('2025-06-02', periods=40, freq='C', weekmask='Mon Tue Wed Thu Fri Sat')[:24])
    d = pd.DataFrame({'branch_id': 'X', 'sku_id': 'Y', 'window': 'W1', 'model': 'flat',
                      'date': dates, 'fc': f, 'actual': a})
    st = bucket_stats(d).iloc[0]
    assert np.isclose(st['day_sae'] / st['day_sa'], 80 / 60)
    assert np.isclose(st['week_sae'], 0) and np.isclose(st['month_sae'], 0)
    assert np.isclose(st['week_sf'], f.sum()) and np.isclose(st['month_sa'], a.sum())
    y = np.array([3, 5, 0, 7, 2, 1, 4, 6, 0, 8, 2, 0, 3], float)
    assert np.isclose(naive_scale(y, 6), np.mean(np.abs(y[6:] - y[:-6])))
    assert np.isclose(naive_scale(np.r_[0., 0., y], 1, True), np.mean(np.diff(y)**2))
    assert np.isnan(naive_scale(np.zeros(10)))
    tr = pd.DataFrame({'date': pd.bdate_range('2025-05-26', '2025-07-31', freq='C',
                                              weekmask='Mon Tue Wed Thu Fri Sat')})
    tr['true_demand'] = 1.0
    wk, mo = complete_buckets(tr)
    assert (wk == 6).all() and list(mo) == [25.0, 27.0]
    assert demand_class(np.array([5., 5, 5, 5, 5]))[0] == 'smooth'
    assert demand_class(np.array([0., 0, 50, 0, 0, 1, 0, 0, 90]))[0] == 'lumpy'
    print('selftest ok: flat forecast old FA = 100%, daily WAPE = 133%, weekly/monthly WAPE = 0%')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('eval')
    r.add_argument('--run', nargs='*', help='run names (default: all finished runs)')
    sub.add_parser('selftest')
    a = p.parse_args()
    {'eval': cmd_eval, 'selftest': cmd_selftest}[a.cmd](a)
