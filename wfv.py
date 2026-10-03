"""Building blocks of the notebook Copy_of_7sep_avian_forecasting_v3, used by retrain.py and metrics.py:
WFV windows, features, the notebook's FA% formula, the spike layer, Croston TSB, Theta and data loading."""
import sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

CLASS_PATH   = 'Dataset/series_classification_v3.csv'

WFV_WINDOWS = [
    {'name':'W1','train_end':'2025-05-31','fc_start':'2025-06-01','fc_end':'2025-06-30'},
    {'name':'W2','train_end':'2025-06-30','fc_start':'2025-07-01','fc_end':'2025-08-31'},
    {'name':'W3','train_end':'2025-08-31','fc_start':'2025-09-01','fc_end':'2025-09-30'},
    {'name':'W4','train_end':'2025-09-30','fc_start':'2025-10-01','fc_end':'2025-11-30'},
    {'name':'W5','train_end':'2025-11-30','fc_start':'2025-12-01','fc_end':'2026-01-31'},
    {'name':'W6','train_end':'2026-01-31','fc_start':'2026-02-01','fc_end':'2026-02-28'},
    {'name':'W7','train_end':'2026-02-28','fc_start':'2026-03-01','fc_end':'2026-03-31'},
    {'name':'W8','train_end':'2026-03-31','fc_start':'2026-04-01','fc_end':'2026-05-31'},
]

QUANTILES    = [0.50, 0.60, 0.75]
MIN_TRAIN    = 90
SPIKE_THRS   = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
LAG_COLS   = ['true_demand_lag_7','true_demand_lag_14','true_demand_lag_28']
CAL_COLS   = ['day_of_week','month','week_of_month','day_of_month','week_of_year','is_saturday']
PROMO_COLS = ['is_promo','discount_pct','is_promo_voucher','is_promo_discount',
              'is_promo_free_item','voucher_nominal']
HOL_COLS   = ['is_holiday','is_bridge_day','is_before_holiday','is_after_holiday']
OTHER_COLS = ['season_ordinal','is_cat_event','is_bb_event','is_mebel_event','branch_enc','sku_enc']
ALL_FEAT   = LAG_COLS+CAL_COLS+PROMO_COLS+HOL_COLS+OTHER_COLS


def monthly_fa(fc, actual):
    n = min(len(fc), len(actual))
    s = actual[:n].sum()
    if s == 0: return 0.0
    return max(0, 100-abs(fc[:n].sum()-s)/s*100)

def get_p90(s_train):
    pos = s_train['true_demand'].values[s_train['true_demand'].values>0]
    return float(np.percentile(pos,90)) if len(pos)>=10 else 72.0

def rolling_fc(hist, n_fc, w, func='mean'):
    ext = list(hist)
    out = []
    for _ in range(n_fc):
        v = np.mean(ext[-w:]) if func=='mean' else np.median(ext[-w:])
        out.append(v); ext.append(v)
    return np.clip(np.array(out),0,None)


SPIKE_FEAT = [
    'dow','wom','dom','is_monday','is_friday','is_saturday',
    'is_week1','is_week2','is_week3','is_week4','is_week5',
    'is_month_end','is_promo','discount_pct',
    'is_holiday','is_bridge_day','is_before_holiday','is_after_holiday',
    'promo_x_monday','promo_x_week_end',
]
SPIKE_RULES = {
    'is_week5':4.0,'is_month_end':3.0,'promo_x_week_end':3.0,
    'is_monday':2.0,'is_friday':1.5,'is_week4':1.5,
    'is_saturday':-0.5,'is_week1':-0.5,'is_week2':-0.5,'is_week3':-0.5,
}

def build_spike_feats(s):
    d = s.copy()
    d['dow']           = d['date'].dt.dayofweek
    d['wom']           = d['date'].dt.day.apply(lambda x:(x-1)//7+1)
    d['dom']           = d['date'].dt.day
    d['is_monday']     = (d['dow']==0).astype(int)
    d['is_friday']     = (d['dow']==4).astype(int)
    d['is_saturday']   = (d['dow']==5).astype(int)
    for w in range(1,6):
        d[f'is_week{w}'] = (d['wom']==w).astype(int)
    d['is_month_end']  = (d['dom']>=25).astype(int)
    d['is_promo']      = d['is_promo'].fillna(0).astype(int)
    d['discount_pct']  = d['discount_pct'].fillna(0)
    for c in ['is_holiday','is_bridge_day','is_before_holiday','is_after_holiday']:
        d[c] = d[c].fillna(0).astype(int)
    d['promo_x_monday']   = d['is_promo']*d['is_monday']
    d['promo_x_week_end'] = d['is_promo']*d['is_month_end']
    return d

def _rule_proba(s, avail_sf, base_rate):
    scores = s[avail_sf].apply(
        lambda r: sum(float(r.get(f,0))*w for f,w in SPIKE_RULES.items()
                      if f in r.index), axis=1).values
    sc_range = scores.max()-scores.min()
    return np.clip(base_rate+((scores-0)/sc_range*3*base_rate)
                   if sc_range>0 else np.full(len(s),base_rate), 0.01, 0.99)

def predict_spike_proba(s_target, s_same, avail_sf, base_rate):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    X_fc = s_target[avail_sf].fillna(0).values
    X_tr = s_same[avail_sf].fillna(0).values
    proba_list = [_rule_proba(s_target, avail_sf, base_rate)]
    y_tr = s_same['is_spike'].values
    if y_tr.sum()>=2 and len(X_tr)>=10:
        try:
            sc = StandardScaler()
            lr = LogisticRegression(class_weight='balanced', max_iter=300, C=0.1, random_state=42)
            lr.fit(sc.fit_transform(X_tr), y_tr)
            proba_list.append(lr.predict_proba(sc.transform(X_fc))[:,1])
        except: pass
        try:
            rf = RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced',
                                        min_samples_leaf=2, random_state=42)
            rf.fit(X_tr,y_tr)
            proba_list.append(rf.predict_proba(X_fc)[:,1])
        except: pass
    return np.mean(proba_list, axis=0)

def apply_override(fc_base, spike_p, spike_mag, thr):
    fc = fc_base.copy()
    for i in range(len(fc)):
        if spike_p[i]>=thr:
            fc[i] = fc_base[i]*(1-spike_p[i])+spike_mag*spike_p[i]
    return np.clip(fc,0,None)


def fit_croston_tsb(series_vals, n_fc):
    alpha, beta = 0.1, 0.1
    nonzero_idx = np.where(series_vals>0)[0]
    if len(nonzero_idx)==0:
        return np.zeros(n_fc)
    d_hat = series_vals[nonzero_idx[0]]
    p_hat = 1.0/max(nonzero_idx[0]+1,1)
    for i in range(1,len(series_vals)):
        if series_vals[i]>0:
            d_hat = alpha*series_vals[i]+(1-alpha)*d_hat
            p_hat = beta*1.0+(1-beta)*p_hat
        else:
            p_hat = (1-beta)*p_hat
    return np.full(n_fc, max(d_hat*p_hat,0))

def fit_theta(series_vals, n_fc):
    n = len(series_vals)
    if n < 4:
        return np.full(n_fc, series_vals.mean())
    x, xf = np.arange(n), np.arange(n, n+n_fc)
    coef = np.polyfit(x, series_vals, 1)
    trend_fc = np.polyval(coef, xf)
    theta2 = 2*series_vals - np.polyval(coef, x)
    level  = theta2[0]
    for v in theta2[1:]:
        level = 0.1*v+0.9*level
    return np.clip((trend_fc+np.full(n_fc, level))/2.0,0,None)


def load(path, feat_list):
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv(path, parse_dates=['date'], low_memory=False)
    df['branch_id'] = df['branch_id'].astype(str)
    df['sku_id']    = df['sku_id'].astype(str)
    df = df.sort_values(['branch_id','sku_id','date']).reset_index(drop=True)
    if 'branch_enc' not in df.columns:
        df['branch_enc'] = LabelEncoder().fit_transform(df['branch_id'])
        df['sku_enc']    = LabelEncoder().fit_transform(df['sku_id'])
    if 'true_demand' not in df.columns:
        sys.exit(f'{path}: no true_demand column')
    return df, [c for c in feat_list if c in df.columns]


def fmt_time(s):
    return f'{int(s//3600):02d}:{int(s%3600//60):02d}:{int(s%60):02d}'


def selftest():
    a = np.array([10., 0., 10.])
    assert monthly_fa(a, a) == 100
    assert monthly_fa(a*1.5, a) == 50
    assert monthly_fa(a, np.zeros(3)) == 0
    assert monthly_fa(a*5, a) == 0
    assert np.allclose(rolling_fc(np.array([1., 2., 3.]), 2, 3), [2., 7/3])
    assert (fit_croston_tsb(np.zeros(5), 3) == 0).all()
    assert fit_croston_tsb(np.array([4., 4., 4., 4.]), 2)[0] > 0
    print('selftest ok')

if __name__ == '__main__':
    selftest()
