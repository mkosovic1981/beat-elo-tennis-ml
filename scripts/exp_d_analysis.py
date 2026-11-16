"""
Helpers for notebook 14 (Exp D, TabPFN v2). Contains NO data -- everything is read from
<workdir>/data (features.parquet, c5_preds_fold*.parquet) and <workdir>/exp_d/<run>/ at run time.
Re-uses corp / ece_pp (summarize_metrics.py) and boot_idx / ci (eval_c5.py).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_c5 import boot_idx, ci  # noqa: E402
from summarize_metrics import corp, ece_pp  # noqa: E402

K = ['SusretId', 'Perspective']
VARIANTS = ['D-Aprime', 'D-Bprime', 'D-Bprime-par']
CTXS = ['last10k', 'random10k']
SEEDS = (42, 1, 2)
BUCKETS = [('<=30', -1, 30), ('31-90', 30, 90), ('91-180', 90, 180), ('181-365', 180, 365), ('>365', 365, 1e9)]


def load_folds(workdir):
    """Return {fold: DataFrame} with ELO / C3 / C3ens / D-run probabilities joined on (SusretId, Perspective)."""
    W = Path(workdir)
    D, E = W / 'data', W / 'exp_d'
    feat = pd.read_parquet(D / 'features.parquet')[K + ['elo_diff']]
    feat['p_ELO'] = 1 / (1 + 10 ** (-feat['elo_diff'] / 400))
    dp = {}
    for d in sorted(E.iterdir()):
        assert (d / 'DONE').exists(), f'unfinished run {d.name}'
        m = re.match(r'f(\d)_(.+)', d.name)
        k, key = int(m[1]), m[2]
        df = pd.read_parquet(next(d.glob('test_predictions_d_fold*.parquet')))
        dp.setdefault(k, {})[key] = df.set_index(K)[f'p_cal__{key}']
        dp[k].setdefault('_meta', df.set_index(K)[['Label', 'liga_tip']])
    out = {}
    for k in sorted(dp):
        c5 = pd.read_parquet(D / f'c5_preds_fold{k}.parquet')
        b = c5[K + ['Datum', 'Label', 'liga_tip', '_rest_days_p', 'p_C3_calibrated']].rename(columns={'p_C3_calibrated': 'p_C3'})
        b = b.merge(feat[K + ['p_ELO']], on=K, how='left')
        ps = [c5.set_index(K)['p_C3_calibrated']]
        for s in (1, 2, 3, 4):
            p = D / f'c5_preds_fold{k}_seed{s}.parquet'
            if p.exists():
                ps.append(pd.read_parquet(p).set_index(K)['p_C3_calibrated'])
        b = b.merge(pd.concat(ps, axis=1).mean(axis=1).rename('p_C3ens'), left_on=K, right_index=True)
        for key, s in dp[k].items():
            if key != '_meta':
                b = b.merge(s.rename('p_' + key), left_on=K, right_index=True, how='left')
                assert b['p_' + key].notna().all()
        chk = b.merge(dp[k]['_meta'], left_on=K, right_index=True, suffixes=('', '_d'))
        assert (chk.Label == chk.Label_d).all()
        out[k] = b
    return out


def met(p, y):
    mcb, dsc, _ = corp(p, y)
    return dict(brier=np.mean((p - y) ** 2), logloss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                auc=roc_auc_score(y, p), acc=np.mean((p > 0.5) == y), ece=ece_pp(p, y), mcb=mcb, dsc=dsc)


def paired_dbrier(pa, pb, y, boots):
    la, lb = (pa - y) ** 2, (pb - y) ** 2
    return ci(np.array([(la[i] - lb[i]).mean() for i in boots]))


def main_split_table(f4):
    """Adds p_<variant>__<ctx>__ens columns to f4 (in place) and returns the per-config summary table."""
    y = f4.Label.values
    rows = []
    for v in VARIANTS:
        for c in CTXS:
            cols = [f'p_{v}__{c}__s{s}' for s in SEEDS]
            ms = [met(f4[x].values, y) for x in cols]
            f4[f'p_{v}__{c}__ens'] = f4[cols].mean(axis=1)
            me = met(f4[f'p_{v}__{c}__ens'].values, y)
            r = dict(variant=v, ctx=c, brier_mean=np.mean([m['brier'] for m in ms]),
                     brier_min=min(m['brier'] for m in ms), brier_max=max(m['brier'] for m in ms))
            r.update({'ens_' + k: x for k, x in me.items()})
            rows.append(r)
    return pd.DataFrame(rows)


def corp_delta(pa, pb, y, boots, n=1000):
    o = []
    for i in boots[:n]:
        ma, da, _ = corp(pa[i], y[i]); mb, db, _ = corp(pb[i], y[i]); o.append((ma - mb, da - db))
    o = np.array(o)
    return {n_: (o[:, j].mean(), *np.percentile(o[:, j], [2.5, 97.5])) for j, n_ in enumerate(['dMCB', 'dDSC'])}


def rest_table(d, pcol):
    rows = []
    for n, lo, hi in BUCKETS:
        mk = ((d._rest_days_p > lo) & (d._rest_days_p <= hi)).values
        if mk.sum():
            s = d[mk]; yy = s.Label.values
            rows.append(dict(rest_days=n, n_rows=len(s), actual=yy.mean(), pred_ELO=s.p_ELO.mean(), pred_C3=s.p_C3.mean(),
                             pred_D=s[pcol].mean(), brier_ELO=np.mean((s.p_ELO - yy) ** 2),
                             brier_C3=np.mean((s.p_C3 - yy) ** 2), brier_D=np.mean((s[pcol] - yy) ** 2)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- notebook 15 (TabPFN-3.5, research-only)
def add_d35(B, workdir):
    """Join TabPFN-3.5 predictions (read from <workdir>/exp_d35) into the fold frames of load_folds(), in place.
    Adds p_d35_full (folds 1-4); on fold 4 also p_d35_10k and p_d35_par_full."""
    E = Path(workdir) / 'exp_d35'
    want = {'d35_full': ('D-Bprime__ctxfull__s42', 'd35__D-Bprime__lastfull__s42', (1, 2, 3, 4)),
            'd35_10k': ('D-Bprime__ctx10000__s42', 'd35__D-Bprime__last10k__s42', (4,)),
            'd35_par_full': ('D-Bprime-par__ctxfull__s42', 'd35__D-Bprime-par__lastfull__s42', (4,))}
    for name, (dirkey, colkey, folds) in want.items():
        for k in folds:
            d = E / f'f{k}_{dirkey}'
            assert (d / 'DONE').exists(), f'unfinished run {d.name}'
            df = pd.read_parquet(next(d.glob('test_predictions_d35_fold*.parquet')))
            s = df.set_index(K)[f'p_cal__{colkey}'].rename('p_' + name)
            B[k] = B[k].merge(s, left_on=K, right_index=True, how='left')
            assert B[k]['p_' + name].notna().all()
    return B


def pool_folds(B):
    """Concatenate folds with fold-unique SusretId (a match is resampled as a unit, folds stay disjoint)."""
    return pd.concat([B[k].assign(SusretId=B[k].SusretId.map(lambda s, k=k: f'{k}_{s}')) for k in B], ignore_index=True)


def brier_ci(p, y, boots):
    l = (p - y) ** 2
    return ci(np.array([l[i].mean() for i in boots]))


def reliability(p, y, bins=10):
    """Equal-mass bins -> (mean predicted, observed rate)."""
    o = np.argsort(p, kind='stable')
    parts = np.array_split(o, bins)
    return np.array([p[i].mean() for i in parts]), np.array([y[i].mean() for i in parts])


# ---------------------------------------------------------------- notebook 16 (GRU + tabular blends)
def add_xgb(B, workdir):
    """Join XGBoost B' predictions (<workdir>/exp_xgb, from run_walkforward_xgb_bprime.py) into the fold frames, in place.
    Adds p_xgb (folds 1-4)."""
    E = Path(workdir) / 'exp_xgb'
    for k in B:
        df = pd.read_parquet(E / f'test_predictions_xgb_bprime_fold{k}.parquet')
        assert not df.duplicated(K).any()
        B[k] = B[k].merge(df.set_index(K)['p_cal_Bprime_xgb'].rename('p_xgb'), left_on=K, right_index=True, how='left')
        assert B[k]['p_xgb'].notna().all()
    return B


def add_blends(B, v2='p_D-Bprime__last10k__s42', v35='p_d35_full'):
    """Add fixed-weight blend columns in place (weights chosen in advance -- nothing is fitted on test data).
    Needs p_C3, p_C3ens, p_xgb, v2 (and v35 for the research-only blend). Returns {column: label}."""
    for k, d in B.items():
        d['b_C3_xgb'] = 0.5 * d.p_C3 + 0.5 * d.p_xgb
        d['b_C3ens_xgb'] = 0.5 * d.p_C3ens + 0.5 * d.p_xgb
        d['b_C3_v2'] = 0.5 * d.p_C3 + 0.5 * d[v2]
        d['b_C3ens_v2'] = 0.5 * d.p_C3ens + 0.5 * d[v2]
        d['b_3way'] = (d.p_C3ens + d[v2] + d.p_xgb) / 3
        if v35 in d:
            d['b_C3ens_v35'] = 0.5 * d.p_C3ens + 0.5 * d[v35]
    return {'b_C3_xgb': '0.5 C3 + 0.5 XGB', 'b_C3ens_xgb': '0.5 C3ens + 0.5 XGB', 'b_C3_v2': '0.5 C3 + 0.5 TabPFN v2',
            'b_C3ens_v2': '0.5 C3ens + 0.5 TabPFN v2', 'b_3way': '(C3ens + v2 + XGB) / 3',
            'b_C3ens_v35': '0.5 C3ens + 0.5 TabPFN-3.5 (research-only)'}
