"""
Helpers for notebook 18 (fair comparison of all models: raw vs calibrated, seed parity, symmetrised versions).
Contains NO data -- everything is read from <workdir>/data, <workdir>/exp_d, <workdir>/exp_d35 and
<workdir>/exp_fair (written by run_walkforward_xgb_seeds.py) at run time.

Column naming in the fold frames returned by `load_fair`:
  p_ELO              ELO win probability 1 / (1 + 10 ** (-elo_diff / 400))           (raw, no fitting)
  p_ELOcal           isotonic recalibration of that probability, fitted on the calibration slice of the fold
  p_ELOplatt         logistic (Platt) recalibration on elo_diff, same slice
  p_C3raw / p_C3     GRU C3 raw / calibrated (seed 42);  p_C3ens_raw / p_C3ens  5-seed mean of raw / of calibrated
  p_v2raw / p_v2cal  TabPFN v2 (10k context, seed 42);   p_v35raw / p_v35cal   TabPFN-3.5 (full context, research-only)
  p_xgb_raw / p_xgb_cal        XGBoost B' (5 seeds: mean of the raw / of the per-seed calibrated probabilities)
  p_xgb_raw_s42 / p_xgb_cal_s42  the same model, seed 42 only
  p_xgbpar_*                   the same for the pair-only variant B'-par
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_d_analysis import K, acc_ci, acc_score, brier_ci, load_folds, pool_folds, symmetrize  # noqa: E402,F401
from run_walkforward_xgb_bprime import carve  # noqa: E402
from eval_c5 import boot_idx  # noqa: E402,F401

XGB_SEEDS = (42, 1, 2, 3, 4)
V2_KEY = 'D-Bprime__last10k__s42'
V35_KEY = ('D-Bprime__ctxfull__s42', 'd35__D-Bprime__lastfull__s42')


def _fold_ids(workdir, n_blocks=5):
    """Train / test match ids per fold -- the same expanding-window rule as tabpfn_data / wf_c3c4_common."""
    sym = pd.read_parquet(Path(workdir) / 'data' / 'features.parquet', columns=['SusretId', 'Datum'])
    mo = (sym.drop_duplicates('SusretId').sort_values(['Datum', 'SusretId'], kind='mergesort').reset_index(drop=True))
    edges = np.linspace(0, len(mo), n_blocks + 1).astype(int)
    blocks = [mo.iloc[edges[i]:edges[i + 1]] for i in range(n_blocks)]
    return mo, {k: (set(pd.concat(blocks[:k])['SusretId']), set(blocks[k]['SusretId'])) for k in range(1, n_blocks)}


def add_elo_calibrated(B, workdir):
    """ELO recalibration per fold on the SAME calibration slice the other models use (chronologically last 15 % of the
    fold's train matches, `carve`), applied to the fold's test rows. Isotonic (main) and logistic on elo_diff (Platt)."""
    mo, ids = _fold_ids(workdir)
    feat = pd.read_parquet(Path(workdir) / 'data' / 'features.parquet', columns=K + ['elo_diff', 'Label'])
    feat['p_ELO'] = 1 / (1 + 10 ** (-feat['elo_diff'] / 400))
    for k in B:
        train_ids, test_ids = ids[k]
        assert set(B[k].SusretId) == test_ids, f'fold {k}: test matches differ'
        _, _, calib_ids = carve(mo, train_ids)
        cal = feat[feat.SusretId.isin(calib_ids)]
        ycal = cal['Label'].values
        assert len(ycal) > 0 and not (calib_ids & test_ids)
        iso = IsotonicRegression(out_of_bounds='clip').fit(cal['p_ELO'].values, ycal)
        platt = LogisticRegression(C=1e6).fit(cal[['elo_diff']].values, ycal)
        te = B[k][K].merge(feat, on=K, how='left')
        assert (te['SusretId'].values == B[k]['SusretId'].values).all() and te['elo_diff'].notna().all()
        B[k]['p_ELOcal'] = iso.predict(te['p_ELO'].values)
        B[k]['p_ELOplatt'] = platt.predict_proba(te[['elo_diff']].values)[:, 1]
        assert np.allclose(B[k]['p_ELO'], te['p_ELO'])
    return B


def add_tabpfn_raw_cal(B, workdir):
    """p_v2raw / p_v2cal (exp_d, 10k context) and p_v35raw / p_v35cal (exp_d35, full context) for folds 1-4.
    The TabPFN outputs are symmetrised by the run scripts (raw = symmetrised model output)."""
    W = Path(workdir)
    for k in B:
        for d, key, raw_n, cal_n in ((W / 'exp_d' / f'f{k}_{V2_KEY}', V2_KEY, 'p_v2raw', 'p_v2cal'),
                                     (W / 'exp_d35' / f'f{k}_{V35_KEY[0]}', V35_KEY[1], 'p_v35raw', 'p_v35cal')):
            assert (d / 'DONE').exists(), f'unfinished run {d.name}'
            df = pd.read_parquet(next(d.glob('test_predictions_d*_fold*.parquet'))).set_index(K)
            for src, name in ((f'p_raw__{key}', raw_n), (f'p_cal__{key}', cal_n)):
                B[k] = B[k].merge(df[src].rename(name), left_on=K, right_index=True, how='left')
                assert B[k][name].notna().all()
    return B


def add_xgb_seeds(B, workdir, seeds=XGB_SEEDS):
    """5-seed XGBoost B' and B'-par: *_raw / *_cal = mean over seeds, *_s42 = seed 42 only."""
    E = Path(workdir) / 'exp_fair'
    for stem, name in (('bprime', 'xgb'), ('bprimepar', 'xgbpar')):
        for k in B:
            df = pd.read_parquet(E / f'test_predictions_xgbseeds_{stem}_fold{k}.parquet').set_index(K)
            assert not df.index.duplicated().any()
            parts = {f'p_{name}_raw': df[[f'p_raw_s{s}' for s in seeds]].mean(axis=1),
                     f'p_{name}_cal': df[[f'p_cal_s{s}' for s in seeds]].mean(axis=1),
                     f'p_{name}_raw_s42': df['p_raw_s42'], f'p_{name}_cal_s42': df['p_cal_s42']}
            for c, s in parts.items():
                B[k] = B[k].merge(s.rename(c), left_on=K, right_index=True, how='left')
                assert B[k][c].notna().all()
    return B


def load_fair(workdir):
    B = load_folds(workdir)
    B = add_elo_calibrated(B, workdir)
    B = add_tabpfn_raw_cal(B, workdir)
    B = add_xgb_seeds(B, workdir)
    return B


def add_sym(B, cols):
    """Add `<col>_sym` = (p_AB + 1 - p_BA) / 2 for every column, per fold (a match stays inside its fold)."""
    for k, d in B.items():
        for c in cols:
            d[c + '_sym'] = symmetrize(d, c)
    return B


def add_fair_blends(B):
    """Fixed 0.5 / 0.5 blends (chosen in advance; nothing is fitted on test data) and their symmetrised versions.
    GRU part is always the raw 5-seed ensemble, the XGBoost part raw or calibrated, B' or B'-par."""
    labels = {}
    for tag, x in (('xgb', "B'"), ('xgbpar', "B'-par")):
        for v in ('raw', 'cal'):
            col = f'b_C3ensraw_{tag}_{v}'
            for d in B.values():
                d[col] = 0.5 * d['p_C3ens_raw'] + 0.5 * d[f'p_{tag}_{v}']
            labels[col] = f"0.5 C3ens raw + 0.5 XGB {x} ({v})"
    for tag, x in (('xgb', "B'"), ('xgbpar', "B'-par")):  # all-calibrated reference
        col = f'b_C3enscal_{tag}_cal'
        for d in B.values():
            d[col] = 0.5 * d['p_C3ens'] + 0.5 * d[f'p_{tag}_cal']
        labels[col] = f"0.5 C3ens cal + 0.5 XGB {x} (cal)"
    add_sym(B, list(labels))
    labels.update({c + '_sym': l + ' / sym' for c, l in list(labels.items())})
    return labels


def fair_table(pool, specs, boots, ref_raw='p_ELO', ref_cal='p_ELOcal'):
    """specs: list of (model, variant, sym, column). One row each: Brier [CI], % Brier vs ELO raw and vs ELO calibrated,
    accuracy [CI] (both perspectives, a tie = half a point), AUC."""
    y = pool.Label.values
    b_raw = np.mean((pool[ref_raw].values - y) ** 2)
    b_cal = np.mean((pool[ref_cal].values - y) ** 2)
    rows = []
    for model, variant, sym, c in specs:
        p = pool[c].values
        b = np.mean((p - y) ** 2)
        _, bl, bh = brier_ci(p, y, boots)
        a, al, ah = acc_ci(p, y, boots)
        rows.append({'Model': model, 'Variant': variant, 'Sym': sym, 'Brier': b, 'lo': bl, 'hi': bh,
                     '% vs ELO raw': 100 * (1 - b / b_raw), '% vs ELO cal': 100 * (1 - b / b_cal),
                     'Acc %': 100 * acc_score(p, y).mean(), 'Acc lo': 100 * al, 'Acc hi': 100 * ah,
                     'AUC': roc_auc_score(y, p)})
    return pd.DataFrame(rows)


def fold_brier(B, c):
    return [float(np.mean((B[k][c] - B[k].Label) ** 2)) for k in sorted(B)]


def folds_better(B, a, b):
    return sum(x < y for x, y in zip(fold_brier(B, a), fold_brier(B, b)))


def add_full_grid(B, workdir):
    """Full blend grid under fair conditions: every component RAW (and symmetrised, `p_sym = (p_AB + 1 - p_BA) / 2`),
    XGBoost = 5-seed ensemble, fixed weights chosen in advance. Returns {column: (label, research_only, no_match_context)}."""
    W = Path(workdir)
    for k in B:  # TabPFN v2 D-B'-par (no match context), raw, 10k context, seed 42
        key = 'D-Bprime-par__last10k__s42'
        d = W / 'exp_d' / f'f{k}_{key}'
        assert (d / 'DONE').exists(), f'unfinished run {d.name}'
        df = pd.read_parquet(next(d.glob('test_predictions_d*_fold*.parquet'))).set_index(K)
        B[k] = B[k].merge(df[f'p_raw__{key}'].rename('p_v2par_raw'), left_on=K, right_index=True, how='left')
        assert B[k]['p_v2par_raw'].notna().all()
    comp = {'C3': 'p_C3raw_sym', 'v2par': 'p_v2par_raw', 'C3ens': 'p_C3ens_raw_sym', 'XGB': 'p_xgb_raw_sym', 'XGBpar': 'p_xgbpar_raw_sym',
            'v2': 'p_v2raw', 'v35': 'p_v35raw'}
    lab = {'C3': 'C3 raw (1 seed)', 'C3ens': 'C3ens raw (5 seeds)', 'XGB': "XGB B' (5s)", 'XGBpar': "XGB B'-par (5s)",
           'v2': 'TabPFN v2 raw', 'v2par': "TabPFN v2 B'-par raw", 'v35': 'TabPFN-3.5 raw'}
    spec = []
    for g in ('C3', 'C3ens'):
        for t in ('XGB', 'XGBpar', 'v2', 'v35', 'v2par'):
            spec.append(((g, t), (0.5, 0.5)))
    spec += [(('C3ens', 'XGBpar', 'v2'), (1 / 3,) * 3), (('C3ens', 'XGB', 'v2'), (1 / 3,) * 3),
             (('C3ens', 'v2', 'v35'), (1 / 3,) * 3), (('XGB', 'v2'), (0.5, 0.5))]
    singles = [('v2par',)]
    spec += [(p, (1.0,)) for p in singles]
    out = {}
    for parts, w in spec:
        col = 'g_' + '_'.join(parts)
        for d in B.values():
            d[col] = sum(wi * d[comp[p]] for p, wi in zip(parts, w))
        wtxt = (lab[parts[0]] if len(parts) == 1 else
                ' + '.join((f'{wi:.2f} ' if len(parts) == 2 else '1/3 ') + lab[p] for p, wi in zip(parts, w)))
        out[col] = (wtxt, 'v35' in parts, 'XGBpar' in parts or 'v2par' in parts)
    return out
