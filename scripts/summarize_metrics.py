"""
Headline test-split metrics (Brier / log-loss / accuracy / AUC) for every experiment found in a
workspace data dir, plus a 2,000-resample paired bootstrap (seed 42, resampling MATCHES, both
perspective rows together) of each model's Brier delta vs the ELO baseline and vs Exp C3.

  python summarize_metrics.py <workdir>/data [--out summary.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

MODELS = [  # (label, file, column)
    ('Exp 0 ELO', 'test_predictions_4way', 'p_exp0_elo'),
    ('Exp A', 'test_predictions_4way', 'p_expA_calibrated'),
    ('Exp B', 'test_predictions_4way', 'p_expB_calibrated'),
    ('Exp C (ELO-free GRU)', 'test_predictions_4way', 'p_expC_calibrated'),
    ("Exp A'", 'test_predictions_trajectory', 'p_expAprime_calibrated'),
    ("Exp B'", 'test_predictions_trajectory', 'p_expBprime_calibrated'),
    ('Exp C2', 'test_predictions_c2', None),
    ('Exp C3', 'test_predictions_c3', None),
    ('Exp C4', 'test_predictions_c4', None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('--out')
    a = ap.parse_args()
    d = Path(a.data_dir)
    base = pd.read_parquet(d / 'test_predictions_4way.parquet').reset_index(drop=True)
    key = base[['SusretId', 'Perspective']]
    y = base['Label'].values
    preds = {}
    for label, f, col in MODELS:
        fp = d / f'{f}.parquet'
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        if col is None:
            col = next(c for c in df.columns if c.startswith('p_') and c.endswith('_calibrated'))
        df = key.merge(df[['SusretId', 'Perspective', col]], on=['SusretId', 'Perspective'], how='left')
        assert df[col].notna().all(), (label, 'missing rows')
        preds[label] = df[col].values.astype(float)
    # bootstrap over matches
    mid = pd.factorize(base['SusretId'])[0]
    n = mid.max() + 1
    rows_by = [np.where(mid == i)[0] for i in range(n)]
    rng = np.random.default_rng(42)
    boots = [np.concatenate([rows_by[i] for i in rng.integers(0, n, n)]) for _ in range(2000)]
    def brier_boot(p):
        return np.array([np.mean((p[b] - y[b]) ** 2) for b in boots])
    bb = {k: brier_boot(p) for k, p in preds.items()}
    out = []
    for k, p in preds.items():
        row = dict(model=k, brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
                   acc=float(accuracy_score(y, p > 0.5)), auc=float(roc_auc_score(y, p)))
        for ref in ('Exp 0 ELO', 'Exp C3'):
            if ref in bb and k != ref:
                dd = bb[k] - bb[ref]
                row[f'dBrier_vs_{ref}'] = [float(np.mean(dd)), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
        out.append(row)
    print(pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in out]).round(4).to_string(index=False))
    for r in out:
        for k, v in r.items():
            if isinstance(v, list):
                print(f"  {r['model']}: {k} = {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]")
    print(f'n test matches: {n}, rows: {len(y)}, test range: {base.Datum.min()} .. {base.Datum.max()}')
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
