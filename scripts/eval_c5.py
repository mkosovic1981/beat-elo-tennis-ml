"""
Evaluation of run_c5.py output (reads <workdir>/data/c5_preds_fold*.parquet).
Brier / ECE (p.b.) / CORP MCB, DSC (imported from summarize_metrics.py), paired bootstrap by SusretId
(2000 resamples, seed 42, both perspective rows together), walk-forward mean+-std, tables by rest length and gender.

  python eval_c5.py --workdir <workdir> [--best C5b] [--wf-variants C3,C5b]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_metrics import corp, ece_pp  # noqa: E402

BUCKETS = [('<=30', -1, 30), ('31-90', 30, 90), ('91-180', 90, 180), ('181-365', 180, 365), ('>365', 365, 1e9)]


def boot_idx(sus, n_boot=2000, seed=42):
    mid, uniq = pd.factorize(sus)
    rows_by = [np.flatnonzero(mid == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    n = len(uniq)
    return [np.concatenate([rows_by[i] for i in rng.integers(0, n, n)]) for _ in range(n_boot)]


def ci(x):
    return np.mean(x), np.percentile(x, 2.5), np.percentile(x, 97.5)


def brier_ci_and_delta(df, a, b, mask=None):
    """Brier of a, b and paired delta (a-b) with bootstrap CI, on rows `mask`."""
    d = df if mask is None else df[mask].reset_index(drop=True)
    y = d['Label'].values
    la, lb = (d[f'p_{a}_calibrated'].values - y) ** 2, (d[f'p_{b}_calibrated'].values - y) ** 2
    boots = boot_idx(d['SusretId'].values)
    ba = np.array([la[i].mean() for i in boots]); bb = np.array([lb[i].mean() for i in boots])
    return la.mean(), ci(ba), lb.mean(), ci(bb), (la.mean() - lb.mean()), ci(ba - bb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--best')
    ap.add_argument('--wf-variants', default='')
    a = ap.parse_args()
    data = Path(a.workdir) / 'data'
    folds = {int(p.stem.split('fold')[1]): pd.read_parquet(p) for p in sorted(data.glob('c5_preds_fold[0-9].parquet'))}
    f4 = folds[4]
    variants = [c[2:-11] for c in f4.columns if c.startswith('p_') and c.endswith('_calibrated')]
    print('variants on fold 4 (main split, test from', f4['Datum'].min().date(), '):', variants)
    y = f4['Label'].values
    n_m = f4['SusretId'].nunique()
    print(f'test matches {n_m}, rows {len(f4)}')
    saved = pd.read_parquet(data / 'test_predictions_c3.parquet')
    m = f4[['SusretId', 'Perspective']].merge(saved, on=['SusretId', 'Perspective'], how='left')
    print(f"saved C3 (notebook 10) Brier on same rows: {np.mean((m['p_expC3_calibrated'].values - y) ** 2):.5f}")

    boots = boot_idx(f4['SusretId'].values)
    base_l = (f4['p_C3_calibrated'].values - y) ** 2
    print('\n## Main split (fold 4)')
    print('| varijanta | Brier | ECE p.b. | MCB | DSC | dBrier vs C3 [95% CI] |')
    print('|---|---|---|---|---|---|')
    rows = {}
    for v in variants:
        p = f4[f'p_{v}_calibrated'].values
        mcb, dsc, _ = corp(p, y)
        l = (p - y) ** 2
        if v == 'C3':
            dtxt = '-'
        else:
            dd = np.array([(l[i] - base_l[i]).mean() for i in boots])
            mu, lo, hi = ci(dd)
            dtxt = f'{mu:+.5f} [{lo:+.5f}, {hi:+.5f}]'
        rows[v] = l.mean()
        print(f'| {v} | {l.mean():.5f} | {ece_pp(p, y):.2f} | {mcb:.5f} | {dsc:.5f} | {dtxt} |')

    best = a.best or min((v for v in variants if v != 'C3'), key=lambda v: rows[v])
    print(f'\nbest C5 on main split: {best}')

    # ---- walk-forward ----
    print('\n## Walk-forward (Brier per fold)')
    wfv = [v for v in (a.wf_variants.split(',') if a.wf_variants else variants)]
    hdr = '| varijanta | ' + ' | '.join(f'fold {k}' for k in sorted(folds)) + ' | mean +- std |'
    print(hdr); print('|' + '---|' * (len(folds) + 2))
    per = {}
    for v in wfv:
        vals = []
        for k in sorted(folds):
            d = folds[k]
            vals.append(np.mean((d[f'p_{v}_calibrated'].values - d['Label'].values) ** 2) if f'p_{v}_calibrated' in d else np.nan)
        per[v] = vals
        print(f'| {v} | ' + ' | '.join(f'{x:.5f}' for x in vals) + f' | {np.nanmean(vals):.5f} +- {np.nanstd(vals, ddof=1):.5f} |')
    for v in wfv:
        if v == 'C3':
            continue
        diffs = [per[v][i] - per['C3'][i] for i in range(len(folds))]
        print(f'  d({v}-C3) per fold: ' + ', '.join(f'{x:+.5f}' for x in diffs) + f'  | mean {np.nanmean(diffs):+.5f}, folds better: {sum(x < 0 for x in diffs)}/{len(diffs)}')
    pooled = pd.concat([d for k, d in folds.items() if all(f'p_{v}_calibrated' in d for v in ('C3', best))], ignore_index=True)
    pooled['SusretId'] = pooled['fold'].astype(str) + '_' + pooled['SusretId'].astype(str)
    mu_a, cia, mu_b, cib, dl, cid = brier_ci_and_delta(pooled, best, 'C3')
    print(f'  pooled walk-forward test rows ({pooled["SusretId"].nunique()} matches): {best} {mu_a:.5f}, C3 {mu_b:.5f}, '
          f'd = {dl:+.5f} [{cid[1]:+.5f}, {cid[2]:+.5f}]')

    # ---- by rest bucket / gender: main split and pooled WF ----
    for title, d in (('Main split (fold 4)', f4), ('Pooled walk-forward folds with both models', pooled)):
        print(f'\n## {title}: by rest length (player perspective rows), C3 vs {best}')
        print(f'| pauza | n | stvarno | pred C3 | pred {best} | Brier C3 | Brier {best} | d [95% CI] |')
        print('|---|---|---|---|---|---|---|---|')
        r = d['_rest_days_p']
        for name, lo, hi in BUCKETS + [('prvi meč (n/a)', None, None)]:
            mask = r.isna().values if lo is None else ((r > lo) & (r <= hi)).values
            if mask.sum() == 0:
                continue
            s = d[mask]
            _, _, _, _, dl, cid = brier_ci_and_delta(d, best, 'C3', mask)
            yy = s['Label'].values
            print(f'| {name} | {len(s)} | {yy.mean():.3f} | {s["p_C3_calibrated"].mean():.3f} | {s[f"p_{best}_calibrated"].mean():.3f} | '
                  f'{np.mean((s["p_C3_calibrated"] - yy) ** 2):.4f} | {np.mean((s[f"p_{best}_calibrated"] - yy) ** 2):.4f} | '
                  f'{dl:+.4f} [{cid[1]:+.4f}, {cid[2]:+.4f}] |')
        print('  (stvarno / pred = prosjek po retcima; oba retka po meču daju 0.5 ukupno, pa su bucketi informativni samo po jačini razlike |pred-stvarno| unutar igračeve perspektive)')
        print(f'\n## {title}: by gender')
        print(f'| spol | n mečeva | Brier C3 [CI] | Brier {best} [CI] | d [95% CI] |')
        print('|---|---|---|---|---|')
        for name, g in (('SM', 0), ('SF', 1)):
            mask = (d['liga_tip'] == g).values
            ma, ca, mb, cb, dl, cid = brier_ci_and_delta(d, 'C3', best, mask)
            # note: here a=C3, b=best -> dl = C3 - best; report best - C3
            print(f'| {name} | {d[mask]["SusretId"].nunique()} | {ma:.4f} [{ca[1]:.4f}, {ca[2]:.4f}] | {mb:.4f} [{cb[1]:.4f}, {cb[2]:.4f}] | '
                  f'{-dl:+.4f} [{-cid[2]:+.4f}, {-cid[1]:+.4f}] |')


if __name__ == '__main__':
    main()
