"""
Exp D - TabPFN v2 on the anonymised Nov-2026 export (plan 2026-11-14-ml-repo-i-tabpfn.md, 2.3 / 2.5 D4+D6).

Variants (v2 ONLY, local weights, CPU):
  D-Aprime     A' features (36, no ELO)
  D-Bprime     B' features (37, with elo_diff)
  D-Bprime-par B' minus match-context columns (production-compatible pair model)
  --d14 <kind><window>   adds D14 columns (log1p rest, rest>180d flag, relative ELO per gender):
                         e.g. --d14 rel365, pct90 (variant name gets a '+d14-..' suffix)
Context: --contexts last,random  (last N rows / random N rows of the train fold; N = --context-rows <= 10000)
Leakage rules: context = train fold minus the chronologically-last 15% (calibration slice, outside the context);
both flip rows of a match on the same side; no global preprocessing; predictions symmetrised
p = (p(A,B) + 1 - p(B,A)) / 2; raw vs isotonic chosen on the calibration slice (same rule as C3).

Usage (after the weights are in place):
  .venv-tabpfn\\Scripts\\python.exe scripts\\run_walkforward_tabpfn.py --workdir <workdir>
      --variants D-Aprime,D-Bprime,D-Bprime-par --contexts last,random --seeds 42,1,2 --folds 4
  ... --folds 1,2,3,4 --contexts last --seeds 42        (D6 walk-forward)
  ... --dry-run     (data + assertions + plan only; does NOT need the weights, never imports tabpfn)
Output: <workdir>/data/test_predictions_d_fold<k>.parquet (p_cal__/p_raw__ column per config, resumable:
finished configs are skipped), <workdir>/data/tabpfn_timing.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabpfn_data as td  # noqa: E402

VARIANTS = {'D-Aprime': 'A_PRIME', 'D-Bprime': 'B_PRIME', 'D-Bprime-par': 'B_PRIME_PAR'}


def predict_chunks(clf, X, chunk):
    out = []
    for i in range(0, len(X), chunk):
        out.append(clf.predict_proba(X[i:i + chunk])[:, 1])
        print(f'      predict {min(i + chunk, len(X)):,}/{len(X):,}', flush=True)
    return np.concatenate(out)


def calibrate(p_calib_sym, y_calib, p_test_sym):
    iso = IsotonicRegression(out_of_bounds='clip').fit(p_calib_sym, y_calib)
    b_raw = brier_score_loss(y_calib, p_calib_sym)
    b_iso = brier_score_loss(y_calib, iso.predict(p_calib_sym))
    if b_iso < b_raw:
        return iso.predict(p_test_sym), 'isotonic', b_raw, b_iso
    return p_test_sym, 'raw', b_raw, b_iso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', default=td.DEFAULT_WORKDIR,
                    help='anonymised workspace dir containing data/ (or env GDSI_ML_WORKDIR)')
    ap.add_argument('--variants', default='D-Aprime,D-Bprime,D-Bprime-par')
    ap.add_argument('--contexts', default='last,random')
    ap.add_argument('--context-rows', type=int, default=10_000)
    ap.add_argument('--seeds', default='42,1,2')
    ap.add_argument('--folds', default='4')
    ap.add_argument('--n-estimators', type=int, default=8)
    ap.add_argument('--chunk', type=int, default=1000)
    ap.add_argument('--d14', default=None, help='rel365 | rel90 | pct365 | pct90')
    ap.add_argument('--limit-test', type=int, default=None, help='debug: only first N test matches')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='cuda = Colab GPU (Exp D on T4)')
    ap.add_argument('--out-dir', default=None, help='where predictions + timing are written (default <workdir>/data)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    assert a.context_rows <= 10_000, 'TabPFN v2 limit is 10,000 rows'

    import tabpfn_guard  # sets env (telemetry off, offline) BEFORE tabpfn is ever imported
    ckpt = tabpfn_guard.ckpt_path(require=not a.dry_run)

    if not a.workdir:
        ap.error('--workdir is required (or set env GDSI_ML_WORKDIR)')
    ds = td.load_tabpfn_dataset(a.workdir)
    sym, COLS = ds['sym'], dict(ds['COLS'])
    d14_tag = ''
    if a.d14:
        kind, win = a.d14[:3], int(a.d14[3:])
        assert kind in ('rel', 'pct') and win in (90, 365)
        new = td.add_d14_features(ds, rel_kind=kind, rel_window=win)
        sym = ds['sym']
        for k in COLS:
            # relative ELO is ELO information -> never in the ELO-free A' variant
            COLS[k] = COLS[k] + [c for c in new if k != 'A_PRIME' or not c.startswith(('rel', 'pct'))]
        d14_tag = f'+d14-{a.d14}'
        td.lap(f'D14 columns added: {new}')
    variants = [v.strip() for v in a.variants.split(',')]
    ctx_modes = [c.strip() for c in a.contexts.split(',')]
    seeds = [int(s) for s in a.seeds.split(',')]
    folds = [int(f) for f in a.folds.split(',')]
    for v in variants:
        assert v in VARIANTS, v

    print(f'plan: folds={folds} variants={variants}{d14_tag} contexts={ctx_modes} seeds={seeds} '
          f'context_rows={a.context_rows} n_estimators={a.n_estimators}')
    print(f'ckpt={ckpt}  (exists={ckpt.is_file()})')
    if a.dry_run:
        for k in folds:
            fo = ds['FOLDS'][k - 1]
            pool, calib = td.split_context_calib(sym, fo['train_ids'], ds['match_order'])
            print(f'  fold {k}: train {len(fo["train_ids"]):,} matches -> pool {len(pool):,} / calib {len(calib):,}; '
                  f'test {len(fo["test_ids"]):,}; cutoff {fo["cutoff"].date()}')
        for v in variants:
            print(f'  {v}{d14_tag}: {len(COLS[VARIANTS[v]])} columns')
        print('dry-run OK (nothing fitted, tabpfn not imported).')
        return

    import torch
    from tabpfn import TabPFNClassifier
    if a.device == 'cuda' and not torch.cuda.is_available():
        sys.exit('--device cuda requested but torch.cuda.is_available() is False (Colab: Runtime > Change runtime type > T4 GPU).')
    data_dir = Path(a.out_dir) if a.out_dir else Path(a.workdir) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_cols = ['SusretId', 'Datum', 'PlayerId', 'OpponentId', 'Label', 'Perspective', 'liga_tip']

    for k in folds:
        fo = ds['FOLDS'][k - 1]
        out_path = data_dir / f'test_predictions_d_fold{k}.parquet'
        test = sym[sym['SusretId'].isin(fo['test_ids'])]
        if a.limit_test:
            keep = list(test['SusretId'].drop_duplicates()[:a.limit_test])
            test = test[test['SusretId'].isin(keep)]
        test = test.reset_index(drop=True)
        res = pd.read_parquet(out_path) if out_path.exists() else test[meta_cols].copy()
        assert (res['SusretId'].values == test['SusretId'].values).all()
        pool, calib_ids = td.split_context_calib(sym, fo['train_ids'], ds['match_order'])
        calib = sym[sym['SusretId'].isin(calib_ids)].reset_index(drop=True)
        assert not (set(calib['SusretId']) & set(test['SusretId']))
        for v in variants:
            cols = COLS[VARIANTS[v]]
            for mode in ctx_modes:
                for seed in seeds:
                    key = f'{v}{d14_tag}__{mode}{a.context_rows // 1000}k__s{seed}'
                    if f'p_cal__{key}' in res.columns:
                        print(f'skip fold {k} {key} (done)')
                        continue
                    ctx_ids = td.pick_context_ids(pool, mode, a.context_rows, seed)
                    ctx = sym[sym['SusretId'].isin(set(ctx_ids))]
                    assert not (set(ctx['SusretId']) & (calib_ids | set(test['SusretId'])))
                    assert ctx['Datum'].max() <= test['Datum'].min()
                    print(f'fold {k} {key}: context {len(ctx):,} rows, calib {len(calib):,}, test {len(test):,}', flush=True)
                    t0 = time.time()
                    clf = TabPFNClassifier(model_path=str(ckpt), device=a.device, n_estimators=a.n_estimators,
                                           random_state=seed, fit_mode='fit_with_cache',
                                           ignore_pretraining_limits=False)
                    clf.fit(ctx[cols].to_numpy(float), ctx['Label'].to_numpy(int))
                    t_fit = time.time() - t0
                    p_cal_raw = predict_chunks(clf, calib[cols].to_numpy(float), a.chunk)
                    p_te_raw = predict_chunks(clf, test[cols].to_numpy(float), a.chunk)
                    t_all = time.time() - t0
                    p_cal_s = td.symmetrise(calib, p_cal_raw)
                    p_te_s = td.symmetrise(test, p_te_raw)
                    p_final, how, b_raw, b_iso = calibrate(p_cal_s, calib['Label'].to_numpy(int), p_te_s)
                    res[f'p_raw__{key}'] = p_te_s
                    res[f'p_cal__{key}'] = p_final
                    res.to_parquet(out_path, index=False)  # incremental, resumable
                    b_test = brier_score_loss(test['Label'], p_final)
                    print(f'  -> {how}; calib Brier raw {b_raw:.5f} / iso {b_iso:.5f}; TEST Brier {b_test:.5f} '
                          f'({t_all / 60:.1f} min)', flush=True)
                    with open(data_dir / 'tabpfn_timing.jsonl', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({'fold': k, 'key': key, 'fit_s': round(t_fit, 1), 'total_s': round(t_all, 1),
                                            'calibration': how, 'brier_test': round(b_test, 5),
                                            'threads': torch.get_num_threads(), 'device': a.device}) + '\n')


if __name__ == '__main__':
    main()
