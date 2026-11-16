"""
Exp D35 - TabPFN-3.5 one-off NON-COMMERCIAL research run (licence tabpfn-3-5-license-v1.0).
Copy of run_walkforward_tabpfn.py adapted for 3.5; the v2 script and its results are untouched.

Differences vs the v2 run:
  - local safetensors weights only (--ckpt / env TABPFN35_CKPT), see tabpfn35_guard.py
  - context is not limited to 10k rows: --context-rows full = whole pool after the calibration slice
    (or an integer N = last N rows, e.g. 10000 for the "context size" control)
  - outputs use the key prefix d35__ and file names ..._d35_...; v2 outputs are never overwritten
Leakage rules (same as v2): context = train fold minus the chronologically-last 15% (calibration slice);
both flip rows of a match on the same side; predictions symmetrised p = (p(A,B) + 1 - p(B,A)) / 2;
raw vs isotonic chosen on the calibration slice.

Usage:
  python scripts/run_walkforward_tabpfn35.py --workdir <ws dir> --ckpt <weights file>
      --variants D-Bprime,D-Bprime-par --context-rows full --seeds 42 --folds 4 --device cuda
  --dry-run   data + assertions + plan only (never imports tabpfn, needs no weights)
Output (in --out-dir, default <workdir>/data): test_predictions_d35_fold<k>.parquet, tabpfn35_timing.jsonl
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
import tabpfn35_guard  # noqa: E402  (env + network block BEFORE tabpfn is ever imported)
import tabpfn_data as td  # noqa: E402

VARIANTS = {'D-Aprime': 'A_PRIME', 'D-Bprime': 'B_PRIME', 'D-Bprime-par': 'B_PRIME_PAR'}
FULL = 10 ** 9


def predict_chunks(clf, X, chunk):
    out = []
    for i in range(0, len(X), chunk):
        out.append(clf.predict_proba(X[i:i + chunk])[:, 1])
        tabpfn35_guard.assert_no_priorlabs_contact()
        print(f'      predict {min(i + chunk, len(X)):,}/{len(X):,}', flush=True)
    return np.concatenate(out)


def calibrate(p_calib_sym, y_calib, p_test_sym):
    iso = IsotonicRegression(out_of_bounds='clip').fit(p_calib_sym, y_calib)
    b_raw = brier_score_loss(y_calib, p_calib_sym)
    b_iso = brier_score_loss(y_calib, iso.predict(p_calib_sym))
    if b_iso < b_raw:
        return iso.predict(p_test_sym), 'isotonic', b_raw, b_iso
    return p_test_sym, 'raw', b_raw, b_iso


def make_classifier(TabPFNClassifier, ckpt, a, seed):
    return TabPFNClassifier(model_path=str(ckpt), device=a.device, n_estimators=a.n_estimators,
                            random_state=seed, fit_mode=a.fit_mode, inference_precision=a.inference_precision,
                            ignore_pretraining_limits=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', required=True, help='anonymised workspace dir containing data/')
    ap.add_argument('--ckpt', default=None, help='local .safetensors weights (or env TABPFN35_CKPT)')
    ap.add_argument('--variants', default='D-Bprime')
    ap.add_argument('--context-rows', default='full', help="'full' or an integer (last N rows)")
    ap.add_argument('--seeds', default='42')
    ap.add_argument('--folds', default='4')
    ap.add_argument('--n-estimators', type=int, default=8)
    ap.add_argument('--fit-mode', default='fit_with_cache',
                    choices=['low_memory', 'fit_preprocessors', 'fit_with_cache'])
    ap.add_argument('--inference-precision', default='auto', help="'auto' | 'autocast'")
    ap.add_argument('--chunk', type=int, default=1000)
    ap.add_argument('--limit-test', type=int, default=None, help='debug: only first N test matches')
    ap.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    ctx_rows = FULL if str(a.context_rows).lower() == 'full' else int(a.context_rows)
    ctx_tag = 'full' if ctx_rows == FULL else f'{ctx_rows // 1000}k'
    ckpt = tabpfn35_guard.ckpt_path(a.ckpt, require=not a.dry_run)

    ds = td.load_tabpfn_dataset(a.workdir)
    sym, COLS = ds['sym'], dict(ds['COLS'])
    variants = [v.strip() for v in a.variants.split(',')]
    seeds = [int(s) for s in a.seeds.split(',')]
    folds = [int(f) for f in a.folds.split(',')]
    for v in variants:
        assert v in VARIANTS, v
    print(f'plan: folds={folds} variants={variants} context={ctx_tag} (mode last) seeds={seeds} '
          f'n_estimators={a.n_estimators} fit_mode={a.fit_mode} device={a.device}')
    print(f'ckpt={ckpt}')
    if a.dry_run:
        for k in folds:
            fo = ds['FOLDS'][k - 1]
            pool, calib = td.split_context_calib(sym, fo['train_ids'], ds['match_order'])
            n_ctx = len(td.pick_context_ids(pool, 'last', ctx_rows, 0)) * 2
            print(f'  fold {k}: pool {len(pool):,} matches / calib {len(calib):,}; test {len(fo["test_ids"]):,}; '
                  f'context rows ({ctx_tag}) {n_ctx:,}; cutoff {fo["cutoff"].date()}')
        for v in variants:
            print(f'  {v}: {len(COLS[VARIANTS[v]])} columns')
        print('dry-run OK (nothing fitted, tabpfn not imported).')
        return

    import importlib.metadata as md
    import torch
    from tabpfn import TabPFNClassifier
    if a.device == 'cuda' and not torch.cuda.is_available():
        sys.exit('--device cuda requested but torch.cuda.is_available() is False.')
    data_dir = Path(a.out_dir) if a.out_dir else Path(a.workdir) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_cols = ['SusretId', 'Datum', 'PlayerId', 'OpponentId', 'Label', 'Perspective', 'liga_tip']

    for k in folds:
        fo = ds['FOLDS'][k - 1]
        out_path = data_dir / f'test_predictions_d35_fold{k}.parquet'
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
            for seed in seeds:
                key = f'd35__{v}__last{ctx_tag}__s{seed}'
                if f'p_cal__{key}' in res.columns:
                    print(f'skip fold {k} {key} (done)')
                    continue
                ctx_ids = td.pick_context_ids(pool, 'last', ctx_rows, seed)
                ctx = sym[sym['SusretId'].isin(set(ctx_ids))]
                assert not (set(ctx['SusretId']) & (calib_ids | set(test['SusretId'])))
                assert ctx['Datum'].max() <= test['Datum'].min()
                print(f'fold {k} {key}: context {len(ctx):,} rows, calib {len(calib):,}, test {len(test):,}', flush=True)
                t0 = time.time()
                clf = make_classifier(TabPFNClassifier, ckpt, a, seed)
                clf.fit(ctx[cols].to_numpy(float), ctx['Label'].to_numpy(int))
                tabpfn35_guard.assert_no_priorlabs_contact()
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
                with open(data_dir / 'tabpfn35_timing.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'fold': k, 'key': key, 'context_rows': int(len(ctx)), 'fit_s': round(t_fit, 1),
                                        'total_s': round(t_all, 1), 'calibration': how,
                                        'brier_test': round(b_test, 5), 'device': a.device,
                                        'tabpfn': md.version('tabpfn'), 'ckpt': Path(ckpt).name,
                                        'n_estimators': a.n_estimators, 'fit_mode': a.fit_mode}) + '\n')


if __name__ == '__main__':
    main()
