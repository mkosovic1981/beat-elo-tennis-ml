"""
Exp D35 - timing + no-network check for TabPFN-3.5 (one-off, non-commercial research run).

Real-data timing (fold 4, D-Bprime, full context, n_test rows, seed 42):
  python scripts/tabpfn35_smoke_timing.py --workdir <ws dir> --ckpt <weights> [--n-test 2000]
Synthetic tiny run (network-block check only, no real data):
  python scripts/tabpfn35_smoke_timing.py --synthetic --ckpt <weights>
Prints 'MIN_PER_RUN=<x>' = fit + per-row predict time extrapolated to calib slice + full test set.
If > 30 min/run the Fast checkpoint (tabpfn-v3.5-fast-20260909.safetensors) is suggested.
Always ends with the guard check: exit code != 0 if tabpfn tried to reach Prior Labs / Hugging Face.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabpfn35_guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--fold', type=int, default=4)
    ap.add_argument('--variant', default='D-Bprime')
    ap.add_argument('--n-test', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-estimators', type=int, default=8)
    ap.add_argument('--fit-mode', default='fit_with_cache')
    ap.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    a = ap.parse_args()
    ckpt = tabpfn35_guard.ckpt_path(a.ckpt)

    import torch
    from tabpfn import TabPFNClassifier
    if a.device == 'cuda' and not torch.cuda.is_available():
        sys.exit('--device cuda requested but no CUDA GPU is available')

    n_calib_rows = n_test_full = 0
    if a.synthetic:
        rng = np.random.default_rng(0)
        w = rng.normal(size=20)

        def make(n):
            X = rng.normal(size=(n, 20))
            return X, ((X @ w + rng.normal(size=n)) > 0).astype(int)
        Xc, yc = make(500)
        Xt, _ = make(100)
    else:
        if not a.workdir:
            sys.exit('--workdir is required for the real-data timing (or use --synthetic)')
        import tabpfn_data as td
        from run_walkforward_tabpfn35 import VARIANTS
        ds = td.load_tabpfn_dataset(a.workdir)
        sym, fo = ds['sym'], ds['FOLDS'][a.fold - 1]
        cols = ds['COLS'][VARIANTS[a.variant]]
        pool, calib_ids = td.split_context_calib(sym, fo['train_ids'], ds['match_order'])
        ctx = sym[sym['SusretId'].isin(set(td.pick_context_ids(pool, 'last', 10 ** 9, a.seed)))]
        test = sym[sym['SusretId'].isin(fo['test_ids'])].reset_index(drop=True)
        n_calib_rows = int(sym['SusretId'].isin(calib_ids).sum())
        n_test_full = len(test)
        Xc, yc = ctx[cols].to_numpy(float), ctx['Label'].to_numpy(int)
        Xt = test[cols].to_numpy(float)[:a.n_test]
        print(f'fold {a.fold} {a.variant}: context {len(Xc):,} rows, calib {n_calib_rows:,} rows, '
              f'test total {n_test_full:,} rows, timing on {len(Xt):,} test rows')

    clf = TabPFNClassifier(model_path=str(ckpt), device=a.device, n_estimators=a.n_estimators,
                           random_state=a.seed, fit_mode=a.fit_mode, ignore_pretraining_limits=False)
    t = time.time(); clf.fit(Xc, yc); tf = time.time() - t
    t = time.time(); clf.predict_proba(Xt); tp = time.time() - t
    per1k = tp / len(Xt) * 1000
    print(f'fit={tf:.1f}s predict({len(Xt)})={tp:.1f}s -> {per1k:.1f}s / 1k rows')
    if a.device == 'cuda':
        print(f'GPU peak memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    tabpfn35_guard.assert_no_priorlabs_contact()
    print('no contact with Prior Labs / Hugging Face attempted: OK')
    if not a.synthetic:
        run_min = (tf + per1k * (n_calib_rows + n_test_full) / 1000) / 60
        print(f'MIN_PER_RUN={run_min:.1f}')
        if run_min > 30:
            print('> 30 min/run: consider the Fast checkpoint tabpfn-v3.5-fast-20260909.safetensors '
                  '(Prior-Labs/tabpfn_3_5) and re-time.')


if __name__ == '__main__':
    main()
