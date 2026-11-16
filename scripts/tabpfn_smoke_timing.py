r"""
Exp D - D2: CPU smoke test on SYNTHETIC data (no real/anonymised data). Time per context size.

  .venv-tabpfn\Scripts\python.exe scripts\tabpfn_smoke_timing.py [--contexts 1000,5000,10000]
                                                                  [--n-test 2000] [--features 37]
                                                                  [--n-estimators 8]
Needs the v2 checkpoint (see tabpfn_guard.py). Prints a table: fit s, predict s, s per 1k test rows,
and an extrapolation for the main split (7,214 test rows) x 3 variants x 2 contexts x 3 seeds.
"""
import argparse
import time

import numpy as np

import tabpfn_guard
CKPT = tabpfn_guard.ckpt_path()

import torch  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contexts', default='1000,5000,10000')
    ap.add_argument('--n-test', type=int, default=2000)
    ap.add_argument('--features', type=int, default=37)
    ap.add_argument('--n-estimators', type=int, default=8)
    ap.add_argument('--fit-mode', default='fit_with_cache')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    a = ap.parse_args()
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device cuda requested but no CUDA GPU is available')
    torch.set_num_threads(torch.get_num_threads())
    print(f'ckpt={CKPT} threads={torch.get_num_threads()} n_estimators={a.n_estimators} fit_mode={a.fit_mode} device={a.device}')
    rng = np.random.default_rng(0)
    w = rng.normal(size=a.features)
    def make(n):
        X = rng.normal(size=(n, a.features)); X[rng.random(X.shape) < 0.05] = np.nan
        y = (np.nan_to_num(X) @ w + rng.normal(size=n) > 0).astype(int)
        return X, y
    Xte, _ = make(a.n_test)
    rows = []
    for n in [int(x) for x in a.contexts.split(',')]:
        X, y = make(n)
        clf = TabPFNClassifier(model_path=str(CKPT), device=a.device, n_estimators=a.n_estimators, random_state=0,
                               fit_mode=a.fit_mode, ignore_pretraining_limits=False)
        t = time.time(); clf.fit(X, y); tf = time.time() - t
        t = time.time(); clf.predict_proba(Xte); tp = time.time() - t
        per1k = tp / a.n_test * 1000
        rows.append((n, tf, tp, per1k))
        print(f'context={n:>6} fit={tf:7.1f}s predict({a.n_test})={tp:8.1f}s -> {per1k:7.1f}s / 1k test rows', flush=True)
    print('\nExtrapolation, main split (test 7,214 rows), per run:')
    for n, tf, tp, per1k in rows:
        run = tf + per1k * 7.214
        print(f'  context {n:>6}: ~{run / 60:6.1f} min/run;  x18 runs (3 var x 2 ctx x 3 seeds) = {run * 18 / 3600:5.1f} h')

if __name__ == '__main__':
    main()
