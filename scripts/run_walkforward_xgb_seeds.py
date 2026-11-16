"""
Exp XGB-B' / B'-par, multi-seed walk-forward with BOTH the raw and the isotonic-calibrated probability per seed.

Same model, hyper-parameters, features, folds and chronological train / early-stopping / calibration split as
`run_walkforward_xgb_bprime.py` (imported, not copied); the only differences are

  * several seeds (xgboost `random_state`; subsample = colsample_bytree = 0.9 make the seed matter),
  * the raw model output `p_raw_s<seed>` is saved next to the calibrated one `p_cal_s<seed>`.

Output (new files, nothing existing is overwritten):
  <workdir>/exp_fair/test_predictions_xgbseeds_<stem>_fold<k>.parquet
  with columns SusretId, Datum, PlayerId, OpponentId, Label, Perspective, p_raw_s<seed>, p_cal_s<seed> ...

Usage:  python scripts/run_walkforward_xgb_seeds.py <workdir> [--variant Bprime|Bprime_par] [--seeds 42 1 2 3 4]
        (or env GDSI_ML_WORKDIR). Contains no data; predictions stay in the (untracked) workdir.
"""
import argparse
import os
import sys
from pathlib import Path

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabpfn_data as td  # noqa: E402
from run_walkforward_xgb_bprime import PARAMS, VARIANTS, carve  # noqa: E402

SEEDS = (42, 1, 2, 3, 4)


def main(workdir, variant, seeds, out_name='exp_fair'):
    ds = td.load_tabpfn_dataset(workdir)
    sym, cols, mo = ds['sym'], ds['COLS'][VARIANTS[variant][0]], ds['match_order']
    stem = VARIANTS[variant][1].replace('xgb_', '')  # bprime | bprimepar
    out_dir = Path(workdir) / out_name
    out_dir.mkdir(exist_ok=True)
    print(variant, len(cols), 'features, seeds', list(seeds))
    for f in ds['FOLDS']:
        k = f['fold']
        core, ev, calib = carve(mo, f['train_ids'])
        m = lambda ids: sym['SusretId'].isin(ids)
        te = sym[m(f['test_ids'])]
        res = te[['SusretId', 'Datum', 'PlayerId', 'OpponentId', 'Label', 'Perspective']].copy()
        for s in seeds:
            model = xgb.XGBClassifier(n_estimators=500, subsample=0.9, colsample_bytree=0.9, random_state=s,
                                      tree_method='hist', eval_metric='logloss', early_stopping_rounds=30,
                                      n_jobs=-1, **PARAMS)
            model.fit(sym.loc[m(core), cols], sym.loc[m(core), 'Label'],
                      eval_set=[(sym.loc[m(ev), cols], sym.loc[m(ev), 'Label'])], verbose=False)
            cal = CalibratedClassifierCV(estimator=model, method='isotonic', cv='prefit').fit(
                sym.loc[m(calib), cols], sym.loc[m(calib), 'Label'])
            res[f'p_raw_s{s}'] = model.predict_proba(te[cols])[:, 1]
            res[f'p_cal_s{s}'] = cal.predict_proba(te[cols])[:, 1]
            print(f'fold {k} seed {s}: best_iteration {model.best_iteration}, '
                  f'Brier raw {((res[f"p_raw_s{s}"] - res.Label) ** 2).mean():.5f} '
                  f'cal {((res[f"p_cal_s{s}"] - res.Label) ** 2).mean():.5f}', flush=True)
        res.to_parquet(out_dir / f'test_predictions_xgbseeds_{stem}_fold{k}.parquet')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('workdir', nargs='?', default=os.environ.get('GDSI_ML_WORKDIR'))
    ap.add_argument('--variant', choices=sorted(VARIANTS), default='Bprime')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    a = ap.parse_args()
    if not a.workdir:
        sys.exit('usage: run_walkforward_xgb_seeds.py <workdir> [--variant ...] [--seeds ...] (or env GDSI_ML_WORKDIR)')
    main(a.workdir, a.variant, a.seeds)
