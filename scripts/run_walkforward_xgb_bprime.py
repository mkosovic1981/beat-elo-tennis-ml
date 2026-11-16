"""
Exp XGB-B' -- walk-forward XGBoost on the anonymised Nov-2026 export, same 37 features (B') and folds as the TabPFN runs
(notebooks 14 / 15), so XGBoost and TabPFN can be compared on identical inputs.

Model code and hyper-parameters are the same as `fit_xgb_fold` in run_walkforward_c3c4.py (notebook 09): max_depth=4,
reg_alpha=0.1, reg_lambda=1.0, learning_rate=0.05, up to 500 trees with early stopping (30 rounds), subsample=colsample=0.9,
seed 42. Per fold the train matches are cut chronologically like `carve_train` (last 15 % = isotonic calibration slice,
then last 10 % of the rest = early-stopping slice, remainder = core). Isotonic calibration on the calibration slice.

Usage:  python scripts/run_walkforward_xgb_bprime.py <workdir>      (or env GDSI_ML_WORKDIR)
Reads:  <workdir>/data (features.parquet, feature_tags.json, ...)   Writes: <workdir>/exp_xgb/test_predictions_xgb_bprime_fold<k>.parquet
Contains no data; predictions stay in the (untracked) workdir.
"""
import os
import sys
from pathlib import Path

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabpfn_data as td  # noqa: E402

PARAMS = {'max_depth': 4, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'learning_rate': 0.05}
SEED = 42


def carve(match_order, train_ids, calib_frac=0.15, early_stop_frac=0.10):
    fo = match_order[match_order['SusretId'].isin(train_ids)].reset_index(drop=True)
    n = len(fo)
    n_calib = max(1, int(round(n * calib_frac)))
    fit = fo.iloc[:n - n_calib]
    calib = set(fo.iloc[n - n_calib:]['SusretId'])
    nf = len(fit)
    n_eval = max(1, int(round(nf * early_stop_frac)))
    return set(fit.iloc[:nf - n_eval]['SusretId']), set(fit.iloc[nf - n_eval:]['SusretId']), calib


def main(workdir):
    ds = td.load_tabpfn_dataset(workdir)
    sym, cols, mo = ds['sym'], ds['COLS']['B_PRIME'], ds['match_order']
    out_dir = Path(workdir) / 'exp_xgb'
    out_dir.mkdir(exist_ok=True)
    print(len(cols), 'features')
    for f in ds['FOLDS']:
        k = f['fold']
        core, ev, calib = carve(mo, f['train_ids'])
        m = lambda ids: sym['SusretId'].isin(ids)
        model = xgb.XGBClassifier(n_estimators=500, subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                                  tree_method='hist', eval_metric='logloss', early_stopping_rounds=30, n_jobs=-1, **PARAMS)
        model.fit(sym.loc[m(core), cols], sym.loc[m(core), 'Label'],
                  eval_set=[(sym.loc[m(ev), cols], sym.loc[m(ev), 'Label'])], verbose=False)
        cal = CalibratedClassifierCV(estimator=model, method='isotonic', cv='prefit').fit(
            sym.loc[m(calib), cols], sym.loc[m(calib), 'Label'])
        te = sym[m(f['test_ids'])]
        res = te[['SusretId', 'Datum', 'PlayerId', 'OpponentId', 'Label', 'Perspective']].copy()
        res['p_cal_Bprime_xgb'] = cal.predict_proba(te[cols])[:, 1]
        res.to_parquet(out_dir / f'test_predictions_xgb_bprime_fold{k}.parquet')
        print(f'fold {k}: {len(res)} rows, best_iteration {model.best_iteration}, '
              f'Brier {((res.p_cal_Bprime_xgb - res.Label) ** 2).mean():.5f}')


if __name__ == '__main__':
    wd = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('GDSI_ML_WORKDIR')
    if not wd:
        sys.exit('usage: run_walkforward_xgb_bprime.py <workdir> (or env GDSI_ML_WORKDIR)')
    main(wd)
