"""
Walk-forward CV (5 models: Exp 0 / B' / C2 / C3 / C4) -- notebook 12 back-end.

Usage (from scripts/):
    python run_walkforward_c3c4.py --folds 1              # quick validation, fold 1 only
    python run_walkforward_c3c4.py --folds 1,2,3,4         # full run (default)

Writes results to ../data/walkforward_c3c4_results.json (gitignored -- data/*.json).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence

from wf_c3c4_common import RANDOM_STATE, DATA_DIR, lap, load_common

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.use_deterministic_algorithms(False)


def compute_metrics_dict(y_true, p):
    p_c = np.clip(p, 1e-6, 1 - 1e-6)
    out = {
        'accuracy': accuracy_score(y_true, (p >= 0.5).astype(int)),
        'brier': brier_score_loss(y_true, p),
        'log_loss': log_loss(y_true, p_c),
    }
    try:
        out['auc'] = roc_auc_score(y_true, p)
    except ValueError:
        out['auc'] = np.nan
    return out


# ============================================================================
# Load everything fold-independent ONCE
# ============================================================================
C = load_common()
sym = C['sym']
FOLDS = C['FOLDS']
EXP_B_PRIME_COLS = C['EXP_B_PRIME_COLS']
BEST_PARAMS_B = C['BEST_PARAMS_B']
player_log = C['player_log']
HIST_C2, LEN_C2 = C['HIST_C2'], C['LEN_C2']
player_pos, opp_pos = C['player_pos'], C['opp_pos']
HEAD_COLS_C3, HEAD_COLS_C4 = C['HEAD_COLS_C3'], C['HEAD_COLS_C4']
carve_train = C['carve_train']
build_hist_c3_fold = C['build_hist_c3_fold']
scores = C['scores']

HIST_C2_T = torch.from_numpy(HIST_C2)
LEN_C2_T = torch.from_numpy(LEN_C2)

HIDDEN_SIZE, DROPOUT, WEIGHT_DECAY, LR = 24, 0.3, 1e-4, 2e-3
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 512, 60, 8


# ============================================================================
# Model classes (EXACT copies of notebooks 09/10/11)
# ============================================================================
class TwoTowerGRUElo(nn.Module):
    """Exp C2 -- copied unchanged from 08_gru_elo.ipynb Step 8 / 09 Step 5."""
    def __init__(self, input_size=7, hidden_size=24, dropout=0.3, mlp_hidden=16):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.drop_embed = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1))

    def encode(self, seq, lengths):
        packed = pack_padded_sequence(seq, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.drop_embed(h_n[-1])

    def forward(self, player_seq, player_len, opp_seq, opp_len, elo_p, elo_o):
        h_p = self.encode(player_seq, player_len)
        h_o = self.encode(opp_seq, opp_len)
        combined = torch.cat([h_p, h_o, h_p - h_o, elo_p.unsqueeze(1), elo_o.unsqueeze(1)], dim=1)
        return self.head(combined).squeeze(-1)


class TwoTowerGRUEnriched(nn.Module):
    """Exp C3/C4 -- copied unchanged from 10_enriched_gru.ipynb / 11_enriched_gru_text.ipynb."""
    def __init__(self, input_size=9, hidden_size=24, dropout=0.3, mlp_hidden=16, n_head=15):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.drop_embed = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 3 + n_head, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1))

    def encode(self, seq, lengths):
        packed = pack_padded_sequence(seq, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.drop_embed(h_n[-1])

    def forward(self, player_seq, player_len, opp_seq, opp_len, head_feats):
        h_p = self.encode(player_seq, player_len)
        h_o = self.encode(opp_seq, opp_len)
        combined = torch.cat([h_p, h_o, h_p - h_o, head_feats], dim=1)
        return self.head(combined).squeeze(-1)


# ============================================================================
# Calibration helper (EXACT copy of notebook 09's _calibrate_and_score)
# ============================================================================
def calibrate_and_score(calib_logit_np, calib_y_np, test_logit_np):
    T_param = torch.nn.Parameter(torch.ones(1))
    clt, cyt = torch.from_numpy(calib_logit_np), torch.from_numpy(calib_y_np)
    opt_T = torch.optim.LBFGS([T_param], lr=0.05, max_iter=100)

    def _closure():
        opt_T.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(clt / T_param, cyt)
        loss.backward()
        return loss
    opt_T.step(_closure)
    T_value = float(T_param.detach().clamp(min=1e-2))

    p_calib_raw = 1 / (1 + np.exp(-calib_logit_np))
    p_calib_temp = torch.sigmoid(clt / T_value).numpy()
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_calib_raw, calib_y_np)
    p_calib_iso = iso.predict(p_calib_raw)

    p_test_raw = 1 / (1 + np.exp(-test_logit_np))
    if brier_score_loss(calib_y_np, p_calib_iso) < brier_score_loss(calib_y_np, p_calib_temp):
        return iso.predict(p_test_raw)
    return torch.sigmoid(torch.from_numpy(test_logit_np) / T_value).numpy()


# ============================================================================
# Exp B' (XGBoost) fit -- EXACT copy of notebook 09's fit_xgb_fold
# ============================================================================
def fit_xgb_fold(core_mask, eval_mask, calib_mask, test_mask, n_estimators=500, isotonic_min_rows=1000):
    cols = EXP_B_PRIME_COLS
    X_core, y_core = sym.loc[core_mask, cols], sym.loc[core_mask, 'Label']
    X_eval, y_eval = sym.loc[eval_mask, cols], sym.loc[eval_mask, 'Label']
    X_calib, y_calib = sym.loc[calib_mask, cols], sym.loc[calib_mask, 'Label']
    X_test, y_test = sym.loc[test_mask, cols], sym.loc[test_mask, 'Label']

    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=BEST_PARAMS_B['max_depth'],
        learning_rate=BEST_PARAMS_B['learning_rate'], reg_alpha=BEST_PARAMS_B['reg_alpha'],
        reg_lambda=BEST_PARAMS_B['reg_lambda'], subsample=0.9, colsample_bytree=0.9,
        random_state=RANDOM_STATE, tree_method='hist', eval_metric='logloss',
        early_stopping_rounds=30, n_jobs=-1,
    )
    model.fit(X_core, y_core, eval_set=[(X_eval, y_eval)], verbose=False)

    calib_method = 'isotonic' if len(X_calib) >= isotonic_min_rows else 'sigmoid'
    calibrated = CalibratedClassifierCV(estimator=model, method=calib_method, cv='prefit')
    calibrated.fit(X_calib, y_calib)

    p_test = calibrated.predict_proba(X_test)[:, 1]
    return p_test, y_test.values


# ============================================================================
# Exp C2 fit -- EXACT copy of notebook 09's bucket_tensors_elo / fit_gru_elo_fold
# ============================================================================
def bucket_tensors_elo(mask_ids, elo_mean, elo_std):
    mask = sym['SusretId'].isin(mask_ids).values
    pp = torch.as_tensor(player_pos[mask], dtype=torch.long)
    op = torch.as_tensor(opp_pos[mask], dtype=torch.long)
    y = torch.as_tensor(sym.loc[mask, 'Label'].values, dtype=torch.float32)
    ep = torch.as_tensor(((sym.loc[mask, 'player_elo'].values - elo_mean) / elo_std), dtype=torch.float32)
    eo = torch.as_tensor(((sym.loc[mask, 'opponent_elo'].values - elo_mean) / elo_std), dtype=torch.float32)
    return HIST_C2_T[pp], LEN_C2_T[pp], HIST_C2_T[op], LEN_C2_T[op], y, ep, eo


def fit_gru_elo_fold(core_ids, eval_ids, calib_ids, test_ids, seed=RANDOM_STATE):
    train_ids_for_scale = core_ids | eval_ids | calib_ids
    scale_mask = sym['SusretId'].isin(train_ids_for_scale)
    elo_mean = float(pd.concat([sym.loc[scale_mask, 'player_elo'], sym.loc[scale_mask, 'opponent_elo']]).mean())
    elo_std = float(pd.concat([sym.loc[scale_mask, 'player_elo'], sym.loc[scale_mask, 'opponent_elo']]).std())

    core = bucket_tensors_elo(core_ids, elo_mean, elo_std)
    ev = bucket_tensors_elo(eval_ids, elo_mean, elo_std)
    calib = bucket_tensors_elo(calib_ids, elo_mean, elo_std)
    test = bucket_tensors_elo(test_ids, elo_mean, elo_std)

    torch.manual_seed(seed)
    model = TwoTowerGRUElo(hidden_size=HIDDEN_SIZE, dropout=DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    n_core = core[4].shape[0]
    rng_local = np.random.RandomState(seed)

    best_brier, best_state, no_improve = np.inf, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = rng_local.permutation(n_core)
        for start in range(0, n_core, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            logit = model(core[0][idx], core[1][idx], core[2][idx], core[3][idx], core[5][idx], core[6][idx])
            loss = criterion(logit, core[4][idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logit = model(ev[0], ev[1], ev[2], ev[3], ev[5], ev[6])
        val_brier = brier_score_loss(ev[4].numpy(), torch.sigmoid(val_logit).numpy())
        if val_brier < best_brier - 1e-5:
            best_brier, best_state, no_improve = val_brier, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        calib_logit = model(calib[0], calib[1], calib[2], calib[3], calib[5], calib[6])
        test_logit = model(test[0], test[1], test[2], test[3], test[5], test[6])
    p_test_cal = calibrate_and_score(calib_logit.numpy(), calib[4].numpy(), test_logit.numpy())
    return p_test_cal, test[4].numpy()


# ============================================================================
# Exp C3 / C4 fit -- generalized from notebooks 10/11 (per-fold rescale, no python-loop rerun)
# ============================================================================
def bucket_tensors_enriched(mask_ids, hist_t, len_t, head_cols):
    mask = sym['SusretId'].isin(mask_ids).values
    pp = torch.as_tensor(player_pos[mask], dtype=torch.long)
    op = torch.as_tensor(opp_pos[mask], dtype=torch.long)
    y = torch.as_tensor(sym.loc[mask, 'Label'].values, dtype=torch.float32)
    hf = torch.as_tensor(sym.loc[mask, head_cols].values, dtype=torch.float32)
    return hist_t[pp], len_t[pp], hist_t[op], len_t[op], y, hf


def fit_gru_enriched_fold(core_ids, eval_ids, calib_ids, test_ids, include_text, seed=RANDOM_STATE):
    train_ids_for_scale = core_ids | eval_ids | calib_ids

    # --- fold-train-only scale constants (never touches this fold's test rows) ---
    scale_mask = sym['SusretId'].isin(train_ids_for_scale)
    elo_mean = float(pd.concat([sym.loc[scale_mask, 'player_elo'], sym.loc[scale_mask, 'opponent_elo']]).mean())
    elo_std = float(pd.concat([sym.loc[scale_mask, 'player_elo'], sym.loc[scale_mask, 'opponent_elo']]).std())
    age_mean = float(pd.concat([sym.loc[scale_mask, 'player_age'], sym.loc[scale_mask, 'opponent_age']]).mean())
    age_std = float(pd.concat([sym.loc[scale_mask, 'player_age'], sym.loc[scale_mask, 'opponent_age']]).std())
    ligakoef_scale = float(player_log.loc[player_log['SusretId'].isin(train_ids_for_scale), 'LigaKoeficjent'].max())

    sym['elo_p_scaled'] = (sym['player_elo'] - elo_mean) / elo_std
    sym['elo_o_scaled'] = (sym['opponent_elo'] - elo_mean) / elo_std
    sym['age_p'] = ((sym['player_age'] - age_mean) / age_std).fillna(0.0)
    sym['age_o'] = ((sym['opponent_age'] - age_mean) / age_std).fillna(0.0)

    if include_text:
        train_rows_ids = sym.loc[scale_mask, ['PlayerId', 'OpponentId']]
        train_player_ids = set(pd.concat([train_rows_ids['PlayerId'], train_rows_ids['OpponentId']]))
        train_scored = scores[scores['IgracId'].isin(train_player_ids)]
        skill_mean = float(train_scored['skill_1_10'].mean())
        skill_std = float(train_scored['skill_1_10'].std())
        sym['text_skill_p'] = ((sym['_raw_skill_p'] - skill_mean) / skill_std).fillna(0.0)
        sym['text_skill_o'] = ((sym['_raw_skill_o'] - skill_mean) / skill_std).fillna(0.0)
        head_cols = HEAD_COLS_C4
    else:
        head_cols = HEAD_COLS_C3

    assert sym[head_cols].isna().sum().sum() == 0, 'NaN leaked into head scalar block!'

    hist_fold = build_hist_c3_fold(ligakoef_scale, elo_mean, elo_std)
    hist_t = torch.from_numpy(hist_fold)
    len_t = torch.from_numpy(C['LEN_C3'])

    core = bucket_tensors_enriched(core_ids, hist_t, len_t, head_cols)
    ev = bucket_tensors_enriched(eval_ids, hist_t, len_t, head_cols)
    calib = bucket_tensors_enriched(calib_ids, hist_t, len_t, head_cols)
    test = bucket_tensors_enriched(test_ids, hist_t, len_t, head_cols)

    torch.manual_seed(seed)
    n_head = len(head_cols)
    model = TwoTowerGRUEnriched(hidden_size=HIDDEN_SIZE, dropout=DROPOUT, n_head=n_head)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    n_core = core[4].shape[0]
    rng_local = np.random.RandomState(seed)

    best_brier, best_state, no_improve = np.inf, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = rng_local.permutation(n_core)
        for start in range(0, n_core, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            logit = model(core[0][idx], core[1][idx], core[2][idx], core[3][idx], core[5][idx])
            loss = criterion(logit, core[4][idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logit = model(ev[0], ev[1], ev[2], ev[3], ev[5])
        val_brier = brier_score_loss(ev[4].numpy(), torch.sigmoid(val_logit).numpy())
        if val_brier < best_brier - 1e-5:
            best_brier, best_state, no_improve = val_brier, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        calib_logit = model(calib[0], calib[1], calib[2], calib[3], calib[5])
        test_logit = model(test[0], test[1], test[2], test[3], test[5])
    p_test_cal = calibrate_and_score(calib_logit.numpy(), calib[4].numpy(), test_logit.numpy())
    return p_test_cal, test[4].numpy()


# ============================================================================
# Main fold loop
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=str, default='1,2,3,4')
    parser.add_argument('--skip-c4', action='store_true', help='skip Exp C4 (needs trenutci_scores.csv)')
    parser.add_argument('--out', type=str, default=str(DATA_DIR / 'walkforward_c3c4_results.json'))
    args = parser.parse_args()
    fold_nums = [int(x) for x in args.folds.split(',')]

    results = []
    t_all = time.time()

    for fspec in FOLDS:
        k = fspec['fold']
        if k not in fold_nums:
            continue
        train_ids, test_ids, cutoff = fspec['train_ids'], fspec['test_ids'], fspec['cutoff']
        t0 = time.time()
        print(f"\n=== FOLD {k} (cutoff {cutoff.date()}, train={len(train_ids):,} matches, "
              f"test={len(test_ids):,} matches) ===", flush=True)

        test_mask = sym['SusretId'].isin(test_ids)
        y_test = sym.loc[test_mask, 'Label'].values

        # Exp 0 -- ELO closed form
        p0 = (1 / (1 + 10 ** (-sym.loc[test_mask, 'elo_diff'] / 400))).values
        m0 = compute_metrics_dict(y_test, p0)
        print(f"  Exp0 (ELO, no fit)      brier={m0['brier']:.5f}  acc={m0['accuracy']:.4f}", flush=True)

        core_ids, eval_ids, calib_ids = carve_train(train_ids)
        core_mask = sym['SusretId'].isin(core_ids)
        eval_mask = sym['SusretId'].isin(eval_ids)
        calib_mask = sym['SusretId'].isin(calib_ids)

        pB, yB = fit_xgb_fold(core_mask, eval_mask, calib_mask, test_mask)
        mB = compute_metrics_dict(yB, pB)
        print(f"  ExpB' (XGB ELO+traj)    brier={mB['brier']:.5f}  acc={mB['accuracy']:.4f}  "
              f"({time.time() - t0:.1f}s elapsed)", flush=True)

        pC2, yC2 = fit_gru_elo_fold(core_ids, eval_ids, calib_ids, test_ids)
        mC2 = compute_metrics_dict(yC2, pC2)
        print(f"  ExpC2 (GRU + ELO)       brier={mC2['brier']:.5f}  acc={mC2['accuracy']:.4f}  "
              f"({time.time() - t0:.1f}s elapsed)", flush=True)

        pC3, yC3 = fit_gru_enriched_fold(core_ids, eval_ids, calib_ids, test_ids, include_text=False)
        mC3 = compute_metrics_dict(yC3, pC3)
        print(f"  ExpC3 (enriched GRU)    brier={mC3['brier']:.5f}  acc={mC3['accuracy']:.4f}  "
              f"({time.time() - t0:.1f}s elapsed)", flush=True)

        exps = [(m0, 'Exp 0 - ELO baseline'), (mB, "Exp B' - ELO-incl XGB + trajectory"),
                (mC2, 'Exp C2 - GRU + ELO'), (mC3, 'Exp C3 - enriched GRU')]
        if not args.skip_c4:
            pC4, yC4 = fit_gru_enriched_fold(core_ids, eval_ids, calib_ids, test_ids, include_text=True)
            mC4 = compute_metrics_dict(yC4, pC4)
            print(f"  ExpC4 (enriched+text)   brier={mC4['brier']:.5f}  acc={mC4['accuracy']:.4f}  "
                  f"({time.time() - t0:.1f}s elapsed)", flush=True)
            exps.append((mC4, 'Exp C4 - enriched GRU + text'))

        for m, expname in exps:
            results.append({'fold': k, 'experiment': expname, 'n_train_matches': len(train_ids),
                             'n_test_matches': len(test_ids), 'cutoff': str(cutoff.date()), **m})

        print(f"  Fold {k} total: {time.time() - t0:.1f}s", flush=True)

    print(f"\nAll requested folds done in {time.time() - t_all:.1f}s", flush=True)

    out_path = Path(args.out)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'Results written to {out_path}')


if __name__ == '__main__':
    main()
