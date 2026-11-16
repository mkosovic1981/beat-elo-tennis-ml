"""
Exp "C5" = C3 + D14 features (plan 2026-11-14-ml-repo-i-tabpfn.md, section 2.8), BEFORE TabPFN (Exp D).

Variants (same seed 42, same carve_train split, same GRU/calibration code as C3 in run_walkforward_c3c4.py):
  C3   baseline (retrained on this workspace, 15 head scalars, 9-dim step features, rest clipped at 90d)
  C5a  C3 + rest without clipping (log1p days, in the step feature AND for the target match)
       + flag rest > 180d + gender flag (liga_tip SM/SF)                      -> 20 head scalars, 10-dim step
  C5b  C5a + relative ELO = ELO - mean ELO of players active in last 90d, same gender, point-in-time
  C5c  C5a + same, window 365d
  C5d  C5a + percentile of ELO among players active in last 90d, same gender
  C5e  C5a + same, window 365d

Fold 4 of the walk-forward == the main split (test from 2025-06-06). Experiment only: does NOT touch serving.

  python run_c5.py --workdir <workdir> --variants C3,C5a,C5b,C5c,C5d,C5e --folds 4
  python run_c5.py --workdir ... --variants C3,C5b --folds 1,2,3
Output: <workdir>/data/c5_preds_fold<k>.parquet (one column per variant, calibrated + raw).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wf_c3c4_common as wf  # noqa: E402

RANDOM_STATE = 42
HIDDEN_SIZE, DROPOUT, WEIGHT_DECAY, LR = 24, 0.3, 1e-4, 2e-3
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 512, 60, 8
REST_SCALE = float(np.log1p(365))
LONG_DAYS = 180

BASE_NEW = ['rest_p', 'rest_o', 'long_p', 'long_o', 'gender']
VARIANTS = {
    'C3': dict(new=[], step10=False),
    'C5a': dict(new=BASE_NEW, step10=True),
    'C5b': dict(new=BASE_NEW + ['rel90_p', 'rel90_o'], step10=True),
    'C5c': dict(new=BASE_NEW + ['rel365_p', 'rel365_o'], step10=True),
    'C5d': dict(new=BASE_NEW + ['pct90_p', 'pct90_o'], step10=True),
    'C5e': dict(new=BASE_NEW + ['pct365_p', 'pct365_o'], step10=True),
}


# ----------------------------------------------------------------------------------------------
# Relative ELO (point-in-time, per gender)
# ----------------------------------------------------------------------------------------------
def build_events(raw):
    """Long event table: one row per (player, match) with ELO AFTER the match, gender from LigaTip."""
    d = raw['Datum'].dt.normalize()
    w = pd.DataFrame({'date': d, 'g': (raw['LigaTip'] == 'SF').astype(int), 'pl': raw['WinnerId'], 'elo': raw['WinnerEloAfter']})
    l = pd.DataFrame({'date': d, 'g': (raw['LigaTip'] == 'SF').astype(int), 'pl': raw['LoserId'], 'elo': raw['LoserEloAfter']})
    return pd.concat([w, l], ignore_index=True)


def pool_stats(events, query_dates, g, window_days):
    """For each (unique) query date D of gender g: sorted array of last-known ELO (after last match)
    of players whose last match is in [D-window, D) -- STRICTLY before D. Returns {D: sorted ndarray}."""
    ev = events[events['g'] == g].sort_values('date', kind='mergesort')
    pl_codes, _ = pd.factorize(ev['pl'])
    n_pl = pl_codes.max() + 1
    last_d = np.full(n_pl, -10 ** 9, dtype=np.int64)
    last_e = np.full(n_pl, np.nan)
    day = (ev['date'].values.astype('datetime64[D]').astype(np.int64))
    elo = ev['elo'].values.astype(float)
    qset = set(int(x) for x in query_dates)
    out = {}
    # iterate over event days in order
    bounds = np.flatnonzero(np.diff(day)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(day)]
    q_sorted = sorted(qset)
    qi = 0
    for s, e in zip(starts, ends):
        d = int(day[s])
        # answer all query dates <= d (state currently holds events < d, and any queries between
        # the previous event day and d see exactly the same state)
        while qi < len(q_sorted) and q_sorted[qi] <= d:
            q = q_sorted[qi]
            m = (last_d >= q - window_days) & (last_d < q)
            out[q] = np.sort(last_e[m])
            qi += 1
        # now apply events of day d (later events of the same player on the same day: last row wins)
        idx = pl_codes[s:e]
        last_d[idx] = d
        last_e[idx] = elo[s:e]
    while qi < len(q_sorted):
        q = q_sorted[qi]
        m = (last_d >= q - window_days) & (last_d < q)
        out[q] = np.sort(last_e[m])
        qi += 1
    return out


def add_relative_elo(sym, events, windows=(90, 365), min_pool=5):
    sday = sym['Datum'].dt.normalize().values.astype('datetime64[D]').astype(np.int64)
    gcol = sym['liga_tip'].values
    for N in windows:
        rel_p = np.zeros(len(sym)); rel_o = np.zeros(len(sym))
        pct_p = np.full(len(sym), 0.5); pct_o = np.full(len(sym), 0.5)
        for g in (0, 1):
            rows_g = np.flatnonzero(gcol == g)
            stats = pool_stats(events, np.unique(sday[rows_g]), g, N)
            order = rows_g[np.argsort(sday[rows_g], kind='mergesort')]
            for D in np.unique(sday[rows_g]):
                pool = stats[int(D)]
                if len(pool) < min_pool:
                    continue
                r = rows_g[sday[rows_g] == D]
                mu = pool.mean()
                for arr_rel, arr_pct, col in ((rel_p, pct_p, 'player_elo'), (rel_o, pct_o, 'opponent_elo')):
                    v = sym[col].values[r]
                    arr_rel[r] = v - mu
                    lo = np.searchsorted(pool, v, side='left')
                    hi = np.searchsorted(pool, v, side='right')
                    arr_pct[r] = (lo + 0.5 * (hi - lo)) / len(pool)
        sym[f'rel{N}_p_raw'] = rel_p
        sym[f'rel{N}_o_raw'] = rel_o
        sym[f'pct{N}_p'] = pct_p - 0.5
        sym[f'pct{N}_o'] = pct_o - 0.5
    return sym


def leakage_check_t15(sym, events, n_check=40, seed=0):
    """T15: (1) brute-force recompute from raw for sampled matches; (2) pool for date D is unchanged
    when ALL events with date >= D are deleted or corrupted (no future information)."""
    rng = np.random.default_rng(seed)
    sday = sym['Datum'].dt.normalize().values.astype('datetime64[D]').astype(np.int64)
    pick = rng.choice(len(sym), n_check, replace=False)
    bad = 0
    for N in (90, 365):
        for i in pick:
            D, g = int(sday[i]), int(sym['liga_tip'].values[i])
            # brute force
            ev = events[(events['g'] == g) & (events['date'].values.astype('datetime64[D]').astype(np.int64) < D)]
            ev = ev.assign(dd=ev['date'].values.astype('datetime64[D]').astype(np.int64))
            last = ev.sort_values(['dd'], kind='mergesort').groupby('pl').tail(1)
            last = last[last['dd'] >= D - N]
            bf = np.sort(last['elo'].values)
            full = pool_stats(events, [D], g, N)[D]
            # truncated / corrupted future
            ev_all_d = events['date'].values.astype('datetime64[D]').astype(np.int64)
            trunc = events[ev_all_d < D]
            corrupt = events.copy()
            fut = ev_all_d >= D
            corrupt.loc[fut, 'elo'] = rng.uniform(500, 2500, fut.sum())
            t1 = pool_stats(trunc, [D], g, N)[D]
            t2 = pool_stats(corrupt, [D], g, N)[D]
            ok = (len(bf) == len(full) == len(t1) == len(t2)) and np.allclose(bf, full) and np.allclose(full, t1) and np.allclose(full, t2)
            bad += (not ok)
    print(f'T15 leakage check (brute force + truncate/corrupt future), {2 * n_check} cases: '
          f'{"PASS" if bad == 0 else f"FAIL ({bad})"}', flush=True)
    assert bad == 0


# ----------------------------------------------------------------------------------------------
# Model (exact copy of C3 model, input_size configurable)
# ----------------------------------------------------------------------------------------------
class TwoTowerGRUEnriched(nn.Module):
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
        return self.head(torch.cat([h_p, h_o, h_p - h_o, head_feats], dim=1)).squeeze(-1)


def calibrate_and_score(calib_logit_np, calib_y_np, test_logit_np):
    """EXACT copy of run_walkforward_c3c4.calibrate_and_score. Returns (p_cal, p_raw)."""
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
        return iso.predict(p_test_raw), p_test_raw
    return torch.sigmoid(torch.from_numpy(test_logit_np) / T_value).numpy(), p_test_raw


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--variants', default='C3,C5a,C5b,C5c,C5d,C5e')
    ap.add_argument('--folds', default='4')
    ap.add_argument('--seed', type=int, default=RANDOM_STATE)
    a = ap.parse_args()
    work = Path(a.workdir)
    wf.DATA_DIR = work / 'data'
    DATA_DIR = wf.DATA_DIR
    variants = a.variants.split(',')
    fold_nums = [int(x) for x in a.folds.split(',')]

    C = wf.load_common()
    sym, FOLDS, player_log = C['sym'], C['FOLDS'], C['player_log']
    player_pos, opp_pos = C['player_pos'], C['opp_pos']
    carve_train = C['carve_train']
    HEAD_COLS_C3 = C['HEAD_COLS_C3']
    HIST_C3_RAW, LEN_C3, IS_PH_C3 = C['HIST_C3_RAW'], C['LEN_C3'], C['IS_PH_C3']

    # ---- target-match rest features (unclipped, point-in-time: from player_log) ----
    d_p = player_log['DaysSinceLastMatch'].values[player_pos].astype(float)
    d_o = player_log['DaysSinceLastMatch'].values[opp_pos].astype(float)
    sym['rest_p'] = np.nan_to_num(np.log1p(d_p) / REST_SCALE, nan=0.0)
    sym['rest_o'] = np.nan_to_num(np.log1p(d_o) / REST_SCALE, nan=0.0)
    sym['long_p'] = (d_p > LONG_DAYS).astype(np.float32)
    sym['long_o'] = (d_o > LONG_DAYS).astype(np.float32)
    sym['gender'] = sym['liga_tip'].astype(np.float32)
    sym['_rest_days_p'] = d_p  # kept for evaluation only (buckets)

    # ---- relative ELO ----
    raw = pd.read_csv(DATA_DIR / 'matches_export.csv', parse_dates=['Datum'])
    raw['Datum'] = pd.to_datetime(raw['Datum']).dt.tz_localize(None)
    events = build_events(raw)
    t = time.time()
    sym = add_relative_elo(sym, events)
    wf.lap(f'relative ELO computed ({time.time() - t:.1f}s); pool-size fallback rows (rel==0 & pct==0.5): '
           f"{((sym['rel90_p_raw'] == 0) & (sym['pct90_p'] == 0)).mean():.2%}")
    leakage_check_t15(sym, events)

    # ---- 10-dim step history: unclipped log1p rest (dim 2) + >180d flag (dim 9) ----
    n_pl = len(player_log)
    days = player_log['DaysSinceLastMatch'].values.astype(float)
    step_rest = np.nan_to_num(np.log1p(days) / REST_SCALE, nan=0.0).astype(np.float32)
    step_long = (np.nan_to_num(days, nan=0.0) > LONG_DAYS).astype(np.float32)
    first_idx = player_log.groupby('IgracId')['_gpos'].transform('min').values
    HIST10_RAW = np.zeros((n_pl, HIST_C3_RAW.shape[1], 10), dtype=np.float32)
    HIST10_RAW[:, :, :9] = HIST_C3_RAW
    MAXL = HIST_C3_RAW.shape[1]
    for i in range(n_pl):
        if IS_PH_C3[i]:
            continue
        lo = max(first_idx[i], i - MAXL)
        L = i - lo
        HIST10_RAW[i, :L, 2] = step_rest[lo:i]
        HIST10_RAW[i, :L, 9] = step_long[lo:i]
        assert L == LEN_C3[i]
    wf.lap('HIST10 built')

    def hist_fold(raw_hist, ligakoef_scale, elo_mean, elo_std):
        h = raw_hist.copy()
        h[:, :, 3] = h[:, :, 3] / ligakoef_scale
        h[:, :, 6] = np.clip((h[:, :, 6] - elo_mean) / elo_std, -4, 4)
        h[IS_PH_C3, 0, 6] = 0.0
        return h

    def bucket(mask_ids, hist_t, len_t, head_cols):
        mask = sym['SusretId'].isin(mask_ids).values
        pp = torch.as_tensor(player_pos[mask], dtype=torch.long)
        op = torch.as_tensor(opp_pos[mask], dtype=torch.long)
        y = torch.as_tensor(sym.loc[mask, 'Label'].values, dtype=torch.float32)
        hf = torch.as_tensor(sym.loc[mask, head_cols].values, dtype=torch.float32)
        return hist_t[pp], len_t[pp], hist_t[op], len_t[op], y, hf

    def fit_variant(name, core_ids, eval_ids, calib_ids, test_ids, seed=RANDOM_STATE):
        spec = VARIANTS[name]
        scale_ids = core_ids | eval_ids | calib_ids
        sm = sym['SusretId'].isin(scale_ids)
        both = lambda c1, c2: pd.concat([sym.loc[sm, c1], sym.loc[sm, c2]])
        elo_mean = float(both('player_elo', 'opponent_elo').mean())
        elo_std = float(both('player_elo', 'opponent_elo').std())
        age_mean = float(both('player_age', 'opponent_age').mean())
        age_std = float(both('player_age', 'opponent_age').std())
        ligakoef_scale = float(player_log.loc[player_log['SusretId'].isin(scale_ids), 'LigaKoeficjent'].max())
        sym['elo_p_scaled'] = (sym['player_elo'] - elo_mean) / elo_std
        sym['elo_o_scaled'] = (sym['opponent_elo'] - elo_mean) / elo_std
        sym['age_p'] = ((sym['player_age'] - age_mean) / age_std).fillna(0.0)
        sym['age_o'] = ((sym['opponent_age'] - age_mean) / age_std).fillna(0.0)
        for N in (90, 365):  # fold-train std only
            sym[f'rel{N}_p'] = sym[f'rel{N}_p_raw'] / elo_std
            sym[f'rel{N}_o'] = sym[f'rel{N}_o_raw'] / elo_std
        head_cols = HEAD_COLS_C3 + spec['new']
        assert sym[head_cols].isna().sum().sum() == 0
        raw_hist = HIST10_RAW if spec['step10'] else HIST_C3_RAW
        hist_t = torch.from_numpy(hist_fold(raw_hist, ligakoef_scale, elo_mean, elo_std))
        len_t = torch.from_numpy(LEN_C3)
        core, ev, calib, test = (bucket(s, hist_t, len_t, head_cols) for s in (core_ids, eval_ids, calib_ids, test_ids))
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = TwoTowerGRUEnriched(input_size=raw_hist.shape[2], hidden_size=HIDDEN_SIZE, dropout=DROPOUT,
                                    n_head=len(head_cols))
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        crit = nn.BCEWithLogitsLoss()
        n_core = core[4].shape[0]
        rng = np.random.RandomState(seed)
        best, best_state, bad = np.inf, None, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            perm = rng.permutation(n_core)
            for s in range(0, n_core, BATCH_SIZE):
                idx = perm[s:s + BATCH_SIZE]
                opt.zero_grad()
                loss = crit(model(core[0][idx], core[1][idx], core[2][idx], core[3][idx], core[5][idx]), core[4][idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vb = brier_score_loss(ev[4].numpy(), torch.sigmoid(model(ev[0], ev[1], ev[2], ev[3], ev[5])).numpy())
            if vb < best - 1e-5:
                best, best_state, bad = vb, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
            if bad >= PATIENCE:
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            cl = model(calib[0], calib[1], calib[2], calib[3], calib[5])
            tl = model(test[0], test[1], test[2], test[3], test[5])
        p_cal, p_raw = calibrate_and_score(cl.numpy(), calib[4].numpy(), tl.numpy())
        return p_cal, p_raw, test[4].numpy(), epoch

    for fspec in FOLDS:
        k = fspec['fold']
        if k not in fold_nums:
            continue
        train_ids, test_ids = fspec['train_ids'], fspec['test_ids']
        core_ids, eval_ids, calib_ids = carve_train(train_ids)
        test_mask = sym['SusretId'].isin(test_ids).values
        meta = sym.loc[test_mask, ['SusretId', 'Datum', 'SezonaId', 'LigaId', 'PlayerId', 'OpponentId', 'Label',
                                   'Perspective', 'liga_tip', '_rest_days_p']].reset_index(drop=True)
        meta['fold'] = k
        out_path = DATA_DIR / (f'c5_preds_fold{k}.parquet' if a.seed == RANDOM_STATE else f'c5_preds_fold{k}_seed{a.seed}.parquet')
        if out_path.exists():
            prev = pd.read_parquet(out_path)
            assert (prev['SusretId'].values == meta['SusretId'].values).all()
            for c in prev.columns:
                if c.startswith('p_'):
                    meta[c] = prev[c].values
        for v in variants:
            t0 = time.time()
            p_cal, p_raw, y, ep = fit_variant(v, core_ids, eval_ids, calib_ids, test_ids, seed=a.seed)
            assert (y == meta['Label'].values).all()
            meta[f'p_{v}_calibrated'] = p_cal
            meta[f'p_{v}_raw'] = p_raw
            print(f'fold {k} {v}: Brier={brier_score_loss(y, p_cal):.5f} epochs={ep} ({time.time() - t0:.0f}s)', flush=True)
            meta.to_parquet(out_path, index=False)  # incremental save
    wf.lap('done')


if __name__ == '__main__':
    main()
