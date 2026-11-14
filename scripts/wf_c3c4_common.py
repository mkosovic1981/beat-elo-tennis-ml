"""
Shared setup for the C3/C4 walk-forward experiment (notebook 12).

Builds, ONCE (fold-independent):
  - `sym` (features.parquet, single canonical sort by Datum/SusretId/Perspective -- used for
    Exp0/B'/C2/C3/C4 alike, masked by SusretId per fold)
  - FOLDS (5-block / 4-fold expanding window, EXACT copy of notebook 09 Step 2)
  - EXP_B_PRIME_COLS (9 razina-trajectory diff cols + EXP_B_COLS, EXACT copy of notebook 09 Step 3)
  - player_log (long-format player-match log with OppEloBefore/H2H, EXACT copy of notebook 10 Steps 2-5)
  - HIST_C2/LEN_C2 (7-dim, ELO-free per-step, LIGAKOEF_SCALE_GLOBAL fixed -- fold-independent,
    EXACT copy of notebook 09 Step 4)
  - HIST_C3_RAW/LEN_C3/IS_PH_C3 (9-dim per-step, dims 3 (LigaKoef) and 6 (OppEloBefore) left
    UNSCALED/raw -- fold-specific scaling applied vectorized per fold in `build_hist_c3_fold`,
    avoiding an O(n_rows) python-loop re-run per fold; everything else identical to notebook 10
    Step 6)
  - player_pos/opp_pos (index arrays resolving sym rows onto player_log history positions)
  - Fixed (fold-independent) head-scalar raw components: cwr/forma/mc/h2h_wr/h2h_total_scaled/
    h2h_sets_margin_scaled, plus raw text-skill/confidence/has_text lookups (`_raw_skill_p/o`)
  - carve_train (chronological TRAIN-only core/eval/calib split, EXACT copy of notebook 09)

Everything that legitimately differs per fold (ELO_MEAN/STD, AGE_MEAN/STD, LIGAKOEF_SCALE,
TEXT_SKILL_MEAN/STD) is computed in `run_walkforward_c3c4.py` from each fold's OWN train_ids,
never that fold's test rows -- the same train-only hygiene notebooks 08/09/10/11 already
established, just correctly re-applied per fold (same reasoning notebook 09 used for Exp C2's
ELO_MEAN/STD, extended here to the extra C3/C4 scale constants).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

T0 = time.time()


def lap(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)


def load_common():
    # ---- Step 1: features.parquet (single canonical sort, used for ALL models) ----
    sym = pd.read_parquet(DATA_DIR / 'features.parquet')
    with open(DATA_DIR / 'feature_tags.json', encoding='utf-8') as f:
        TAGS = json.load(f)
    EXP_B_COLS = TAGS['experiment_b_cols']
    sym['liga_tip'] = sym['liga_tip'].map({'SM': 0, 'SF': 1}).astype(int)
    sym = sym.sort_values(['Datum', 'SusretId', 'Perspective'], kind='mergesort').reset_index(drop=True)
    lap(f'features.parquet loaded: {sym.shape}')
    BEST_PARAMS_B = {'max_depth': 4, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'learning_rate': 0.05}

    # ---- Step 2: 5-block / 4-fold expanding-window walk-forward split (EXACT copy of 09) ----
    match_order_all = (sym[['SusretId', 'Datum']].drop_duplicates()
                       .sort_values(['Datum', 'SusretId'], kind='mergesort').reset_index(drop=True))
    n_matches_all = len(match_order_all)
    N_BLOCKS = 5
    block_edges = np.linspace(0, n_matches_all, N_BLOCKS + 1).astype(int)
    blocks = [match_order_all.iloc[block_edges[i]:block_edges[i + 1]] for i in range(N_BLOCKS)]

    FOLDS = []
    for k in range(1, N_BLOCKS):
        train_ids = set(pd.concat(blocks[:k])['SusretId'])
        test_ids = set(blocks[k]['SusretId'])
        cutoff_date = blocks[k]['Datum'].min()
        FOLDS.append({'fold': k, 'train_ids': train_ids, 'test_ids': test_ids, 'cutoff': cutoff_date})
        print(f"Fold {k}: train={len(train_ids):,} matches, test={len(test_ids):,} matches, "
              f"cutoff={cutoff_date.date()}")
    assert FOLDS[-1]['train_ids'].union(FOLDS[-1]['test_ids']).issuperset(set(match_order_all['SusretId']))
    assert len(FOLDS[-1]['train_ids']) == TAGS['n_matches_train']
    assert len(FOLDS[-1]['test_ids']) == TAGS['n_matches_test']
    lap('OK -- fold 4 reproduces the main 80/20 split boundary (sanity anchor).')

    # ---- Step 3: raw matches export (needed for trajectory + GRU player_log) ----
    raw = pd.read_csv(DATA_DIR / 'matches_export.csv', parse_dates=['Datum'])
    raw['Datum'] = pd.to_datetime(raw['Datum']).dt.tz_localize(None)
    raw = raw.sort_values(['Datum', 'SusretId']).reset_index(drop=True)
    lap(f'matches_export.csv loaded: {raw.shape}')

    # ---- Step 4: razina-trajectory features -> EXP_B_PRIME_COLS (EXACT copy of 07/09) ----
    common_cols = ['SusretId', 'Datum', 'SezonaId', 'LigaId', 'LigaTip', 'LigaKoeficjent', 'Lokacija']
    winner_rows_t = raw[common_cols + ['WinnerId', 'LoserId']].rename(
        columns={'WinnerId': 'IgracId', 'LoserId': 'OpponentId'}).assign(Won=1)
    loser_rows_t = raw[common_cols + ['LoserId', 'WinnerId']].rename(
        columns={'LoserId': 'IgracId', 'WinnerId': 'OpponentId'}).assign(Won=0)
    player_log_t = pd.concat([winner_rows_t, loser_rows_t], ignore_index=True)
    player_log_t = player_log_t.sort_values(['IgracId', 'Datum', 'SusretId'], kind='mergesort').reset_index(drop=True)

    g_t = player_log_t.groupby('IgracId', sort=False)
    player_log_t['RazinaAgo5'] = g_t['LigaKoeficjent'].shift(5)
    player_log_t['RazinaAgo10'] = g_t['LigaKoeficjent'].shift(10)
    player_log_t['RazinaAgo20'] = g_t['LigaKoeficjent'].shift(20)

    def rolling_slope(frame, t_col, y_col, window, min_periods):
        t = frame[t_col].astype(float)
        y = frame[y_col].astype(float)
        ty, tt = t * y, t * t
        grp = frame['IgracId']
        Sy = y.groupby(grp).transform(lambda s: s.rolling(window, min_periods=min_periods).sum())
        St = t.groupby(grp).transform(lambda s: s.rolling(window, min_periods=min_periods).sum())
        Sty = ty.groupby(grp).transform(lambda s: s.rolling(window, min_periods=min_periods).sum())
        Stt = tt.groupby(grp).transform(lambda s: s.rolling(window, min_periods=min_periods).sum())
        n = y.groupby(grp).transform(lambda s: s.rolling(window, min_periods=min_periods).count())
        denom = (n * Stt - St ** 2)
        return (n * Sty - St * Sy) / denom.replace(0, np.nan)

    player_log_t['_pos'] = g_t.cumcount()
    WINDOW = 10
    player_log_t['RazinaSlope10'] = rolling_slope(player_log_t, '_pos', 'LigaKoeficjent', WINDOW, WINDOW).groupby(player_log_t['IgracId']).shift(1)
    player_log_t['FormaSlope10'] = rolling_slope(player_log_t, '_pos', 'Won', WINDOW, WINDOW).groupby(player_log_t['IgracId']).shift(1)

    prev_razina = g_t['LigaKoeficjent'].shift(1)
    player_log_t['_changed'] = (player_log_t['LigaKoeficjent'] != prev_razina).astype(int)
    player_log_t['_block_id'] = player_log_t.groupby('IgracId')['_changed'].cumsum()
    player_log_t['RazinaStint'] = player_log_t.groupby(['IgracId', '_block_id']).cumcount()

    player_log_t['RazinaBestEver'] = player_log_t.groupby('IgracId')['LigaKoeficjent'].apply(
        lambda s: s.cummin().shift(1)).reset_index(level=0, drop=True)
    player_log_t['RazinaCurrentMinusBest'] = player_log_t['LigaKoeficjent'] - player_log_t['RazinaBestEver']

    recent_min5 = player_log_t.groupby('IgracId')['LigaKoeficjent'].apply(
        lambda s: s.rolling(5, min_periods=1).min().shift(1)).reset_index(level=0, drop=True)
    recent_max5 = player_log_t.groupby('IgracId')['LigaKoeficjent'].apply(
        lambda s: s.rolling(5, min_periods=1).max().shift(1)).reset_index(level=0, drop=True)
    promo = pd.Series(0.0, index=player_log_t.index)
    promo = promo.where(~(player_log_t['LigaKoeficjent'] < recent_min5), -1)
    promo = promo.where(~(player_log_t['LigaKoeficjent'] > recent_max5), 1)
    promo = promo.where(recent_min5.notna(), np.nan)
    player_log_t['RazinaPromotionSignal'] = promo

    TRAJ_RAW_COLS = ['RazinaAgo5', 'RazinaAgo10', 'RazinaAgo20', 'RazinaSlope10', 'RazinaStint',
                      'RazinaBestEver', 'RazinaCurrentMinusBest', 'RazinaPromotionSignal', 'FormaSlope10']
    lap('Trajectory features computed (vectorized, fold-independent)')

    traj = player_log_t[['SusretId', 'IgracId'] + TRAJ_RAW_COLS].copy()
    own = sym[['SusretId', 'PlayerId']].merge(
        traj, left_on=['SusretId', 'PlayerId'], right_on=['SusretId', 'IgracId'], how='left', validate='one_to_one'
    ).drop(columns=['IgracId'])
    opp = sym[['SusretId', 'OpponentId']].merge(
        traj, left_on=['SusretId', 'OpponentId'], right_on=['SusretId', 'IgracId'], how='left', validate='one_to_one'
    ).drop(columns=['IgracId'])
    assert (own['SusretId'].values == sym['SusretId'].values).all()
    assert (opp['SusretId'].values == sym['SusretId'].values).all()

    TRAJ_DIFF_COLS = []
    for raw_col in TRAJ_RAW_COLS:
        diff_col = f'{raw_col}_diff'
        sym[diff_col] = own[raw_col].values - opp[raw_col].values
        TRAJ_DIFF_COLS.append(diff_col)

    EXP_B_PRIME_COLS = EXP_B_COLS + TRAJ_DIFF_COLS
    assert 'elo_diff' in EXP_B_PRIME_COLS
    lap(f"Exp B' columns ready: {len(EXP_B_PRIME_COLS)} ({len(EXP_B_COLS)} base + {len(TRAJ_DIFF_COLS)} trajectory)")

    # ---- Step 5: long-format player_log for GRU models, extended w/ OppEloBefore + H2H (10) ----
    common_cols2 = ['SusretId', 'Datum', 'SezonaId', 'LigaId', 'LigaTip', 'LigaKoeficjent', 'Lokacija']
    winner_rows = raw[common_cols2 + [
        'WinnerId', 'LoserId', 'WinnerEloBefore', 'LoserEloBefore',
        'W_Set1', 'W_Set2', 'W_Set3', 'L_Set1', 'L_Set2', 'L_Set3',
    ]].rename(columns={
        'WinnerId': 'IgracId', 'LoserId': 'OpponentId',
        'WinnerEloBefore': 'MyEloBefore', 'LoserEloBefore': 'OppEloBefore',
        'W_Set1': 'MySet1', 'W_Set2': 'MySet2', 'W_Set3': 'MySet3',
        'L_Set1': 'OppSet1', 'L_Set2': 'OppSet2', 'L_Set3': 'OppSet3',
    }).assign(Won=1)
    loser_rows = raw[common_cols2 + [
        'LoserId', 'WinnerId', 'LoserEloBefore', 'WinnerEloBefore',
        'L_Set1', 'L_Set2', 'L_Set3', 'W_Set1', 'W_Set2', 'W_Set3',
    ]].rename(columns={
        'LoserId': 'IgracId', 'WinnerId': 'OpponentId',
        'LoserEloBefore': 'MyEloBefore', 'WinnerEloBefore': 'OppEloBefore',
        'L_Set1': 'MySet1', 'L_Set2': 'MySet2', 'L_Set3': 'MySet3',
        'W_Set1': 'OppSet1', 'W_Set2': 'OppSet2', 'W_Set3': 'OppSet3',
    }).assign(Won=0)
    player_log = pd.concat([winner_rows, loser_rows], ignore_index=True)
    player_log['LokacijaNorm'] = player_log['Lokacija'].str.strip().str.lower()
    player_log = player_log.sort_values(['IgracId', 'Datum', 'SusretId'], kind='mergesort').reset_index(drop=True)

    g = player_log.groupby('IgracId', sort=False)
    player_log['MatchCountBefore'] = g.cumcount()
    player_log['HomeCountBefore'] = player_log.groupby(['IgracId', 'LokacijaNorm'], sort=False).cumcount()
    player_log['HomeShareBefore'] = np.where(
        player_log['MatchCountBefore'] > 0,
        player_log['HomeCountBefore'] / player_log['MatchCountBefore'], np.nan)
    player_log['DaysSinceLastMatch'] = player_log.groupby('IgracId')['Datum'].diff().dt.days
    player_log['CareerWinRateBefore'] = g['Won'].transform(lambda s: s.shift().expanding().mean())

    for c in ['MySet1', 'MySet2', 'MySet3', 'OppSet1', 'OppSet2', 'OppSet3']:
        player_log[c + '_f'] = player_log[c].fillna(0)
    player_log['SetMargin'] = (
        (player_log['MySet1_f'] + player_log['MySet2_f'] + player_log['MySet3_f'])
        - (player_log['OppSet1_f'] + player_log['OppSet2_f'] + player_log['OppSet3_f']))
    player_log = player_log.drop(columns=[c + '_f' for c in ['MySet1', 'MySet2', 'MySet3', 'OppSet1', 'OppSet2', 'OppSet3']])

    check_w = raw[['SusretId', 'WinnerId', 'WinnerMatchCountBefore']].rename(columns={'WinnerId': 'IgracId', 'WinnerMatchCountBefore': 'Exp'})
    check_l = raw[['SusretId', 'LoserId', 'LoserMatchCountBefore']].rename(columns={'LoserId': 'IgracId', 'LoserMatchCountBefore': 'Exp'})
    chk = pd.concat([check_w, check_l])
    mm = player_log.merge(chk, on=['SusretId', 'IgracId'], how='left')
    assert (mm['MatchCountBefore'] != mm['Exp']).sum() == 0, 'pandas point-in-time ordering mismatch!'

    opp_cwr = player_log[['SusretId', 'IgracId', 'CareerWinRateBefore']].rename(
        columns={'IgracId': 'OpponentId', 'CareerWinRateBefore': 'OpponentCareerWinRateBefore'})
    player_log = player_log.merge(opp_cwr, on=['SusretId', 'OpponentId'], how='left', validate='many_to_one')

    h2h_g = player_log.groupby(['IgracId', 'OpponentId'], sort=False)
    player_log['H2HMeetingsBefore'] = h2h_g.cumcount()
    player_log['WasH2H'] = (player_log['H2HMeetingsBefore'] > 0).astype(np.float32)
    player_log['H2HSetsMarginBefore'] = h2h_g['SetMargin'].transform(lambda s: s.cumsum().shift(fill_value=0.0))

    # leakage spot-check (mandatory, reapplied)
    sample_pair = player_log.loc[player_log['H2HMeetingsBefore'] >= 2].iloc[0]
    pair_hist = player_log[
        (player_log['IgracId'] == sample_pair['IgracId'])
        & (player_log['OpponentId'] == sample_pair['OpponentId'])
        & (player_log['Datum'] < sample_pair['Datum'])
    ]
    assert len(pair_hist) == sample_pair['H2HMeetingsBefore']
    assert abs(pair_hist['SetMargin'].sum() - sample_pair['H2HSetsMarginBefore']) < 1e-6
    lap(f'player_log built: {player_log.shape} (H2H leakage spot-check OK)')

    # ---- Step 6: index lookups ----
    player_log['_gpos'] = np.arange(len(player_log))
    first_idx = player_log.groupby('IgracId')['_gpos'].min().to_dict()
    idx_lookup = dict(zip(zip(player_log['IgracId'], player_log['SusretId']), player_log['_gpos']))

    # ---- Step 7a: 7-dim step feats for Exp C2 (ELO-free per-step, LIGAKOEF_SCALE_GLOBAL fixed
    #      -- fold-independent by construction, same reasoning as notebook 09) ----
    MAX_LEN = 24
    LIGAKOEF_SCALE_GLOBAL = float(player_log['LigaKoeficjent'].max())
    SETMARGIN_SCALE, DAYSREST_CLIP = 12.0, 90.0

    n_pl = len(player_log)
    days_rest = player_log['DaysSinceLastMatch'].fillna(DAYSREST_CLIP).clip(upper=DAYSREST_CLIP).values

    step_feats_c2 = np.zeros((n_pl, 7), dtype=np.float32)
    step_feats_c2[:, 0] = player_log['Won'].values
    step_feats_c2[:, 1] = np.clip(player_log['SetMargin'].values / SETMARGIN_SCALE, -3, 3)
    step_feats_c2[:, 2] = days_rest / DAYSREST_CLIP
    step_feats_c2[:, 3] = player_log['LigaKoeficjent'].values / LIGAKOEF_SCALE_GLOBAL
    step_feats_c2[:, 4] = player_log['HomeShareBefore'].fillna(0.0).values
    step_feats_c2[:, 5] = player_log['OpponentCareerWinRateBefore'].fillna(0.5).values
    step_feats_c2[:, 6] = 0.0
    PLACEHOLDER_C2 = np.zeros(7, dtype=np.float32)
    PLACEHOLDER_C2[5] = 0.5
    PLACEHOLDER_C2[6] = 1.0

    # ---- Step 7b: 9-dim RAW step feats for Exp C3/C4 (dims 3, 6 left UNSCALED -- fold-specific
    #      scaling applied vectorized in build_hist_c3_fold, no python-loop rerun per fold) ----
    step_feats_c3_raw = np.zeros((n_pl, 9), dtype=np.float32)
    step_feats_c3_raw[:, 0] = player_log['Won'].values
    step_feats_c3_raw[:, 1] = np.clip(player_log['SetMargin'].values / SETMARGIN_SCALE, -3, 3)
    step_feats_c3_raw[:, 2] = days_rest / DAYSREST_CLIP
    step_feats_c3_raw[:, 3] = player_log['LigaKoeficjent'].values  # RAW -- scaled per fold
    step_feats_c3_raw[:, 4] = player_log['HomeShareBefore'].fillna(0.0).values
    step_feats_c3_raw[:, 5] = player_log['OpponentCareerWinRateBefore'].fillna(0.5).values
    step_feats_c3_raw[:, 6] = player_log['OppEloBefore'].values  # RAW -- scaled per fold
    step_feats_c3_raw[:, 7] = player_log['WasH2H'].values
    step_feats_c3_raw[:, 8] = 0.0
    PLACEHOLDER_C3_RAW = np.zeros(9, dtype=np.float32)
    PLACEHOLDER_C3_RAW[5] = 0.5
    PLACEHOLDER_C3_RAW[6] = 0.0  # dummy raw -- overridden to neutral 0 post-scale (IS_PH mask)
    PLACEHOLDER_C3_RAW[7] = 0.0
    PLACEHOLDER_C3_RAW[8] = 1.0

    igrac_arr, susret_arr = player_log['IgracId'].values, player_log['SusretId'].values

    def assemble_hist(step_feats, placeholder_vec):
        n_feats = step_feats.shape[1]
        hist = np.zeros((n_pl, MAX_LEN, n_feats), dtype=np.float32)
        lens = np.zeros(n_pl, dtype=np.int64)
        is_ph = np.zeros(n_pl, dtype=bool)
        for i in range(n_pl):
            gpos = idx_lookup[(igrac_arr[i], susret_arr[i])]
            start = first_idx[igrac_arr[i]]
            p = gpos - start
            if p == 0:
                hist[i, 0] = placeholder_vec
                lens[i] = 1
                is_ph[i] = True
            else:
                lo = max(start, gpos - MAX_LEN)
                chunk = step_feats[lo:gpos]
                length = chunk.shape[0]
                hist[i, :length] = chunk
                lens[i] = length
        return hist, lens, is_ph

    t_hist = time.time()
    HIST_C2, LEN_C2, _ = assemble_hist(step_feats_c2, PLACEHOLDER_C2)
    lap(f'HIST_C2 assembled ({time.time() - t_hist:.1f}s)')
    t_hist = time.time()
    HIST_C3_RAW, LEN_C3, IS_PH_C3 = assemble_hist(step_feats_c3_raw, PLACEHOLDER_C3_RAW)
    lap(f'HIST_C3_RAW assembled ({time.time() - t_hist:.1f}s)')

    # leakage spot-check (mandatory, reapplied)
    p_check = (player_log['_gpos'] - player_log['IgracId'].map(first_idx))
    assert (p_check == player_log['MatchCountBefore']).all()
    lap('OK -- group contiguity / history leakage spot-check passed.')

    # ---- Step 8: resolve sym rows onto history positions ----
    player_pos = np.array([idx_lookup[(pid, sid)] for pid, sid in zip(sym['PlayerId'], sym['SusretId'])])
    opp_pos = np.array([idx_lookup[(oid, sid)] for oid, sid in zip(sym['OpponentId'], sym['SusretId'])])
    lap(f'Resolved player/opponent history positions for {len(sym):,} sym rows.')

    # ---- Step 9: fixed (fold-independent) head-scalar raw components ----
    sym['cwr_p'] = sym['player_career_winrate_before'].fillna(0.5)
    sym['cwr_o'] = sym['opponent_career_winrate_before'].fillna(0.5)
    sym['forma_p'] = sym['player_forma_last10'].fillna(0.5)
    sym['forma_o'] = sym['opponent_forma_last10'].fillna(0.5)
    MATCHCOUNT_SCALE = 100.0
    sym['mc_p'] = np.clip(sym['player_match_count_before'] / MATCHCOUNT_SCALE, 0, 4)
    sym['mc_o'] = np.clip(sym['opponent_match_count_before'] / MATCHCOUNT_SCALE, 0, 4)
    H2HTOTAL_SCALE, H2H_SETSMARGIN_SCALE = 10.0, 20.0
    sym['h2h_wr'] = sym['player_h2h_win_rate_before']
    sym['h2h_total_scaled'] = np.clip(sym['h2h_total_before'] / H2HTOTAL_SCALE, 0, 3)
    sym['h2h_sets_margin_scaled'] = np.clip(
        player_log['H2HSetsMarginBefore'].values[player_pos] / H2H_SETSMARGIN_SCALE, -3, 3)
    sym['text_p'] = 0.0  # Exp C3 neutral slot (fixed, no fold dependency)
    sym['text_o'] = 0.0

    # text-skill raw lookups (has_text / confidence are fold-independent; skill_1_10 scaling
    # constant IS fold-train dependent -- computed per fold in run_walkforward_c3c4.py)
    scores_path = DATA_DIR / 'trenutci_scores.csv'
    if scores_path.exists():
        scores = pd.read_csv(scores_path)[['IgracId', 'skill_1_10', 'confidence']]
    else:  # no free-text scores for this dataset (e.g. the Nov-2026 export) -> Exp C4 is skipped
        scores = pd.DataFrame({'IgracId': [], 'skill_1_10': [], 'confidence': []})
    skill_lookup = scores.set_index('IgracId')['skill_1_10'].to_dict()
    conf_lookup = scores.set_index('IgracId')['confidence'].to_dict()
    for who, id_col in [('p', 'PlayerId'), ('o', 'OpponentId')]:
        raw_skill = sym[id_col].map(skill_lookup)
        sym[f'_raw_skill_{who}'] = raw_skill
        sym[f'has_text_{who}'] = raw_skill.notna().astype(np.float32)
        sym[f'text_conf_{who}'] = sym[id_col].map(conf_lookup).fillna(0.0)
    lap(f"has_text_p coverage over ALL sym rows: {sym['has_text_p'].mean():.2%}")

    HEAD_COLS_C3 = ['elo_p_scaled', 'elo_o_scaled', 'cwr_p', 'cwr_o', 'forma_p', 'forma_o',
                    'mc_p', 'mc_o', 'age_p', 'age_o', 'text_p', 'text_o',
                    'h2h_wr', 'h2h_total_scaled', 'h2h_sets_margin_scaled']
    HEAD_COLS_C4 = ['elo_p_scaled', 'elo_o_scaled', 'cwr_p', 'cwr_o', 'forma_p', 'forma_o',
                    'mc_p', 'mc_o', 'age_p', 'age_o',
                    'text_skill_p', 'text_skill_o', 'text_conf_p', 'text_conf_o', 'has_text_p', 'has_text_o',
                    'h2h_wr', 'h2h_total_scaled', 'h2h_sets_margin_scaled']
    assert len(HEAD_COLS_C3) == 15 and len(HEAD_COLS_C4) == 19

    # ---- carve_train helper (EXACT copy of notebook 09) ----
    def carve_train(train_match_ids, calib_frac=0.15, early_stop_frac=0.10):
        fold_order = match_order_all[match_order_all['SusretId'].isin(train_match_ids)].reset_index(drop=True)
        n = len(fold_order)
        n_calib = max(1, int(round(n * calib_frac)))
        fit_order = fold_order.iloc[: n - n_calib].reset_index(drop=True)
        calib_ids = set(fold_order.iloc[n - n_calib:]['SusretId'])
        n_fit = len(fit_order)
        n_eval = max(1, int(round(n_fit * early_stop_frac)))
        core_ids = set(fit_order.iloc[: n_fit - n_eval]['SusretId'])
        eval_ids = set(fit_order.iloc[n_fit - n_eval:]['SusretId'])
        return core_ids, eval_ids, calib_ids

    def build_hist_c3_fold(ligakoef_scale, elo_mean, elo_std):
        """Vectorized per-fold rescale of dims 3 (LigaKoef) and 6 (OppEloBefore) -- no python
        loop. Placeholder rows' single (position-0) OppEloBefore entry is forced back to the
        neutral 0.0 (mean-centered) regardless of this fold's ELO_MEAN/STD, matching the
        notebook 10/11 convention exactly."""
        hist = HIST_C3_RAW.copy()
        hist[:, :, 3] = hist[:, :, 3] / ligakoef_scale
        hist[:, :, 6] = np.clip((hist[:, :, 6] - elo_mean) / elo_std, -4, 4)
        hist[IS_PH_C3, 0, 6] = 0.0
        return hist

    return {
        'sym': sym, 'TAGS': TAGS, 'FOLDS': FOLDS, 'match_order_all': match_order_all,
        'EXP_B_PRIME_COLS': EXP_B_PRIME_COLS, 'BEST_PARAMS_B': BEST_PARAMS_B,
        'player_log': player_log,
        'HIST_C2': HIST_C2, 'LEN_C2': LEN_C2,
        'HIST_C3_RAW': HIST_C3_RAW, 'LEN_C3': LEN_C3, 'IS_PH_C3': IS_PH_C3,
        'player_pos': player_pos, 'opp_pos': opp_pos,
        'HEAD_COLS_C3': HEAD_COLS_C3, 'HEAD_COLS_C4': HEAD_COLS_C4,
        'carve_train': carve_train, 'build_hist_c3_fold': build_hist_c3_fold,
        'scores': scores,
    }
