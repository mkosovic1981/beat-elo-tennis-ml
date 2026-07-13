"""feature_pipeline.py — shared, notebook-free GRU feature-construction module.

Extracted verbatim (same computation, same numbers) from
`notebooks/10_enriched_gru.ipynb` Steps 2-9 — the #1 anti-training/serving-skew
measure for this project (reuse the exact training-time feature logic at
serving time, don't reimplement it).

Imported by BOTH:
  - the training notebook (`10_enriched_gru.ipynb`, updated to call these
    functions instead of defining the logic inline — plumbing-only change)
  - the serving batch job (`serving/precompute_gru_matrix.py`)

Two usage modes, unified by a single `upto_gpos_exclusive` position argument:
  1. "As of now" (serving): every active player's own full real history is
     used — `upto_gpos_exclusive = start_of_player + count_of_player`, i.e. a
     virtual position one past the player's last real row. Simpler than
     training's point-in-time construction (no need to hide any future row).
  2. "Point in time" (training / regression test): `upto_gpos_exclusive =
     idx_lookup[(igrac_id, susret_id)]` for a specific historical row — used
     ONLY by the notebook and by the regression test that diffs rebuilt
     feature vectors against notebook-computed ones.

CRITICAL discipline carried over unchanged from the notebook: every function
here only ever reads rows strictly BEFORE `upto_gpos_exclusive` within a
player's own chronologically-sorted (IgracId, Datum, SusretId) group — the
same point-in-time leakage discipline used throughout every notebook in this
project (see the README's "Point-in-time feature engineering" section).

Train-only scaling constants (ELO_MEAN/STD, AGE_MEAN/STD, LIGAKOEF_SCALE,
SETMARGIN_SCALE, DAYSREST_CLIP, etc.) are NEVER recomputed here — they are
always passed in via the `scaling` dict. At export time they are read from
the trained model's `.pt` checkpoint and bundled into `exp_c3_scaling.json`
(`serving/model.py::dump_scaling_json`); at serving time they are loaded back
verbatim from that JSON file (`precompute_gru_matrix.py::load_scaling_bundle`).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# ─── Architecture-level constants (fixed, NOT data-derived — mirror the
#     notebook's MAX_LEN / N_FEATS / N_HEAD / HEAD_COLS exactly) ─────────────
MAX_LEN = 24
N_FEATS = 9
STEP_FEAT_NAMES = [
    'Won', 'SetMarginScaled', 'DaysRestScaled', 'LigaKoefScaled', 'HomeShareBefore',
    'OpponentCareerWinRateBefore', 'OppEloBeforeScaled', 'WasH2H', 'IsPlaceholder',
]
HEAD_COLS = [
    'elo_p_scaled', 'elo_o_scaled', 'cwr_p', 'cwr_o', 'forma_p', 'forma_o',
    'mc_p', 'mc_o', 'age_p', 'age_o', 'text_p', 'text_o',
    'h2h_wr', 'h2h_total_scaled', 'h2h_sets_margin_scaled',
]
N_HEAD = len(HEAD_COLS)

# Godiste (birth-year int) QC bounds — pinned in notebook 01_data_exploration.ipynb,
# applied in notebook 02_feature_engineering.ipynb Step 4. Fixed domain constants,
# NOT part of the model checkpoint's `scaling` dict (they gate which raw Godiste
# values are trusted at all, upstream of any scaling).
GODISTE_LOWER = 1930
GODISTE_UPPER = 2015

# H2H shrinkage prior weight toward 0.5 — notebook 02 Step 6, unchanged.
H2H_PRIOR = 2.0

# Rolling window size for FormaLast10 (per-player recent-form head scalar) —
# notebook 02 Step 3 SQL: "AVG(Won) OVER ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING".
FORMA_WINDOW = 10

REQUIRED_RAW_COLS = [
    'SusretId', 'Datum', 'SezonaId', 'LigaId', 'LigaTip', 'LigaKoeficjent', 'Lokacija',
    'WinnerId', 'LoserId', 'WinnerEloBefore', 'LoserEloBefore',
    'W_Set1', 'W_Set2', 'W_Set3', 'L_Set1', 'L_Set2', 'L_Set3',
]


def placeholder_vector() -> np.ndarray:
    """The fixed per-step vector used for a player's very first (debut) match,
    where no real history exists yet. Byte-identical to notebook 10 Cell 13."""
    v = np.zeros(N_FEATS, dtype=np.float32)
    v[5] = 0.5   # neutral opponent career win rate
    v[6] = 0.0   # neutral (mean) opponent ELO -- 0 after centering
    v[7] = 0.0   # no prior H2H possible at a debut match
    v[8] = 1.0   # IsPlaceholder flag
    return v


def build_player_log(raw_matches: pd.DataFrame) -> pd.DataFrame:
    """Long-format (one row per player per match) point-in-time feature log.

    `raw_matches` must contain `REQUIRED_RAW_COLS` (the exact grain returned by
    both `scripts/export_data.mjs`'s SinglesMatches CTE and the private club
    app's `ml-features-raw.php` endpoint (not part of this repo — see
    `data_source.py`'s `PhpRawMatchSource`) — WinnerEloAfter/LoserEloAfter are
    optional extras, kept if present but not required).

    Replicates notebook 10 Cells 5, 7, 9, 11 verbatim:
      - duplicate each match into a winner-perspective and loser-perspective row
      - sort strictly ascending (IgracId, Datum, SusretId) — the single global
        sort every point-in-time column in this module depends on
      - MatchCountBefore, HomeShareBefore, DaysSinceLastMatch, CareerWinRateBefore,
        FormaLast10 (rolling-10), SetMargin
      - OpponentCareerWinRateBefore (self-merge on (SusretId, OpponentId))
      - WasH2H, H2HMeetingsBefore (per-pair cumcount)
    """
    missing = [c for c in REQUIRED_RAW_COLS if c not in raw_matches.columns]
    if missing:
        raise ValueError(f'build_player_log: raw_matches missing required columns: {missing}')

    raw = raw_matches.copy()
    raw['Datum'] = pd.to_datetime(raw['Datum'])
    if raw['Datum'].dt.tz is not None:
        raw['Datum'] = raw['Datum'].dt.tz_localize(None)

    common_cols = ['SusretId', 'Datum', 'SezonaId', 'LigaId', 'LigaTip', 'LigaKoeficjent', 'Lokacija']
    has_elo_after = 'WinnerEloAfter' in raw.columns and 'LoserEloAfter' in raw.columns

    winner_cols = common_cols + [
        'WinnerId', 'LoserId', 'WinnerEloBefore', 'LoserEloBefore',
        'W_Set1', 'W_Set2', 'W_Set3', 'L_Set1', 'L_Set2', 'L_Set3',
    ]
    winner_rows = raw[winner_cols].rename(columns={
        'WinnerId': 'IgracId', 'LoserId': 'OpponentId',
        'WinnerEloBefore': 'MyEloBefore', 'LoserEloBefore': 'OppEloBefore',
        'W_Set1': 'MySet1', 'W_Set2': 'MySet2', 'W_Set3': 'MySet3',
        'L_Set1': 'OppSet1', 'L_Set2': 'OppSet2', 'L_Set3': 'OppSet3',
    }).assign(Won=1)

    loser_cols = common_cols + [
        'LoserId', 'WinnerId', 'LoserEloBefore', 'WinnerEloBefore',
        'L_Set1', 'L_Set2', 'L_Set3', 'W_Set1', 'W_Set2', 'W_Set3',
    ]
    loser_rows = raw[loser_cols].rename(columns={
        'LoserId': 'IgracId', 'WinnerId': 'OpponentId',
        'LoserEloBefore': 'MyEloBefore', 'WinnerEloBefore': 'OppEloBefore',
        'L_Set1': 'MySet1', 'L_Set2': 'MySet2', 'L_Set3': 'MySet3',
        'W_Set1': 'OppSet1', 'W_Set2': 'OppSet2', 'W_Set3': 'OppSet3',
    }).assign(Won=0)

    if has_elo_after:
        winner_rows['MyEloAfter'] = raw['WinnerEloAfter'].values
        loser_rows['MyEloAfter'] = raw['LoserEloAfter'].values

    player_log = pd.concat([winner_rows, loser_rows], ignore_index=True)
    player_log['LokacijaNorm'] = player_log['Lokacija'].astype(str).str.strip().str.lower()
    # CRITICAL: strict ascending (IgracId, Datum, SusretId) order — every
    # point-in-time feature below only ever looks at rows ABOVE the current one
    # within a player's own group, i.e. strictly BEFORE this match.
    player_log = player_log.sort_values(
        ['IgracId', 'Datum', 'SusretId'], kind='mergesort',
    ).reset_index(drop=True)

    g = player_log.groupby('IgracId', sort=False)
    player_log['MatchCountBefore'] = g.cumcount()
    player_log['HomeCountBefore'] = player_log.groupby(
        ['IgracId', 'LokacijaNorm'], sort=False,
    ).cumcount()
    player_log['HomeShareBefore'] = np.where(
        player_log['MatchCountBefore'] > 0,
        player_log['HomeCountBefore'] / player_log['MatchCountBefore'],
        np.nan,
    )
    player_log['DaysSinceLastMatch'] = player_log.groupby('IgracId')['Datum'].diff().dt.days
    player_log['CareerWinRateBefore'] = g['Won'].transform(lambda s: s.shift().expanding().mean())
    player_log['FormaLast10'] = g['Won'].transform(
        lambda s: s.shift().rolling(FORMA_WINDOW, min_periods=1).mean()
    )

    for c in ['MySet1', 'MySet2', 'MySet3', 'OppSet1', 'OppSet2', 'OppSet3']:
        player_log[c + '_f'] = player_log[c].fillna(0)
    player_log['SetMargin'] = (
        (player_log['MySet1_f'] + player_log['MySet2_f'] + player_log['MySet3_f'])
        - (player_log['OppSet1_f'] + player_log['OppSet2_f'] + player_log['OppSet3_f'])
    )
    player_log = player_log.drop(
        columns=[c + '_f' for c in ['MySet1', 'MySet2', 'MySet3', 'OppSet1', 'OppSet2', 'OppSet3']]
    )

    # Step 4 — opponent-quality-at-that-time proxy (self-merge on (SusretId, OpponentId))
    opp_cwr = player_log[['SusretId', 'IgracId', 'CareerWinRateBefore']].rename(
        columns={'IgracId': 'OpponentId', 'CareerWinRateBefore': 'OpponentCareerWinRateBefore'}
    )
    player_log = player_log.merge(
        opp_cwr, on=['SusretId', 'OpponentId'], how='left', validate='many_to_one',
    )

    # Step 5 — per-step H2H features
    h2h_g = player_log.groupby(['IgracId', 'OpponentId'], sort=False)
    player_log['H2HMeetingsBefore'] = h2h_g.cumcount()
    player_log['WasH2H'] = (player_log['H2HMeetingsBefore'] > 0).astype(np.float32)
    # Cumulative sum of this player's SetMargin against this specific opponent,
    # shifted by one position within the (IgracId, OpponentId) subgroup — i.e.
    # the sum over STRICTLY PRIOR meetings only (notebook 10 Cell 11's
    # H2HSetsMarginBefore). Kept here for full parity with the training
    # notebook's own diagnostics/leakage-check cell; `build_h2h_block` below
    # recomputes the equivalent quantity independently via row-slicing (used
    # at serving time, where "as of now" needs the FULL to-date sum, not the
    # strictly-prior-to-a-specific-row version this column represents).
    player_log['H2HSetsMarginBefore'] = h2h_g['SetMargin'].transform(
        lambda s: s.cumsum().shift(fill_value=0.0)
    )

    return player_log


def build_indices(player_log: pd.DataFrame):
    """Position bookkeeping shared by every sequence-builder below.

    Returns:
        player_log  — same frame, with an added `_gpos` global-position column
        first_idx   — {IgracId: gpos of that player's first row}
        count_idx   — {IgracId: total number of rows for that player}
        idx_lookup  — {(IgracId, SusretId): gpos} — O(1) point-in-time lookup
    """
    player_log = player_log.copy()
    player_log['_gpos'] = np.arange(len(player_log))
    first_idx = player_log.groupby('IgracId')['_gpos'].min().to_dict()
    count_idx = player_log.groupby('IgracId')['_gpos'].count().to_dict()
    idx_lookup = dict(zip(
        zip(player_log['IgracId'], player_log['SusretId']), player_log['_gpos'],
    ))
    return player_log, first_idx, count_idx, idx_lookup


def build_step_features(player_log: pd.DataFrame, scaling: dict) -> np.ndarray:
    """The N_FEATS=9-wide per-step array for every row in `player_log`, in the
    SAME row order as `player_log` (aligned by position, not by index).

    `scaling` must supply (verbatim, train-only, from the model checkpoint):
    setmargin_scale, daysrest_clip, ligakoef_scale, elo_mean, elo_std.
    """
    n = len(player_log)
    step_feats = np.zeros((n, N_FEATS), dtype=np.float32)
    step_feats[:, 0] = player_log['Won'].values
    step_feats[:, 1] = np.clip(
        player_log['SetMargin'].values / scaling['setmargin_scale'], -3, 3
    )
    days_rest = (
        player_log['DaysSinceLastMatch']
        .fillna(scaling['daysrest_clip'])
        .clip(upper=scaling['daysrest_clip'])
        .values
    )
    step_feats[:, 2] = days_rest / scaling['daysrest_clip']
    step_feats[:, 3] = player_log['LigaKoeficjent'].values / scaling['ligakoef_scale']
    step_feats[:, 4] = player_log['HomeShareBefore'].fillna(0.0).values
    step_feats[:, 5] = player_log['OpponentCareerWinRateBefore'].fillna(0.5).values
    step_feats[:, 6] = np.clip(
        (player_log['OppEloBefore'].values - scaling['elo_mean']) / scaling['elo_std'], -4, 4
    )
    step_feats[:, 7] = player_log['WasH2H'].values
    step_feats[:, 8] = 0.0  # real step (placeholder rows are synthesized separately)
    return step_feats


def build_sequence_at(
    step_feats: np.ndarray,
    first_idx: dict,
    igrac_id: int,
    gpos: int,
    max_len: int = MAX_LEN,
) -> tuple[np.ndarray, int]:
    """Core windowing primitive — history STRICTLY BEFORE global position `gpos`,
    right-padded into a `(max_len, N_FEATS)` buffer with real steps LEFT-aligned
    (byte-identical construction to notebook 10 Cell 13's `build_history`)."""
    start = first_idx[igrac_id]
    p = gpos - start
    seq = np.zeros((max_len, N_FEATS), dtype=np.float32)
    if p == 0:
        seq[0] = placeholder_vector()
        length = 1
    else:
        lo = max(start, gpos - max_len)
        chunk = step_feats[lo:gpos]
        length = chunk.shape[0]
        seq[:length] = chunk
    return seq, length


def build_current_sequence(
    step_feats: np.ndarray,
    first_idx: dict,
    count_idx: dict,
    igrac_id: int,
    max_len: int = MAX_LEN,
) -> tuple[np.ndarray, int]:
    """'As of now' sequence for serving — uses ALL of the player's real history
    (virtual position one past their last real row). Simpler than training's
    point-in-time construction: no future to hide."""
    start = first_idx[igrac_id]
    gpos = start + count_idx[igrac_id]
    return build_sequence_at(step_feats, first_idx, igrac_id, gpos, max_len=max_len)


def build_sequence_for_match(
    step_feats: np.ndarray,
    first_idx: dict,
    idx_lookup: dict,
    igrac_id: int,
    susret_id: int,
    max_len: int = MAX_LEN,
) -> tuple[np.ndarray, int]:
    """Point-in-time sequence for an EXISTING historical (IgracId, SusretId)
    row — used only by the training notebook and the feature-pipeline
    regression test (diffing against notebook-computed tensors)."""
    gpos = idx_lookup[(igrac_id, susret_id)]
    return build_sequence_at(step_feats, first_idx, igrac_id, gpos, max_len=max_len)


def build_head_scalars_for_player(
    player_log: pd.DataFrame,
    first_idx: dict,
    igrac_id: int,
    upto_gpos_exclusive: int,
    elo_value: float,
    godiste: Optional[int],
    scaling: dict,
    ref_year: int,
) -> dict:
    """Per-player head-scalar block (elo/cwr/forma/mc/age), unified across both
    usage modes via `upto_gpos_exclusive`:
      - serving ("as of now"): pass `start_of_player + count_of_player`
      - training / regression test (point-in-time for a specific historical
        row): pass `idx_lookup[(igrac_id, susret_id)]` (== that row's own gpos)

    `elo_value` and `ref_year` are supplied by the caller because they differ
    by mode: serving uses the player's CURRENT ELO (the app's own source of
    truth, e.g. `igraci.php`'s `CurrentElo`) and today's year; the regression
    test uses that historical match's own `EloBefore` and the match's year.
    """
    start = first_idx[igrac_id]
    prior = player_log.iloc[start:upto_gpos_exclusive]
    p = len(prior)

    cwr = float(prior['Won'].mean()) if p > 0 else 0.5
    forma = float(prior['Won'].tail(FORMA_WINDOW).mean()) if p > 0 else 0.5
    mc_scaled = float(np.clip(p / scaling['matchcount_scale'], 0, 4))
    elo_scaled = (elo_value - scaling['elo_mean']) / scaling['elo_std']

    if godiste is not None and GODISTE_LOWER <= godiste <= GODISTE_UPPER:
        age = ref_year - godiste
        age_scaled = (age - scaling['age_mean']) / scaling['age_std']
    else:
        age_scaled = 0.0  # neutral (train mean, after centering) — same convention as notebook 10

    return {
        'elo_scaled': float(elo_scaled),
        'cwr': cwr,
        'forma': forma,
        'mc_scaled': mc_scaled,
        'age_scaled': float(age_scaled),
    }


def build_h2h_block(
    player_log: pd.DataFrame,
    first_idx: dict,
    id_a: int,
    id_b: int,
    upto_gpos_exclusive: int,
    scaling: dict,
) -> dict:
    """Per-matchup H2H head-scalar block (h2h_wr, h2h_total_scaled,
    h2h_sets_margin_scaled) FROM player A's PERSPECTIVE, using only A's rows
    strictly before `upto_gpos_exclusive` where OpponentId == id_b.

    Same unification as `build_head_scalars_for_player`: pass
    `start_of_a + count_of_a` for "as of now", or a specific historical row's
    own gpos for point-in-time reconstruction.
    """
    start = first_idx[id_a]
    prior = player_log.iloc[start:upto_gpos_exclusive]
    h2h_rows = prior.loc[prior['OpponentId'] == id_b]
    total = len(h2h_rows)
    wins = float(h2h_rows['Won'].sum()) if total > 0 else 0.0
    h2h_wr = (wins + H2H_PRIOR * 0.5) / (total + H2H_PRIOR)
    sets_margin_total = float(h2h_rows['SetMargin'].sum()) if total > 0 else 0.0

    return {
        'h2h_wr': float(h2h_wr),
        'h2h_total_scaled': float(np.clip(total / scaling['h2htotal_scale'], 0, 3)),
        'h2h_sets_margin_scaled': float(
            np.clip(sets_margin_total / scaling['h2h_setsmargin_scale'], -3, 3)
        ),
    }


def assemble_head_vector(
    head_a: dict, head_b: dict, h2h_block: dict, text_skill_neutral: float = 0.0,
) -> np.ndarray:
    """Concatenate per-player + per-matchup head scalars into the model's
    N_HEAD=15-wide input vector, in the EXACT `HEAD_COLS` order the checkpoint
    was trained with. `text_p`/`text_o` are filled with the neutral constant —
    the TextSkill slot is left open per notebook 10 (never wired in this pass).
    """
    values = {
        'elo_p_scaled': head_a['elo_scaled'], 'elo_o_scaled': head_b['elo_scaled'],
        'cwr_p': head_a['cwr'], 'cwr_o': head_b['cwr'],
        'forma_p': head_a['forma'], 'forma_o': head_b['forma'],
        'mc_p': head_a['mc_scaled'], 'mc_o': head_b['mc_scaled'],
        'age_p': head_a['age_scaled'], 'age_o': head_b['age_scaled'],
        'text_p': text_skill_neutral, 'text_o': text_skill_neutral,
        'h2h_wr': h2h_block['h2h_wr'],
        'h2h_total_scaled': h2h_block['h2h_total_scaled'],
        'h2h_sets_margin_scaled': h2h_block['h2h_sets_margin_scaled'],
    }
    return np.array([values[c] for c in HEAD_COLS], dtype=np.float32)
