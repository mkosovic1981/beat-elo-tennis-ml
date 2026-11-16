"""
Exp D (TabPFN v2) - D3: data loading + assertions + feature sets + symmetrisation helpers.

Reads ONLY the anonymised workspace (<workdir>/data; workdir from argument or env GDSI_ML_WORKDIR): features.parquet,
feature_tags.json, matches_export.csv. Never touches the database or the pseudonym map.

Provides:
  load_tabpfn_dataset(workdir)  -> dict(sym, TAGS, FOLDS, MAIN_FOLD, COLS={'A_PRIME','B_PRIME','B_PRIME_PAR'}, ...)
  add_d14_features(ds, rel_kind, rel_window)  -> optional D14 columns (log rest, rest>180d flag, relative ELO)
  symmetrise(sym_test, p)       -> p = (p(A,B) + 1 - p(B,A)) / 2  (per SusretId, both rows get the same value
                                   in their own perspective)
  split_context_calib(...)      -> leakage-safe context / calibration split inside a train fold

Feature sets (same as notebooks 07 / 09, plan 2.3):
  A'      = experiment_a_cols (27, no ELO) + 9 trajectory diffs            (36)
  B'      = experiment_b_cols (28, with elo_diff) + 9 trajectory diffs     (37)
  B'-par  = B' without MATCH-CONTEXT columns (league / season-size / razina-vs-league features, gender flag).
            These do not exist for an arbitrary player pair A vs B in the nightly 250k-pair precompute, so
            this is the only variant that can go to production (plan 2.5). h2h_total_before is a property
            of the pair, so it stays.  ASSUMPTION - confirm with the architect if the context list differs.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_WORKDIR = os.environ.get('GDSI_ML_WORKDIR')  # None -> workdir must be passed explicitly
SPLIT_DATE = pd.Timestamp('2025-06-06')
MIN_ANON_ID = 100001
N_BLOCKS = 5

# match-context columns removed for the production-compatible "par" variant
CONTEXT_COLS = [
    'liga_koeficjent', 'liga_tip', 'broj_liga_u_sezoni', 'broj_igraca_ukupno', 'broj_igraca_uligi',
    'relative_razina_by_liga_count', 'relative_razina_by_player_count', 'relative_razina_pct',
]

T0 = time.time()


def lap(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)


def _trajectory_diffs(sym, raw):
    """Razina-trajectory diff features. EXACT copy of wf_c3c4_common.load_common Step 4 / notebook 07."""
    common_cols = ['SusretId', 'Datum', 'SezonaId', 'LigaId', 'LigaTip', 'LigaKoeficjent', 'Lokacija']
    w = raw[common_cols + ['WinnerId', 'LoserId']].rename(
        columns={'WinnerId': 'IgracId', 'LoserId': 'OpponentId'}).assign(Won=1)
    l = raw[common_cols + ['LoserId', 'WinnerId']].rename(
        columns={'LoserId': 'IgracId', 'WinnerId': 'OpponentId'}).assign(Won=0)
    pl = pd.concat([w, l], ignore_index=True)
    pl = pl.sort_values(['IgracId', 'Datum', 'SusretId'], kind='mergesort').reset_index(drop=True)
    g = pl.groupby('IgracId', sort=False)
    pl['RazinaAgo5'] = g['LigaKoeficjent'].shift(5)
    pl['RazinaAgo10'] = g['LigaKoeficjent'].shift(10)
    pl['RazinaAgo20'] = g['LigaKoeficjent'].shift(20)

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

    pl['_pos'] = g.cumcount()
    W = 10
    pl['RazinaSlope10'] = rolling_slope(pl, '_pos', 'LigaKoeficjent', W, W).groupby(pl['IgracId']).shift(1)
    pl['FormaSlope10'] = rolling_slope(pl, '_pos', 'Won', W, W).groupby(pl['IgracId']).shift(1)
    prev = g['LigaKoeficjent'].shift(1)
    pl['_changed'] = (pl['LigaKoeficjent'] != prev).astype(int)
    pl['_block_id'] = pl.groupby('IgracId')['_changed'].cumsum()
    pl['RazinaStint'] = pl.groupby(['IgracId', '_block_id']).cumcount()
    pl['RazinaBestEver'] = pl.groupby('IgracId')['LigaKoeficjent'].apply(
        lambda s: s.cummin().shift(1)).reset_index(level=0, drop=True)
    pl['RazinaCurrentMinusBest'] = pl['LigaKoeficjent'] - pl['RazinaBestEver']
    rmin = pl.groupby('IgracId')['LigaKoeficjent'].apply(
        lambda s: s.rolling(5, min_periods=1).min().shift(1)).reset_index(level=0, drop=True)
    rmax = pl.groupby('IgracId')['LigaKoeficjent'].apply(
        lambda s: s.rolling(5, min_periods=1).max().shift(1)).reset_index(level=0, drop=True)
    promo = pd.Series(0.0, index=pl.index)
    promo = promo.where(~(pl['LigaKoeficjent'] < rmin), -1)
    promo = promo.where(~(pl['LigaKoeficjent'] > rmax), 1)
    promo = promo.where(rmin.notna(), np.nan)
    pl['RazinaPromotionSignal'] = promo
    raw_cols = ['RazinaAgo5', 'RazinaAgo10', 'RazinaAgo20', 'RazinaSlope10', 'RazinaStint',
                'RazinaBestEver', 'RazinaCurrentMinusBest', 'RazinaPromotionSignal', 'FormaSlope10']
    traj = pl[['SusretId', 'IgracId'] + raw_cols].copy()
    own = sym[['SusretId', 'PlayerId']].merge(
        traj, left_on=['SusretId', 'PlayerId'], right_on=['SusretId', 'IgracId'], how='left',
        validate='one_to_one').drop(columns=['IgracId'])
    opp = sym[['SusretId', 'OpponentId']].merge(
        traj, left_on=['SusretId', 'OpponentId'], right_on=['SusretId', 'IgracId'], how='left',
        validate='one_to_one').drop(columns=['IgracId'])
    assert (own['SusretId'].values == sym['SusretId'].values).all()
    assert (opp['SusretId'].values == sym['SusretId'].values).all()
    diff_cols = []
    for c in raw_cols:
        d = f'{c}_diff'
        sym[d] = own[c].values - opp[c].values
        diff_cols.append(d)
    return diff_cols


def load_tabpfn_dataset(workdir=DEFAULT_WORKDIR, verify_against_wf=False):
    if not workdir:
        raise ValueError('workdir is required (pass it or set env GDSI_ML_WORKDIR)')
    workdir = Path(workdir)
    data_dir = workdir / 'data'
    # ---- hard guard: only the anonymised workspace, never the original PII folder ----
    resolved = str(data_dir.resolve()).replace('\\', '/').lower()
    assert not resolved.endswith('/ml/data'), 'REFUSING to read the original PII folder ml/data'

    sym = pd.read_parquet(data_dir / 'features.parquet')
    with open(data_dir / 'feature_tags.json', encoding='utf-8') as f:
        TAGS = json.load(f)
    sym['liga_tip'] = sym['liga_tip'].map({'SM': 0, 'SF': 1}).astype(int)
    sym = sym.sort_values(['Datum', 'SusretId', 'Perspective'], kind='mergesort').reset_index(drop=True)
    lap(f'features.parquet: {sym.shape}')

    # ---- assertions: rows, ids, split ----
    n_matches = sym['SusretId'].nunique()
    assert len(sym) == 2 * n_matches, 'every match must have exactly 2 rows (flip)'
    assert sym.groupby('SusretId').size().eq(2).all()
    assert sym['PlayerId'].min() >= MIN_ANON_ID and sym['OpponentId'].min() >= MIN_ANON_ID, \
        f'found original (non-pseudonymised) player ids < {MIN_ANON_ID}'
    assert n_matches > 17128, f'expected the new export (>17,128 matches), got {n_matches}'
    assert str(TAGS['split_date']) == str(SPLIT_DATE.date()), TAGS['split_date']
    tr, te = sym[sym['Split'] == 'train'], sym[sym['Split'] == 'test']
    assert tr['SusretId'].nunique() == TAGS['n_matches_train']
    assert te['SusretId'].nunique() == TAGS['n_matches_test']
    assert tr['Datum'].max() < SPLIT_DATE <= te['Datum'].min() or tr['Datum'].max() <= te['Datum'].min(), 'split not chronological'
    lap(f'OK matches={n_matches:,} train={TAGS["n_matches_train"]:,} test={TAGS["n_matches_test"]:,} '
        f'split={TAGS["split_date"]}; min anon id={min(sym["PlayerId"].min(), sym["OpponentId"].min())}')

    # ---- walk-forward folds: EXACT copy of notebook 09 Step 2 / wf_c3c4_common ----
    match_order = (sym[['SusretId', 'Datum']].drop_duplicates()
                   .sort_values(['Datum', 'SusretId'], kind='mergesort').reset_index(drop=True))
    edges = np.linspace(0, len(match_order), N_BLOCKS + 1).astype(int)
    blocks = [match_order.iloc[edges[i]:edges[i + 1]] for i in range(N_BLOCKS)]
    FOLDS = []
    for k in range(1, N_BLOCKS):
        FOLDS.append({'fold': k, 'train_ids': set(pd.concat(blocks[:k])['SusretId']),
                      'test_ids': set(blocks[k]['SusretId']), 'cutoff': blocks[k]['Datum'].min()})
    assert len(FOLDS[-1]['train_ids']) == TAGS['n_matches_train']
    assert len(FOLDS[-1]['test_ids']) == TAGS['n_matches_test']
    lap('OK fold 4 == main split (' + ', '.join(f"f{f['fold']}: {len(f['train_ids']):,}/{len(f['test_ids']):,}" for f in FOLDS) + ')')

    # ---- trajectory features + feature sets ----
    raw = pd.read_csv(data_dir / 'matches_export.csv', parse_dates=['Datum'])
    raw['Datum'] = pd.to_datetime(raw['Datum']).dt.tz_localize(None)
    raw = raw.sort_values(['Datum', 'SusretId']).reset_index(drop=True)
    assert min(raw['WinnerId'].min(), raw['LoserId'].min()) >= MIN_ANON_ID
    traj_cols = _trajectory_diffs(sym, raw)
    a_cols, b_cols = TAGS['experiment_a_cols'], TAGS['experiment_b_cols']
    COLS = {
        'A_PRIME': a_cols + traj_cols,
        'B_PRIME': b_cols + traj_cols,
    }
    COLS['B_PRIME_PAR'] = [c for c in COLS['B_PRIME'] if c not in CONTEXT_COLS]
    assert len(COLS['A_PRIME']) == 36 and len(COLS['B_PRIME']) == 37, (len(COLS['A_PRIME']), len(COLS['B_PRIME']))
    assert 'elo_diff' in COLS['B_PRIME'] and 'elo_diff' not in COLS['A_PRIME']
    assert not any(c in COLS['B_PRIME_PAR'] for c in CONTEXT_COLS)
    lap(f"feature sets: A'={len(COLS['A_PRIME'])} B'={len(COLS['B_PRIME'])} B'-par={len(COLS['B_PRIME_PAR'])}")

    if verify_against_wf:  # parity with the baseline pipeline (slow: builds GRU histories)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import wf_c3c4_common as wf
        wf.DATA_DIR = data_dir
        c = wf.load_common()
        assert c['EXP_B_PRIME_COLS'] == COLS['B_PRIME']
        a = c['sym'][COLS['B_PRIME']].to_numpy(float)
        b = sym[COLS['B_PRIME']].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True), 'feature mismatch vs wf_c3c4_common'
        lap('OK B-prime parity with wf_c3c4_common')

    return {'sym': sym, 'raw': raw, 'TAGS': TAGS, 'FOLDS': FOLDS, 'MAIN_FOLD': FOLDS[-1],
            'match_order': match_order, 'COLS': COLS, 'TRAJ_COLS': traj_cols, 'data_dir': data_dir}


def add_d14_features(ds, rel_kind='rel', rel_window=365):
    """D14 add-ons (plan 2.8): log1p rest without the 90d cut, rest>180d flag, relative ELO per gender
    (point-in-time, via run_c5.add_relative_elo). Returns the list of new column names.
    rel_kind: 'rel' (ELO - mean of active, same gender) | 'pct' (percentile among active, same gender)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_c5  # noqa: E402  (needs torch; fine in the tabpfn venv)
    sym = ds['sym']
    new = []
    for who in ('player', 'opponent'):
        d = sym[f'{who}_days_since_last_match']
        sym[f'rest_log_{who}'] = np.log1p(d)
        sym[f'long_{who}'] = (d > 180).astype(float).where(d.notna())
        new += [f'rest_log_{who}', f'long_{who}']
    sym = run_c5.add_relative_elo(sym, run_c5.build_events(ds['raw']), windows=(rel_window,))
    suffix = f'{rel_window}'
    src = {'rel': (f'rel{suffix}_p_raw', f'rel{suffix}_o_raw'), 'pct': (f'pct{suffix}_p', f'pct{suffix}_o')}[rel_kind]
    sym[f'{rel_kind}{suffix}_player'], sym[f'{rel_kind}{suffix}_opponent'] = sym[src[0]], sym[src[1]]
    new += [f'{rel_kind}{suffix}_player', f'{rel_kind}{suffix}_opponent']
    ds['sym'] = sym
    return new


def split_context_calib(sym, train_ids, match_order, calib_frac=0.15):
    """Leakage-safe split inside a train fold (plan 2.3).
    calib = chronologically LAST calib_frac of the fold's train matches (same rule as carve_train), kept
    OUT of the context. context pool = the rest. Both flip rows of a match always stay on the same side."""
    fold_order = match_order[match_order['SusretId'].isin(train_ids)].reset_index(drop=True)
    n_calib = max(1, int(round(len(fold_order) * calib_frac)))
    calib_ids = set(fold_order.iloc[len(fold_order) - n_calib:]['SusretId'])
    pool_ids = list(fold_order.iloc[:len(fold_order) - n_calib]['SusretId'])
    assert not (calib_ids & set(pool_ids))
    return pool_ids, calib_ids


def pick_context_ids(pool_ids, mode, n_rows, seed):
    """mode 'last' = chronologically last n_rows/2 matches of the pool; 'random' = random n_rows/2 matches.
    Returns match ids (each contributes 2 rows -> n_rows rows)."""
    n_m = n_rows // 2
    if len(pool_ids) <= n_m:
        return list(pool_ids)
    if mode == 'last':
        return list(pool_ids[-n_m:])
    rng = np.random.default_rng(seed)
    return list(rng.choice(np.array(pool_ids), size=n_m, replace=False))


def symmetrise(frame, p):
    """p_sym = (p(A,B) + 1 - p(B,A)) / 2, per match; each row gets the value for ITS perspective.
    `frame` has SusretId + Perspective rows (2 per match); `p` aligned with frame rows."""
    s = pd.DataFrame({'SusretId': frame['SusretId'].values, 'Perspective': frame['Perspective'].values, 'p': p})
    own_is_first = ~s.duplicated('SusretId', keep='first')
    out = np.empty(len(s))
    idx_first = s.index[own_is_first]
    idx_second = s.index[~own_is_first]
    m1 = s.loc[idx_first].set_index('SusretId')['p']
    m2 = s.loc[idx_second].set_index('SusretId')['p']
    sid = s['SusretId']
    p1 = sid.map(m1).values
    p2 = sid.map(m2).values
    avg_first = (p1 + (1 - p2)) / 2.0           # perspective of the first row
    out[idx_first] = avg_first[idx_first]
    out[idx_second] = 1 - avg_first[idx_second]
    return out


if __name__ == '__main__':
    # D3 check without any model:  python tabpfn_data.py [workdir] [--verify]
    wd = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else DEFAULT_WORKDIR
    ds = load_tabpfn_dataset(wd, verify_against_wf='--verify' in sys.argv)
    main = ds['MAIN_FOLD']
    pool, calib = split_context_calib(ds['sym'], main['train_ids'], ds['match_order'])
    for mode in ('last', 'random'):
        ctx = pick_context_ids(pool, mode, 10_000, 42)
        assert len(ctx) == 5000 and not (set(ctx) & calib) and not (set(ctx) & main['test_ids'])
    # symmetrise sanity: perfectly antisymmetric predictions are unchanged
    t = ds['sym'][ds['sym']['SusretId'].isin(main['test_ids'])]
    rng = np.random.default_rng(0)
    base = pd.Series(rng.uniform(0.1, 0.9, t['SusretId'].nunique()), index=t['SusretId'].unique())
    p = np.where(t['Perspective'].values == t.groupby('SusretId')['Perspective'].transform('first').values,
                 t['SusretId'].map(base).values, 1 - t['SusretId'].map(base).values)
    assert np.allclose(symmetrise(t, p), p)
    print('D3 OK (loader, assertions, context/calib disjointness, symmetrise sanity).')
