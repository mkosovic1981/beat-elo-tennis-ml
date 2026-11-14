"""precompute_gru_matrix.py — nightly batch job: builds the active-singles
pairwise GRU win-probability matrix and UPSERTs it into Supabase `gru_win_prob`.

Run (local dev, direct SQL Server read):
    python -m serving.precompute_gru_matrix --source sql --dry-run --limit 10
    python -m serving.precompute_gru_matrix --source sql

Run (production shape, PHP raw-data endpoint):
    python -m serving.precompute_gru_matrix --source php

Pipeline (runs on a Vercel Cron-triggered Vercel Python function in the
private club app; this module is imported unchanged by both the local CLI
entrypoint below and that cron function):

  1. Fetch the FULL comparable singles roster via the club app's `igraci.php`
     endpoint (IgracPar=0, OfficialMatches >= 1 — no AktivniSingl filter, no
     reliability floor, since the app's player-comparison UI lets an admin
     pick any two comparable players) — reuses the existing, already-vetted
     `igraci.php` projection instead of re-deriving it.
  2. Fetch the FULL raw singles match log (SM/SF, non-WO) — either directly
     from SQL Server (`--source sql`, local dev only) or via the club app's
     `ml-features-raw.php` endpoint (`--source php`, the production shape;
     not part of this repo).
  3. Build the point-in-time player log + step-feature array via the SHARED
     `feature_pipeline.py` module (byte-identical to the training notebook).
  4. For every roster player: build their "as of now" 24-step sequence + head
     scalars (elo/cwr/forma/mc/age).
  5. For every eligible SAME-GENDER pair (both roster, both already
     >=1-match-gated by the roster fetch) — cross-gender pairs are excluded
     entirely (the GRU never saw a single cross-gender example in training)
     — build the H2H block and run the ONNX model + isotonic calibration.
  6. UPSERT one canonical row per unordered pair (igrac_a_id=min, igrac_b_id=
     max, prob_a_wins=P(a beats b)) into Supabase `gru_win_prob`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .data_source import (
    PhpRawMatchSource,
    SqlServerRawMatchSource,
    fetch_roster,
)
from .feature_pipeline import (
    assemble_head_vector,
    build_current_sequence,
    build_h2h_block,
    build_head_scalars_for_player,
    build_indices,
    build_player_log,
    build_step_features,
)

SERVING_DIR = Path(__file__).resolve().parent
MODELS_DIR = SERVING_DIR / 'models'
ONNX_PATH = MODELS_DIR / 'exp_c3.onnx'
ISOTONIC_PATH = MODELS_DIR / 'exp_c3_isotonic.pkl'
SCALING_JSON_PATH = MODELS_DIR / 'exp_c3_scaling.json'

# Reliability floor lowered from 8 -> 1 per Marin's directive (2026-07-12):
# comparison UI lets an admin pick ANY two comparable singles players, so the
# only remaining eligibility gates are IgracPar=0, >=1 in-scope match (a
# player needs at least one match to build a non-empty GRU sequence -- see
# `build_matrix`'s `skipped_no_history` guard below), and same-gender pairing.
MIN_OFFICIAL_MATCHES = 1
MODEL_VERSION = 'exp_c3_v1'


def load_scaling_bundle() -> dict:
    return json.loads(SCALING_JSON_PATH.read_text(encoding='utf-8'))


def load_isotonic():
    import pickle
    with open(ISOTONIC_PATH, 'rb') as f:
        return pickle.load(f)


def fetch_current_sezona_id(api_url: str, api_key: str) -> int | None:
    """Best-effort — sezone.php returns past+current seasons, newest first.
    Used only for the `sezona_id` staleness/debug column; never blocks the
    run on failure."""
    try:
        import requests
        resp = requests.get(
            f'{api_url.rstrip("/")}/sezone.php',
            headers={'x-api-key': api_key}, timeout=15,
        )
        resp.raise_for_status()
        seasons = resp.json()
        if seasons:
            return int(seasons[0]['SezonaId'])
    except Exception as exc:  # noqa: BLE001 -- best-effort, never fatal
        print(f'[precompute] WARNING: could not fetch current sezona_id: {exc}')
    return None


def build_matrix(roster: list[dict], player_log, first_idx, count_idx, step_feats,
                  scaling: dict, session: ort.InferenceSession, iso, ref_year: int,
                  verbose: bool = True) -> list[dict]:
    """Core computation — returns a list of row-dicts ready for UPSERT.
    `roster` entries: IgracId, Spol, Godiste, CurrentElo, OfficialMatches.
    """
    from .feature_pipeline import MAX_LEN, N_FEATS

    # ── Per-player "as of now" state, built ONCE per roster player ───────────
    player_state = {}
    skipped_no_history = []
    for p in roster:
        pid = p['IgracId']
        if pid not in first_idx:
            skipped_no_history.append(pid)
            continue
        seq, length = build_current_sequence(step_feats, first_idx, count_idx, pid)
        upto = first_idx[pid] + count_idx[pid]
        head = build_head_scalars_for_player(
            player_log, first_idx, pid, upto,
            elo_value=p['CurrentElo'], godiste=p['Godiste'],
            scaling=scaling['scaling'], ref_year=ref_year,
        )
        player_state[pid] = {'seq': seq, 'length': length, 'head': head,
                              'spol': p.get('Spol'), 'upto': upto}

    if skipped_no_history:
        print(f'[precompute] {len(skipped_no_history)} roster player(s) with no SM/SF '
              f'history in the raw match log -- skipped: {skipped_no_history[:20]}'
              f'{"..." if len(skipped_no_history) > 20 else ""}')

    eligible_ids = sorted(player_state.keys())
    n = len(eligible_ids)
    if verbose:
        print(f'[precompute] {n} roster players with usable history '
              f'(of {len(roster)} fetched from igraci.php).')

    # ── Build eligible SAME-GENDER pairs (a=min id, b=max id) ────────────────
    pairs = []
    excluded_cross_gender = 0
    for i in range(n):
        for j in range(i + 1, n):
            id_a, id_b = eligible_ids[i], eligible_ids[j]
            spol_a, spol_b = player_state[id_a]['spol'], player_state[id_b]['spol']
            if spol_a and spol_b and spol_a != spol_b:
                excluded_cross_gender += 1
                continue
            pairs.append((id_a, id_b))

    if verbose:
        print(f'[precompute] {len(pairs)} eligible same-gender pairs '
              f'({excluded_cross_gender} cross-gender pairs excluded).')

    if not pairs:
        return []

    # ── Assemble batched ONNX inputs ──────────────────────────────────────────
    batch = len(pairs)
    player_seq = np.zeros((batch, MAX_LEN, N_FEATS), dtype=np.float32)
    player_len = np.zeros(batch, dtype=np.int64)
    opp_seq = np.zeros((batch, MAX_LEN, N_FEATS), dtype=np.float32)
    opp_len = np.zeros(batch, dtype=np.int64)
    head_feats = np.zeros((batch, 15), dtype=np.float32)

    for k, (id_a, id_b) in enumerate(pairs):
        st_a, st_b = player_state[id_a], player_state[id_b]
        player_seq[k] = st_a['seq']
        player_len[k] = st_a['length']
        opp_seq[k] = st_b['seq']
        opp_len[k] = st_b['length']
        h2h = build_h2h_block(player_log, first_idx, id_a, id_b, st_a['upto'], scaling['scaling'])
        head_feats[k] = assemble_head_vector(
            st_a['head'], st_b['head'], h2h,
            text_skill_neutral=scaling['scaling'].get('text_skill_neutral', 0.0),
        )

    t0 = time.time()
    logits = session.run(
        ['logit'],
        {
            'player_seq': player_seq, 'player_len': player_len,
            'opp_seq': opp_seq, 'opp_len': opp_len, 'head_feats': head_feats,
        },
    )[0]
    p_raw = 1.0 / (1.0 + np.exp(-logits))
    p_calibrated = iso.predict(p_raw)
    p_calibrated = np.clip(p_calibrated, 1e-4, 1 - 1e-4)
    if verbose:
        print(f'[precompute] ONNX inference for {batch} pairs done in {time.time() - t0:.2f}s.')

    rows = []
    for k, (id_a, id_b) in enumerate(pairs):
        rows.append({
            'igrac_a_id': id_a,
            'igrac_b_id': id_b,
            'prob_a_wins': float(p_calibrated[k]),
        })
    return rows


# psycopg2.extras.execute_values' own `page_size` defaults to 100. With ~250k pairs that is
# ~2,500 sequential network round trips to the database from a serverless function in a
# different region -- at ~100 ms each that alone eats almost the whole 300 s duration cap.
UPSERT_PAGE_SIZE = 5000
# Commit in outer chunks so a duration-cap kill mid-run leaves whatever was already committed
# intact instead of losing the whole night's work to one giant implicit rollback (defence in
# depth on top of the page_size fix above, not a replacement for it).
UPSERT_COMMIT_CHUNK = 25000


def upsert_rows(database_url: str, rows: list[dict], model_version: str, sezona_id: int | None) -> int:
    import psycopg2
    import psycopg2.extras

    if not rows:
        return 0

    sql = """
        INSERT INTO gru_win_prob (igrac_a_id, igrac_b_id, prob_a_wins, model_version, sezona_id, computed_at)
        VALUES %s
        ON CONFLICT (igrac_a_id, igrac_b_id)
        DO UPDATE SET prob_a_wins = EXCLUDED.prob_a_wins,
                      model_version = EXCLUDED.model_version,
                      sezona_id = EXCLUDED.sezona_id,
                      computed_at = NOW()
    """

    conn = psycopg2.connect(database_url)
    written = 0
    try:
        for chunk_start in range(0, len(rows), UPSERT_COMMIT_CHUNK):
            chunk = rows[chunk_start:chunk_start + UPSERT_COMMIT_CHUNK]
            values = [
                (r['igrac_a_id'], r['igrac_b_id'], r['prob_a_wins'], model_version, sezona_id)
                for r in chunk
            ]
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur, sql, values,
                    template='(%s, %s, %s, %s, %s, NOW())',
                    page_size=UPSERT_PAGE_SIZE,
                )
            conn.commit()
            written += len(chunk)
    finally:
        conn.close()
    return written


def run_precompute(
    *,
    source: str = 'sql',
    api_url: str,
    api_key: str,
    database_url: str | None = None,
    model_version: str = MODEL_VERSION,
    dry_run: bool = False,
    limit: int | None = None,
    verbose: bool = True,
) -> dict:
    """Programmatic entrypoint — shared by the CLI (`main()` below) AND the
    Vercel Python cron function (`api/cron/gru-precompute.py`), so production
    and local-dev code paths run the EXACT same logic, differing only in
    `source` ('php' in production, 'sql' for local direct-DB testing) and
    which env vars/secrets happen to be present in each environment.

    Returns a small JSON-serializable summary dict — used by the Vercel
    function to build its HTTP response body.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    t_start = time.time()

    log('[precompute] Fetching active singles roster from igraci.php...')
    roster = fetch_roster(api_url, api_key, min_official_matches=MIN_OFFICIAL_MATCHES)
    if limit:
        roster = roster[:limit]
    n_male = sum(1 for p in roster if p['Spol'] == 'M')
    n_female = sum(1 for p in roster if p['Spol'] == 'F')
    log(f'[precompute] Roster: {len(roster)} players ({n_male} M / {n_female} F).')

    log(f'[precompute] Fetching raw match log (source={source})...')
    if source == 'sql':
        raw_source = SqlServerRawMatchSource()
    else:
        raw_source = PhpRawMatchSource(api_url, api_key)
    t0 = time.time()
    raw_matches = raw_source.fetch_matches()
    log(f'[precompute] Raw match log: {len(raw_matches):,} rows in {time.time() - t0:.2f}s.')

    log('[precompute] Building point-in-time player log (feature_pipeline.build_player_log)...')
    t0 = time.time()
    player_log = build_player_log(raw_matches)
    player_log, first_idx, count_idx, idx_lookup = build_indices(player_log)
    scaling = load_scaling_bundle()
    step_feats = build_step_features(player_log, scaling['scaling'])
    log(f'[precompute] player_log: {len(player_log):,} rows, built in {time.time() - t0:.2f}s.')

    session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    iso = load_isotonic()

    ref_year = time.gmtime().tm_year
    rows = build_matrix(roster, player_log, first_idx, count_idx, step_feats,
                         scaling, session, iso, ref_year=ref_year, verbose=verbose)

    summary = {
        'ok': True,
        'roster_size': len(roster),
        'roster_male': n_male,
        'roster_female': n_female,
        'raw_match_rows': int(len(raw_matches)),
        'pairs_computed': len(rows),
        'dry_run': dry_run,
        'model_version': model_version,
        'elapsed_seconds': None,  # filled in below
    }

    if not rows:
        log('[precompute] No eligible pairs computed -- nothing to write.')
        summary['elapsed_seconds'] = round(time.time() - t_start, 2)
        return summary

    probs = np.array([r['prob_a_wins'] for r in rows])
    summary.update({
        'prob_min': float(probs.min()), 'prob_max': float(probs.max()),
        'prob_mean': float(probs.mean()), 'prob_median': float(np.median(probs)),
    })
    log(f'[precompute] {len(rows)} pairs computed. '
        f'prob_a_wins: min={probs.min():.3f} max={probs.max():.3f} '
        f'mean={probs.mean():.3f} median={np.median(probs):.3f}')

    if dry_run:
        log('[precompute] dry_run set -- NOT writing to Supabase. Sample rows:')
        for r in rows[:10]:
            log(f"  {r['igrac_a_id']:>5} vs {r['igrac_b_id']:>5}  prob_a_wins={r['prob_a_wins']:.4f}")
        summary['elapsed_seconds'] = round(time.time() - t_start, 2)
        return summary

    if not database_url:
        raise ValueError('database_url is required to write to Supabase (dry_run=False).')

    sezona_id = fetch_current_sezona_id(api_url, api_key)
    n_written = upsert_rows(database_url, rows, model_version, sezona_id)
    summary['rows_written'] = n_written
    summary['sezona_id'] = sezona_id
    log(f'[precompute] UPSERTed {n_written} rows into gru_win_prob '
        f'(model_version={model_version}, sezona_id={sezona_id}).')
    summary['elapsed_seconds'] = round(time.time() - t_start, 2)
    log(f'[precompute] Total elapsed: {summary["elapsed_seconds"]}s.')
    return summary


def main():
    parser = argparse.ArgumentParser(description='GRU win-probability nightly precompute')
    parser.add_argument('--source', choices=['sql', 'php'], default='sql',
                         help='Raw match data source: sql (local dev, direct SQL Server) '
                              'or php (production shape, ml-features-raw.php over HTTP)')
    parser.add_argument('--dry-run', action='store_true',
                         help='Compute the matrix but do not write to Supabase')
    parser.add_argument('--limit', type=int, default=None,
                         help='Limit roster to the first N players (fast local testing)')
    parser.add_argument('--api-url', default=os.environ.get('API_URL'))
    parser.add_argument('--api-key', default=os.environ.get('API_KEY'))
    parser.add_argument('--database-url', default=os.environ.get('DATABASE_URL'))
    parser.add_argument('--model-version', default=MODEL_VERSION)
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        raise SystemExit('--api-url/--api-key (or API_URL/API_KEY env vars) are required.')

    run_precompute(
        source=args.source, api_url=args.api_url, api_key=args.api_key,
        database_url=args.database_url, model_version=args.model_version,
        dry_run=args.dry_run, limit=args.limit,
    )


if __name__ == '__main__':
    main()
