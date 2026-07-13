"""data_source.py — swappable data access for the GRU precompute batch job.

Two concrete raw-match sources, both returning the SAME DataFrame shape
(`feature_pipeline.REQUIRED_RAW_COLS` + optional `WinnerEloAfter`/`LoserEloAfter`):

  - `SqlServerRawMatchSource` — direct SQL Server read (read-only creds from
    `scripts/db-config.json`), for LOCAL testing only. Mirrors the exact join
    shape of `scripts/export_data.mjs`'s SinglesMatches CTE and of the private
    club app's `ml-features-raw.php` endpoint (not part of this repo).
  - `PhpRawMatchSource` — HTTP GET against that same `ml-features-raw.php`
    endpoint (raw rows only, no server-side feature computation), for the
    PRODUCTION path (Vercel Python function → PHP → SQL Server, no direct DB
    access from Vercel).

Roster fetch (`fetch_roster`) always goes through the club app's existing
`igraci.php` endpoint over HTTP, regardless of which raw-match source is
chosen — this reuses the already-vetted active/eligibility logic instead of
re-deriving it, and is cheap enough (~1 request) that there is no local-SQL
fast path for it.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

RAW_MATCH_COLUMNS = [
    'SusretId', 'Datum', 'SezonaId', 'LigaId', 'LigaTip', 'LigaKoeficjent', 'Lokacija',
    'WinnerId', 'LoserId', 'WinnerEloBefore', 'WinnerEloAfter', 'LoserEloBefore', 'LoserEloAfter',
    'W_Set1', 'W_Set2', 'W_Set3', 'L_Set1', 'L_Set2', 'L_Set3',
]

# Identical join shape to the private club app's `ml-features-raw.php` endpoint
# (unfiltered, single flat query, no window functions — that endpoint is
# deliberately dumb, all feature logic lives in feature_pipeline.py instead).
# Kept here too so the LOCAL SQL path never silently drifts from the PHP
# path's grain.
SQL_QUERY = """
SELECT
    s.SusretId,
    s.Datum,
    s.SezonaId,
    s.LigaId,
    l.LigaTip,
    l.LigaKoeficjent,
    ISNULL(s.Lokacija, '') AS Lokacija,
    si_w.IgracId AS WinnerId,
    si_l.IgracId AS LoserId,
    sie_w.EloBefore AS WinnerEloBefore,
    sie_w.EloAfter  AS WinnerEloAfter,
    sie_l.EloBefore AS LoserEloBefore,
    sie_l.EloAfter  AS LoserEloAfter,
    si_w.Set1 AS W_Set1, si_w.Set2 AS W_Set2, si_w.Set3 AS W_Set3,
    si_l.Set1 AS L_Set1, si_l.Set2 AS L_Set2, si_l.Set3 AS L_Set3
FROM Susret s
JOIN Liga l ON s.LigaId = l.LigaId
JOIN SusretIgrac si_w ON si_w.SusretId = s.SusretId AND si_w.Pobjednik = 1
JOIN SusretIgrac si_l ON si_l.SusretId = s.SusretId AND si_l.Pobjednik = 0
    AND si_l.IgracId != si_w.IgracId
JOIN SusretIgracElo sie_w ON sie_w.SusretIgracId = si_w.SusretIgracId
JOIN SusretIgracElo sie_l ON sie_l.SusretIgracId = si_l.SusretIgracId
JOIN Igrac iw ON iw.IgracId = si_w.IgracId AND iw.IgracPar = 0
JOIN Igrac il ON il.IgracId = si_l.IgracId AND il.IgracPar = 0
WHERE l.LigaTip IN ('SM', 'SF')
  AND (s.NijeOdigran = 0 OR s.NijeOdigran IS NULL)
  AND s.Datum IS NOT NULL
ORDER BY s.Datum ASC, s.SusretId ASC
"""


class RawMatchSource(ABC):
    @abstractmethod
    def fetch_matches(self) -> pd.DataFrame:
        """Returns a DataFrame with (at least) `feature_pipeline.REQUIRED_RAW_COLS`."""


class SqlServerRawMatchSource(RawMatchSource):
    """LOCAL-ONLY. Direct SQL Server read via `pytds` (pure-Python driver, no
    ODBC install required), using the same read-only creds as
    `scripts/db-config.json`. Never used in production — Vercel cannot
    reach the production SQL Server host directly (no such credential is
    deployed there); production always goes through `PhpRawMatchSource`."""

    def __init__(self, db_config_path: Path | str = None):
        if db_config_path is None:
            db_config_path = Path(__file__).resolve().parents[1] / 'scripts' / 'db-config.json'
        self.cfg = json.loads(Path(db_config_path).read_text(encoding='utf-8'))

    def fetch_matches(self) -> pd.DataFrame:
        import pytds  # local, dev-only dependency — see requirements-serving.txt

        conn = pytds.connect(
            server=self.cfg['server'], port=self.cfg['port'], database=self.cfg['database'],
            user=self.cfg['user'], password=self.cfg['password'],
            timeout=self.cfg.get('connectionTimeout', 15000) / 1000,
            login_timeout=self.cfg.get('connectionTimeout', 15000) / 1000,
        )
        try:
            df = pd.read_sql(SQL_QUERY, conn)
        finally:
            conn.close()
        return df


class PhpRawMatchSource(RawMatchSource):
    """PRODUCTION path. HTTP GET against the private club app's
    `ml-features-raw.php` endpoint (raw rows only, no server-side feature
    computation — kept dumb on purpose so all feature logic lives in
    `feature_pipeline.py`, not duplicated in PHP). Same `x-api-key`
    server-to-server auth pattern the club app's other Next.js → PHP proxy
    routes use."""

    def __init__(self, api_url: str, api_key: str, timeout: int = 30):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

    def fetch_matches(self) -> pd.DataFrame:
        resp = requests.get(
            f'{self.api_url}/ml-features-raw.php',
            headers={'x-api-key': self.api_key},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = payload['Matches']
        df = pd.DataFrame(rows, columns=RAW_MATCH_COLUMNS)
        return df


def fetch_roster(api_url: str, api_key: str, min_official_matches: int = 1,
                  timeout: int = 30) -> list[dict]:
    """FULL comparable singles roster via the club app's existing `igraci.php`
    endpoint — `IgracPar=0, OfficialMatches >= min_official_matches`.
    Deliberately does NOT filter on `AktivniSingl` (that field means "still
    playing in the current season", not "eligible to be compared") — the club
    app's player-comparison UI lets an admin pick ANY two players, including
    dormant ones (e.g. a player dormant for years but with a long official
    match history), so the precomputed matrix must cover every comparable player,
    not just currently-active ones.

    Returns a list of lean dicts: IgracId, Spol, Godiste, CurrentElo,
    OfficialMatches — everything `precompute_gru_matrix.py` needs to build the
    eligible-pair matrix (gender for cross-gender exclusion, ELO/Godiste for
    head scalars, match-count for the cold-start cutoff).
    """
    resp = requests.get(
        f'{api_url.rstrip("/")}/igraci.php',
        headers={'x-api-key': api_key},
        timeout=timeout,
    )
    resp.raise_for_status()
    players = resp.json()

    roster = []
    for p in players:
        if int(p.get('IgracPar') or 0) != 0:
            continue
        official_matches = int(p.get('OfficialMatches') or 0)
        if official_matches < min_official_matches:
            continue
        current_elo = p.get('CurrentElo')
        if current_elo is None:
            continue  # no ELO on record -- cannot build head scalars
        roster.append({
            'IgracId': int(p['IgracId']),
            'Spol': p.get('Spol'),
            'Godiste': int(p['Godiste']) if p.get('Godiste') else None,
            'CurrentElo': float(current_elo),
            'OfficialMatches': official_matches,
        })
    return roster
