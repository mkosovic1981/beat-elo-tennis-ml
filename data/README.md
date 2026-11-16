# `matches_export.csv` — Schema, Regeneration, Data-Quality Notes

`matches_export.csv` is **gitignored** (contains real player IDs, match
results, and `Trenutci` free-text self-descriptions — PII). This file
documents how to regenerate it and what to expect.

## Exporter Tech

The original plan was to reuse the league app's PHP export endpoint. **This
machine has no local PHP/sqlsrv install** (checked during Phase P1 — `php` is
not on PATH), so the exporter was implemented instead as a **Node.js script
using the `mssql` package**: `scripts/export_data.mjs`. This keeps this repo
fully self-contained (own `package.json`, own `node_modules/`, no dependency
on any other codebase) and uses the same read-only DB credentials the league
app's PHP endpoints use.

The SQL itself mirrors the same CTEs, window functions, and filters the PHP
export would have used — only the driver differs, not the query.

## Regenerating

```bash
npm install                  # installs mssql from the repo-root package.json
cd scripts
node step0_pin_startdate.mjs # confirms @StartDate (already pinned below)
node export_data.mjs         # ~100-110s cold run, writes ../data/matches_export.csv
node verify_export.mjs       # re-run sanity checks
```

Requires `scripts/db-config.json` (gitignored, never committed):

```json
{
  "server": "your-sql-server-host",
  "port": 1555,
  "database": "your_database",
  "user": "your_readonly_user",
  "password": "<see project memory / ask Marin>",
  "options": { "encrypt": false, "trustServerCertificate": true },
  "connectionTimeout": 15000,
  "requestTimeout": 120000
}
```

## Pinned `@StartDate` (STEP 0)

`Sezona` query for seasons starting between 2017-04-01 and 2017-06-30 returned:

| SezonaId | SezonaNaziv | DatumOd | DatumDo |
|---|---|---|---|
| 99 | 2017/05 | 2017-05-01 | 2017-05-31 |
| 100 | 2017/06 | 2017-06-01 | 2017-06-30 |

**Pinned `@StartDate = '2017-05-01'`** (SezonaId 99, `2017/05`) — matches
Marin's "od negdje 5. mjeseca 2017."

## Row/Column Counts (last export)

- **17,128 rows**, **47 columns**
- Cold-run query time: **~104.5s** (one-time; heavy ROW_NUMBER/window CTEs —
  acceptable since this runs once, offline, not at request time)

## Column List (as exported)

All planned columns are present, using the same names as the SQL `SELECT`
aliases:

```
SusretId, Datum, LigaId, SezonaId, LigaTip, LigaKoeficjent, Lokacija,
WinnerId, WinnerEloBefore, WinnerEloAfter,
LoserId, LoserEloBefore, LoserEloAfter,
W_Set1, W_Set2, W_Set3, L_Set1, L_Set2, L_Set3,
WinnerTrenutci, WinnerDatumRodjenja, LoserTrenutci, LoserDatumRodjenja,
WinnerMatchCountBefore, LoserMatchCountBefore,
WinnerRazinaMatchCountBefore, LoserRazinaMatchCountBefore,
WinnerDaysSinceLastMatch, LoserDaysSinceLastMatch,
WinnerFormaLast10, LoserFormaLast10,
H2HWinsWinnerBefore, H2HWinsLoserBefore, H2HTotalBefore,
WinnerCareerWinRateBefore, LoserCareerWinRateBefore,
WinnerDebutRazina, LoserDebutRazina,
WinnerDebutBrojLiga, WinnerDebutBrojIgraca,
LoserDebutBrojLiga, LoserDebutBrojIgraca,
BrojLigaUSezoni, BrojIgracaUkupno, BrojIgracaULigi
```

**One addition beyond the original plan (not a renamed/missing column — an
addition):** `WinnerGodiste` and `LoserGodiste` (int, birth year). See the
finding below — `DatumRodjenja` turned out to be unusable, so `Godiste` was
added as the real age-proxy source while keeping the originally planned
`DatumRodjenja` columns present (they will just be NULL/empty).

## Data-Quality Findings (empirical checks before building the exporter)

### 1. `Igrac.Trenutci` population

- All `IgracPar=0` players: **872 / 957 filled (91.1%)**
- Players with >=1 in-scope SM/SF match since 2017-05-01: **671 / 779 filled
  (86.1%)**

Confirms Marin's expectation — coverage is high, not sparse. The
Gemini-derived skill-score feature (Exp C4, see the main README) has a solid
population to work with, though not 100%.

### 2. `Lokacija` (home-court feature)

- **17,131 / 17,131 in-scope league matches have a non-NULL `Lokacija`
  (100%)** — contrary to a possible "mostly NULL" concern, this is fully
  populated.
- Only **17 distinct raw values**, and normalizing to `lower(trim())` still
  yields 17 distinct values — i.e. **no messy free-text variants** ("Firule"
  vs "TK Firule" vs "firule") were found. The home-court feature should be
  strong with little/no normalization work needed. Top values: Solin (3578),
  Pomak (3097), Firule (3090), Peka (2281), Stobreč (2077), Baterije (1137),
  Kod Jakova (739), Resnik (453), Bačvice (Baterije) (313), Stinice (Pomak)
  (135), and 7 more with <150 matches each.

### 3. `DatumRodjenja` / age — IMPORTANT FINDING

**`Igrac.DatumRodjenja` (datetime) is 100% NULL for all 957 `IgracPar=0`
players.** This is the column originally assumed to hold birth date and
expected to be near-fully populated based on Marin's correction — instead it
is empty across the board (confirmed both in aggregate and by sampling
individual rows).

The **actual populated birth-year field is `Igrac.Godiste`** (int, e.g.
`1988`) — **858 / 957 (89.7%) filled overall**, **704 / 779 (90.4%) filled**
among in-scope players. This is consistent with Marin's expectation that age
data is well-populated — it's just stored in a different column than
originally assumed.

`Godiste` also has exactly the kind of intentionally-wrong placeholder
entries Marin flagged: 5 implausible values outside the 1920–2020 range:

| Player (pseudonym) | Godiste | Likely explanation |
|---|---|---|
| Player A | 0 | placeholder/blank |
| Player B | 42 | probably meant 1942 (2-digit year mistake) |
| Player C | 81 | probably meant 1981 (2-digit year mistake) |
| Player D | 7-digit number | looks like a DDMMYYYY date typed into a year field |
| Player E | 8-digit number | looks like a DDMMYYYY date typed into a year field |

(Real IDs and values deliberately omitted — they are personal data.)

**Action taken in feature engineering (notebook 02):** derive the age feature
from `Godiste` (birth year only — no month/day precision, so "age at match
date" is year-granularity, e.g. `YEAR(Datum) - Godiste`), not from
`DatumRodjenja`. Cap/null out age for the 5 implausible rows above rather than
using the raw garbage values. Both `WinnerDatumRodjenja`/`LoserDatumRodjenja`
(NULL) and `WinnerGodiste`/`LoserGodiste` (usable) are in the CSV.

### 4. STB vs Decider frequency

Out of 17,131 in-scope league matches (before the `IgracPar=0` join further
trims to 17,128):

- **4,553 matches (26.6%) went to a 3rd set** (`DeciderCountBefore` support)
- **2,912 matches (17.0%) were super-tiebreaks** (`Set3 >= 10` on either
  side) — i.e. **64% of 3rd-set matches were STBs**, only ~36% were full 3rd
  sets

Both `STBWinRateBefore` and `DeciderWinRateBefore` have reasonable support at
the aggregate level; per-player sparsity still applies (most players
individually will have few STB/decider matches), so the same
shrinkage-toward-0.5 treatment used for H2H (notebook 02) is applied here too.

## Export Sanity Checks (`verify_export.mjs`)

All passed:

- Total rows: **17,128**, columns: **47**
- Rows with `LigaTip='MA'`: **0**
- Rows with `LigaId` NULL: **0**
- Rows with `Datum` before `2017-05-01`: **0**
- Distinct `LigaTip` values present: **`['SM', 'SF']`** only
- **Spot-check player (any in-scope singles player):** contributes **12 rows**, all from
  her SF league matches. Cross-checked directly against the DB: she has 12 SF
  matches, 4 MA (Masters) matches, and 3 friendly matches (`LigaId IS NULL`)
  in the in-scope date range — the export correctly includes only the 12 SF
  rows and excludes both the 4 Masters and 3 friendly matches.
- **`IgracPar=0` enforcement:** spot-checked 5 known pair IDs (`IgracPar=1`,
  e.g. a known doubles pair) — all confirmed absent from the export.

## Refreshing the dataset (pseudonymised workflow)

The raw export contains real player IDs and birth years and must never be written inside this
repository. Pipeline used for refreshes:

```bash
cd scripts
# 1. read-only export -> raw CSV OUTSIDE the repo (free text / DatumRodjenja are not exported)
DB_CONFIG=/path/db-config.json OUT_PATH=/private/dir/matches_export_raw.csv node export_data.mjs
# 2. pseudonymise (ids -> 100001+, birth year -> age at match year; the id map stays private
#    and is extended in place for new players)
python anonimiziraj.py --raw /private/dir/matches_export_raw.csv \
    --map /private/dir/_pseudonim_mapa.json --out /private/dir/anon/matches_export.csv
# 3. rebuild features + all experiments in a throw-away workspace (notebooks stay output-free)
python run_pipeline.py --matches /private/dir/anon/matches_export.csv \
    --workdir /private/dir/run --notebooks 02 03 05 07 08 10
python /private/dir/run/scripts/run_walkforward_c3c4.py --skip-c4   # 4-fold walk-forward
```

Notebook 02 accepts either the raw columns (`WinnerGodiste`/`LoserGodiste`) or the pseudonymised
`WinnerAge`/`LoserAge`. Exp C4 needs the Gemini free-text scores (`trenutci_scores.csv`), which
are only available for the July 2026 snapshot; `--skip-c4` skips it. Enable the pre-commit hook
that strips notebook outputs once per clone: `git config core.hooksPath .githooks`.
