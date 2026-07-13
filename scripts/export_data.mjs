// ML win-probability project — Phase P1 data export.
// Local-only, NEVER FTP-uploaded, NEVER run against production write path
// (uses a read-only DB user). Reads db-config.json (gitignored).
//
// Exports singles matches (SM/SF only) with point-in-time features via SQL window
// functions, with one documented addition: Igrac.DatumRodjenja was found (P1
// empirical check) to be 100% NULL in this database — the actually-populated
// birth-year field is `Godiste` (int, e.g. 1988). Both are exported: DatumRodjenja
// columns are kept for schema fidelity (they will be NULL), and WinnerGodiste/LoserGodiste
// are added as the real usable age-proxy source. See ../data/README.md.
//
// Run: node export_data.mjs
import sql from 'mssql';
import { createWriteStream } from 'node:fs';
import { readFileSync } from 'node:fs';

const cfg = JSON.parse(readFileSync(new URL('./db-config.json', import.meta.url)));
const config = {
  server: cfg.server, port: cfg.port, database: cfg.database,
  user: cfg.user, password: cfg.password, options: cfg.options,
  connectionTimeout: cfg.connectionTimeout, requestTimeout: cfg.requestTimeout,
};

// Pinned in STEP 0 (Part 1.5): Sezona 99 "2017/05", DatumOd = 2017-05-01
const START_DATE = '2017-05-01';

const QUERY = `
DECLARE @StartDate DATE = '${START_DATE}';

;WITH SinglesMatches AS (
    SELECT
        s.SusretId, s.Datum, s.LigaId, s.SezonaId, s.Lokacija,
        l.LigaTip, l.LigaKoeficjent,
        si_w.IgracId  AS WinnerId, si_l.IgracId  AS LoserId,
        sie_w.EloBefore AS WinnerEloBefore, sie_w.EloAfter AS WinnerEloAfter,
        sie_l.EloBefore AS LoserEloBefore,  sie_l.EloAfter AS LoserEloAfter,
        si_w.Set1 AS W_Set1, si_w.Set2 AS W_Set2, si_w.Set3 AS W_Set3,
        si_l.Set1 AS L_Set1, si_l.Set2 AS L_Set2, si_l.Set3 AS L_Set3,
        iw.Trenutci AS WinnerTrenutci, iw.DatumRodjenja AS WinnerDatumRodjenja, iw.Godiste AS WinnerGodiste,
        il.Trenutci AS LoserTrenutci,  il.DatumRodjenja AS LoserDatumRodjenja,  il.Godiste AS LoserGodiste
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
      AND s.Datum >= @StartDate
),

PlayerMatchLog AS (
    SELECT SusretId, Datum, WinnerId AS IgracId, LoserId AS OpponentId,
           1 AS Won, WinnerEloBefore AS EloBefore,
           LigaKoeficjent, SezonaId, LigaTip
    FROM SinglesMatches
    UNION ALL
    SELECT SusretId, Datum, LoserId AS IgracId, WinnerId AS OpponentId,
           0 AS Won, LoserEloBefore AS EloBefore,
           LigaKoeficjent, SezonaId, LigaTip
    FROM SinglesMatches
),

PlayerFeatures AS (
    SELECT
        SusretId, IgracId, OpponentId, Datum, Won,
        ROW_NUMBER() OVER (PARTITION BY IgracId ORDER BY Datum ASC, SusretId ASC) - 1
            AS MatchCountBefore,
        ROW_NUMBER() OVER (PARTITION BY IgracId, LigaKoeficjent ORDER BY Datum ASC, SusretId ASC) - 1
            AS RazinaMatchCountBefore,
        DATEDIFF(DAY,
            LAG(Datum) OVER (PARTITION BY IgracId ORDER BY Datum ASC, SusretId ASC),
            Datum)
            AS DaysSinceLastMatch,
        AVG(CAST(Won AS FLOAT)) OVER (
            PARTITION BY IgracId ORDER BY Datum ASC, SusretId ASC
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS FormaLast10,
        SUM(CASE WHEN Won = 1 THEN 1 ELSE 0 END) OVER (
            PARTITION BY IgracId, OpponentId ORDER BY Datum ASC, SusretId ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS H2HWinsBefore,
        COUNT(*) OVER (
            PARTITION BY IgracId, OpponentId ORDER BY Datum ASC, SusretId ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS H2HTotalBefore,
        AVG(CAST(Won AS FLOAT)) OVER (
            PARTITION BY IgracId ORDER BY Datum ASC, SusretId ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS CareerWinRateBefore,
        FIRST_VALUE(LigaKoeficjent) OVER (
            PARTITION BY IgracId ORDER BY Datum ASC, SusretId ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS DebutRazina,
        FIRST_VALUE(SezonaId) OVER (
            PARTITION BY IgracId ORDER BY Datum ASC, SusretId ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS DebutSezonaId,
        FIRST_VALUE(LigaTip) OVER (
            PARTITION BY IgracId ORDER BY Datum ASC, SusretId ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS DebutLigaTip
    FROM PlayerMatchLog
),

SeasonGenderPop AS (
    SELECT sli.SezonaId, l.LigaTip,
           COUNT(DISTINCT sli.LigaId)  AS BrojLigaUSezoni,
           COUNT(DISTINCT sli.IgracId) AS BrojIgracaUkupno
    FROM SezonaLigaIgrac sli
    JOIN Liga l ON l.LigaId = sli.LigaId
    WHERE l.LigaTip IN ('SM', 'SF')
    GROUP BY sli.SezonaId, l.LigaTip
),

SeasonLeagueDepth AS (
    SELECT sli.SezonaId, sli.LigaId,
           COUNT(DISTINCT sli.IgracId) AS BrojIgracaULigi
    FROM SezonaLigaIgrac sli
    GROUP BY sli.SezonaId, sli.LigaId
)

SELECT
    m.SusretId, m.Datum, m.LigaId, m.SezonaId, m.LigaTip, m.LigaKoeficjent, m.Lokacija,
    m.WinnerId, m.WinnerEloBefore, m.WinnerEloAfter,
    m.LoserId,  m.LoserEloBefore,  m.LoserEloAfter,
    m.W_Set1, m.W_Set2, m.W_Set3, m.L_Set1, m.L_Set2, m.L_Set3,
    m.WinnerTrenutci, m.WinnerDatumRodjenja, m.WinnerGodiste,
    m.LoserTrenutci,  m.LoserDatumRodjenja,  m.LoserGodiste,
    pw.MatchCountBefore       AS WinnerMatchCountBefore,
    pl.MatchCountBefore       AS LoserMatchCountBefore,
    pw.RazinaMatchCountBefore AS WinnerRazinaMatchCountBefore,
    pl.RazinaMatchCountBefore AS LoserRazinaMatchCountBefore,
    pw.DaysSinceLastMatch     AS WinnerDaysSinceLastMatch,
    pl.DaysSinceLastMatch     AS LoserDaysSinceLastMatch,
    pw.FormaLast10            AS WinnerFormaLast10,
    pl.FormaLast10            AS LoserFormaLast10,
    pw.H2HWinsBefore          AS H2HWinsWinnerBefore,
    pl.H2HWinsBefore          AS H2HWinsLoserBefore,
    pw.H2HTotalBefore         AS H2HTotalBefore,
    pw.CareerWinRateBefore    AS WinnerCareerWinRateBefore,
    pl.CareerWinRateBefore    AS LoserCareerWinRateBefore,
    pw.DebutRazina            AS WinnerDebutRazina,
    pl.DebutRazina            AS LoserDebutRazina,
    sgp_w.BrojLigaUSezoni     AS WinnerDebutBrojLiga,
    sgp_w.BrojIgracaUkupno    AS WinnerDebutBrojIgraca,
    sgp_l.BrojLigaUSezoni     AS LoserDebutBrojLiga,
    sgp_l.BrojIgracaUkupno    AS LoserDebutBrojIgraca,
    sgp.BrojLigaUSezoni       AS BrojLigaUSezoni,
    sgp.BrojIgracaUkupno      AS BrojIgracaUkupno,
    sld.BrojIgracaULigi       AS BrojIgracaULigi
FROM SinglesMatches m
JOIN PlayerFeatures pw ON pw.SusretId = m.SusretId AND pw.IgracId = m.WinnerId
JOIN PlayerFeatures pl ON pl.SusretId = m.SusretId AND pl.IgracId = m.LoserId
LEFT JOIN SeasonGenderPop sgp    ON sgp.SezonaId = m.SezonaId AND sgp.LigaTip = m.LigaTip
LEFT JOIN SeasonLeagueDepth sld  ON sld.SezonaId = m.SezonaId AND sld.LigaId = m.LigaId
LEFT JOIN SeasonGenderPop sgp_w  ON sgp_w.SezonaId = pw.DebutSezonaId AND sgp_w.LigaTip = pw.DebutLigaTip
LEFT JOIN SeasonGenderPop sgp_l  ON sgp_l.SezonaId = pl.DebutSezonaId AND sgp_l.LigaTip = pl.DebutLigaTip
ORDER BY m.Datum ASC, m.SusretId ASC;
`;

function csvEscape(val) {
  if (val === null || val === undefined) return '';
  if (val instanceof Date) return val.toISOString();
  const s = String(val);
  if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

const t0 = Date.now();
const pool = await sql.connect(config);
console.log('Connected. Running export query (this may take a while)...');
const result = await pool.request().query(QUERY);
const elapsedMs = Date.now() - t0;
console.log(`Query finished in ${(elapsedMs / 1000).toFixed(1)}s. Rows: ${result.recordset.length}`);

const rows = result.recordset;
const columns = rows.length > 0 ? Object.keys(rows[0]) : [];
const outPath = new URL('../data/matches_export.csv', import.meta.url);
const stream = createWriteStream(outPath);
stream.write(columns.join(',') + '\n');
for (const row of rows) {
  stream.write(columns.map((c) => csvEscape(row[c])).join(',') + '\n');
}
stream.end();

await new Promise((resolve) => stream.on('finish', resolve));
await pool.close();

console.log(`Wrote ${rows.length} rows, ${columns.length} columns to ${outPath.pathname}`);
console.log('Columns:', columns.join(', '));
