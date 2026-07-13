// Phase P1 empirical data-quality/coverage checks, run against the source SQL Server
// database before the exporter was written (see data/README.md for the findings).
// Local-only, never deployed. Run: node p1_checks.mjs
import sql from 'mssql';
import { readFileSync } from 'node:fs';

const cfg = JSON.parse(readFileSync(new URL('./db-config.json', import.meta.url)));
const config = {
  server: cfg.server, port: cfg.port, database: cfg.database,
  user: cfg.user, password: cfg.password, options: cfg.options,
  connectionTimeout: cfg.connectionTimeout, requestTimeout: cfg.requestTimeout,
};

const START_DATE = '2017-05-01';

const pool = await sql.connect(config);
const req = () => pool.request();

console.log('=== CHECK 1: Igrac.Trenutci population ===');
const c1a = await req().query(`
  SELECT COUNT(*) AS total, COUNT(Trenutci) AS filled
  FROM Igrac WHERE IgracPar = 0;
`);
console.log('All singles/individual players (IgracPar=0):', c1a.recordset[0]);

const c1b = await req().query(`
  SELECT COUNT(DISTINCT i.IgracId) AS total_players_in_scope,
         COUNT(DISTINCT CASE WHEN i.Trenutci IS NOT NULL AND LTRIM(RTRIM(i.Trenutci)) != '' THEN i.IgracId END) AS filled_players_in_scope
  FROM Igrac i
  WHERE i.IgracPar = 0
    AND EXISTS (
      SELECT 1
      FROM Susret s
      JOIN Liga l ON s.LigaId = l.LigaId
      JOIN SusretIgrac si ON si.SusretId = s.SusretId AND si.IgracId = i.IgracId
      WHERE l.LigaTip IN ('SM','SF')
        AND (s.NijeOdigran = 0 OR s.NijeOdigran IS NULL)
        AND s.Datum >= '${START_DATE}'
    );
`);
console.log('Players WITH >=1 in-scope SM/SF match since 2017-05-01:', c1b.recordset[0]);

console.log('\n=== CHECK 2: Lokacija cardinality/population for in-scope LEAGUE matches ===');
const c2 = await req().query(`
  SELECT COUNT(*) AS total_inscope_matches,
         COUNT(s.Lokacija) AS non_null_lokacija,
         COUNT(DISTINCT s.Lokacija) AS distinct_raw_values,
         COUNT(DISTINCT LOWER(LTRIM(RTRIM(s.Lokacija)))) AS distinct_normalized_lower_trim
  FROM Susret s
  JOIN Liga l ON s.LigaId = l.LigaId
  WHERE l.LigaTip IN ('SM','SF')
    AND (s.NijeOdigran = 0 OR s.NijeOdigran IS NULL)
    AND s.Datum >= '${START_DATE}';
`);
console.log(c2.recordset[0]);

const c2top = await req().query(`
  SELECT TOP 20 s.Lokacija, COUNT(*) AS cnt
  FROM Susret s
  JOIN Liga l ON s.LigaId = l.LigaId
  WHERE l.LigaTip IN ('SM','SF')
    AND (s.NijeOdigran = 0 OR s.NijeOdigran IS NULL)
    AND s.Datum >= '${START_DATE}'
  GROUP BY s.Lokacija
  ORDER BY cnt DESC;
`);
console.log('Top 20 Lokacija values by frequency:');
console.table(c2top.recordset);

console.log('\n=== CHECK 3: DatumRodjenja population + plausibility ===');
const c3a = await req().query(`
  SELECT COUNT(*) AS total, COUNT(DatumRodjenja) AS filled,
         SUM(CASE WHEN DatumRodjenja IS NOT NULL AND YEAR(DatumRodjenja) <= 1901 THEN 1 ELSE 0 END) AS placeholder_1900_or_earlier
  FROM Igrac WHERE IgracPar = 0;
`);
console.log('All singles players — DatumRodjenja population:', c3a.recordset[0]);

const c3b = await req().query(`
  WITH InScope AS (
    SELECT si.IgracId, s.Datum
    FROM Susret s
    JOIN Liga l ON s.LigaId = l.LigaId
    JOIN SusretIgrac si ON si.SusretId = s.SusretId
    JOIN Igrac i ON i.IgracId = si.IgracId AND i.IgracPar = 0
    WHERE l.LigaTip IN ('SM','SF')
      AND (s.NijeOdigran = 0 OR s.NijeOdigran IS NULL)
      AND s.Datum >= '${START_DATE}'
  )
  SELECT
    COUNT(*) AS total_player_match_rows,
    SUM(CASE WHEN i.DatumRodjenja IS NULL THEN 1 ELSE 0 END) AS null_birthdate_rows,
    SUM(CASE WHEN i.DatumRodjenja IS NOT NULL AND DATEDIFF(YEAR, i.DatumRodjenja, isc.Datum) < 10 THEN 1 ELSE 0 END) AS age_under_10_rows,
    SUM(CASE WHEN i.DatumRodjenja IS NOT NULL AND DATEDIFF(YEAR, i.DatumRodjenja, isc.Datum) > 90 THEN 1 ELSE 0 END) AS age_over_90_rows,
    SUM(CASE WHEN i.DatumRodjenja IS NOT NULL AND YEAR(i.DatumRodjenja) <= 1901 THEN 1 ELSE 0 END) AS placeholder_1900_rows
  FROM InScope isc
  JOIN Igrac i ON i.IgracId = isc.IgracId;
`);
console.log('In-scope match rows — age plausibility at match date:', c3b.recordset[0]);

console.log('\n=== CHECK 4: STB vs Decider frequency ===');
const c4 = await req().query(`
  WITH InScopeMatches AS (
    SELECT s.SusretId,
           si_w.Set3 AS W_Set3, si_l.Set3 AS L_Set3
    FROM Susret s
    JOIN Liga l ON s.LigaId = l.LigaId
    JOIN SusretIgrac si_w ON si_w.SusretId = s.SusretId AND si_w.Pobjednik = 1
    JOIN SusretIgrac si_l ON si_l.SusretId = s.SusretId AND si_l.Pobjednik = 0 AND si_l.IgracId != si_w.IgracId
    JOIN Igrac iw ON iw.IgracId = si_w.IgracId AND iw.IgracPar = 0
    JOIN Igrac il ON il.IgracId = si_l.IgracId AND il.IgracPar = 0
    WHERE l.LigaTip IN ('SM','SF')
      AND (s.NijeOdigran = 0 OR s.NijeOdigran IS NULL)
      AND s.Datum >= '${START_DATE}'
  )
  SELECT
    COUNT(*) AS total_matches,
    SUM(CASE WHEN W_Set3 IS NOT NULL OR L_Set3 IS NOT NULL THEN 1 ELSE 0 END) AS decider_3rd_set_matches,
    SUM(CASE WHEN (W_Set3 IS NOT NULL AND W_Set3 >= 10) OR (L_Set3 IS NOT NULL AND L_Set3 >= 10) THEN 1 ELSE 0 END) AS super_tiebreak_matches
  FROM InScopeMatches;
`);
console.log(c4.recordset[0]);

await pool.close();
