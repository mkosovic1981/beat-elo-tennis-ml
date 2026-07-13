// P1 empirical check — Igrac.DatumRodjenja vs Igrac.Godiste.
// Documents a real finding: DatumRodjenja (datetime, the column originally assumed to
// hold birth date) is 100% NULL in this database for all IgracPar=0 players. The
// actually populated birth-year field is `Godiste` (int, e.g. 1988). See ../data/README.md.
// Local-only, never deployed. Run: node check_age_columns.mjs
import sql from 'mssql';
import { readFileSync } from 'node:fs';

const cfg = JSON.parse(readFileSync(new URL('./db-config.json', import.meta.url)));
const config = {
  server: cfg.server, port: cfg.port, database: cfg.database,
  user: cfg.user, password: cfg.password, options: cfg.options,
  connectionTimeout: cfg.connectionTimeout, requestTimeout: cfg.requestTimeout,
};

const pool = await sql.connect(config);

const r1 = await pool.request().query(`
  SELECT COUNT(*) AS total, COUNT(DatumRodjenja) AS filled_datumrodjenja, COUNT(Godiste) AS filled_godiste
  FROM Igrac WHERE IgracPar = 0;
`);
console.log('All IgracPar=0 players — DatumRodjenja vs Godiste population:', r1.recordset[0]);

const r2 = await pool.request().query(`
  SELECT IgracId, Godiste FROM Igrac
  WHERE IgracPar = 0 AND Godiste IS NOT NULL AND (Godiste < 1920 OR Godiste > 2020)
  ORDER BY Godiste;
`);
console.log('Implausible/placeholder Godiste values (outside 1920-2020):', r2.recordset);

await pool.close();
