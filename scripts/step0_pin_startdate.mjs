// STEP 0 — pin the exact @StartDate boundary before running the exporter.
// Local-only, never deployed. Run: node step0_pin_startdate.mjs
import sql from 'mssql';
import { readFileSync } from 'node:fs';

const cfg = JSON.parse(readFileSync(new URL('./db-config.json', import.meta.url)));

const config = {
  server: cfg.server,
  port: cfg.port,
  database: cfg.database,
  user: cfg.user,
  password: cfg.password,
  options: cfg.options,
  connectionTimeout: cfg.connectionTimeout,
  requestTimeout: cfg.requestTimeout,
};

const pool = await sql.connect(config);
const result = await pool.request().query(`
  SELECT SezonaId, SezonaNaziv, DatumOd, DatumDo
  FROM Sezona
  WHERE DatumOd BETWEEN '2017-04-01' AND '2017-06-30'
  ORDER BY DatumOd ASC;
`);
console.log(result.recordset);
await pool.close();
