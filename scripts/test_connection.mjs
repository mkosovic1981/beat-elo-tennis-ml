// One-off connectivity check against the read-only SQL Server DB.
// Local-only, never deployed. Run: node test_connection.mjs
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

try {
  const pool = await sql.connect(config);
  const result = await pool.request().query('SELECT TOP 1 SezonaId, SezonaNaziv, DatumOd FROM Sezona ORDER BY SezonaId DESC');
  console.log('Connected OK. Sample row:', result.recordset);
  await pool.close();
} catch (err) {
  console.error('Connection failed:', err.message);
  process.exit(1);
}
