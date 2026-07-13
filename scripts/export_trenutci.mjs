// ML text-feature v2 — exports Igrac.Trenutci (self-description at signup) for in-scope
// singles players, to be scored offline by Gemini (see score_trenutci.mjs).
// Read-only SQL Server query (your_readonly_user). Never writes to DB.
// Run: node export_trenutci.mjs
import sql from 'mssql';
import { readFileSync, writeFileSync } from 'node:fs';

const cfg = JSON.parse(readFileSync(new URL('./db-config.json', import.meta.url)));
const config = {
  server: cfg.server, port: cfg.port, database: cfg.database,
  user: cfg.user, password: cfg.password, options: cfg.options,
  connectionTimeout: cfg.connectionTimeout, requestTimeout: cfg.requestTimeout,
};

const inscopeIds = new Set(
  JSON.parse(readFileSync(new URL('../data/inscope_player_ids.json', import.meta.url)))
);

async function main() {
  const pool = await sql.connect(config);
  const result = await pool.request().query(`
    SELECT IgracId, Ime, Prezime, Trenutci
    FROM Igrac
    WHERE IgracPar = 0 AND Trenutci IS NOT NULL AND LTRIM(RTRIM(Trenutci)) != ''
  `);
  await pool.close();

  const all = result.recordset;
  console.log(`DB rows with non-empty Trenutci (IgracPar=0): ${all.length}`);

  const inscopeWithText = all.filter((r) => inscopeIds.has(r.IgracId));
  console.log(`In-scope players (${inscopeIds.size} total) with non-empty Trenutci: ${inscopeWithText.length}`);
  console.log(
    `Coverage: ${((inscopeWithText.length / inscopeIds.size) * 100).toFixed(1)}%`
  );

  writeFileSync(
    new URL('../data/trenutci_export.json', import.meta.url),
    JSON.stringify(inscopeWithText, null, 2)
  );
  console.log('Wrote ../data/trenutci_export.json');
}

main().catch((err) => {
  console.error('ERROR', err);
  process.exit(1);
});
