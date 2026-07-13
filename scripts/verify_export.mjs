// Sanity checks on ../data/matches_export.csv — row/column counts, scope filters
// (LigaTip, LigaId, date floor), and spot-checks against the source database.
// Local-only, never deployed. Run: node verify_export.mjs
import { readFileSync } from 'node:fs';
import sql from 'mssql';

// Minimal CSV parser that handles quoted fields with embedded commas/newlines
// (Trenutci free-text can contain both).
function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else inQuotes = false; }
      else field += c;
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ',') { row.push(field); field = ''; }
      else if (c === '\n') { if (text[i - 1] !== '\r') { row.push(field); rows.push(row); row = []; field = ''; } }
      else if (c === '\r') { /* skip */ }
      else field += c;
    }
  }
  if (field.length > 0 || row.length > 0) { row.push(field); rows.push(row); }
  return rows.filter((r) => !(r.length === 1 && r[0] === ''));
}

const text = readFileSync(new URL('../data/matches_export.csv', import.meta.url), 'utf-8');
const rows = parseCSV(text);
const header = rows[0];
const data = rows.slice(1);
const idx = (name) => header.indexOf(name);

console.log('Total data rows:', data.length);
console.log('Total columns:', header.length);
console.log('Header:', header.join(', '));

const iLigaTip = idx('LigaTip');
const iLigaId = idx('LigaId');
const iDatum = idx('Datum');
const iWinnerId = idx('WinnerId');
const iLoserId = idx('LoserId');

const startDate = new Date('2017-05-01T00:00:00.000Z');
let maCount = 0, nullLigaId = 0, beforeStart = 0;
for (const r of data) {
  if (r[iLigaTip] === 'MA') maCount++;
  if (!r[iLigaId] || r[iLigaId] === '') nullLigaId++;
  if (new Date(r[iDatum]) < startDate) beforeStart++;
}
console.log('\nRows with LigaTip=MA:', maCount, '(expect 0)');
console.log('Rows with NULL LigaId:', nullLigaId, '(expect 0)');
console.log('Rows with Datum before 2017-05-01:', beforeStart, '(expect 0)');
console.log('Distinct LigaTip values present:', [...new Set(data.map((r) => r[iLigaTip]))]);

// Known Masters participant (in-scope singles player) — must contribute 0 rows from MA
// The player ID is supplied via the SPOT_CHECK_ID env var (IDs are personal data, not hardcoded).
const SPOT_ID = process.env.SPOT_CHECK_ID;
if (SPOT_ID) {
  const spotCheckRows = data.filter((r) => r[iWinnerId] === SPOT_ID || r[iLoserId] === SPOT_ID);
  console.log('\nRows involving spot-check player:', spotCheckRows.length, '(all must be SF, 0 from MA — MA already proven 0 above)');
} else {
  console.log('\nSPOT_CHECK_ID not set — skipping per-player spot check.');
}

// Spot-check known IgracPar=1 pair IDs are absent (fetched live so this stays correct
// even if pair IDs change).
const cfg = JSON.parse(readFileSync(new URL('./db-config.json', import.meta.url)));
const config = {
  server: cfg.server, port: cfg.port, database: cfg.database,
  user: cfg.user, password: cfg.password, options: cfg.options,
  connectionTimeout: cfg.connectionTimeout, requestTimeout: cfg.requestTimeout,
};
const pool = await sql.connect(config);
const pairs = await pool.request().query('SELECT TOP 5 IgracId FROM Igrac WHERE IgracPar = 1 ORDER BY IgracId');
console.log('\nSpot-checking known IgracPar=1 pair IDs are absent from export:');
for (const { IgracId } of pairs.recordset) {
  const pid = String(IgracId);
  const found = data.some((r) => r[iWinnerId] === pid || r[iLoserId] === pid);
  console.log(`  Pair IgracId ${pid} present in export:`, found, found ? '<-- FAIL' : '(OK)');
}
await pool.close();
