// Extracts the unique set of in-scope IgracId values (players with >=1 in-scope
// singles match since 2017-05) from ../data/matches_export.csv (WinnerId + LoserId columns).
// Writes ../data/inscope_player_ids.json (gitignored — data/*.json not covered, but it's
// pure numeric IDs, no PII; still keep it out of git as it's a derived data artifact).
import { readFileSync, writeFileSync } from 'node:fs';
import { parseCsv } from './csv-utils.mjs';

const csvPath = new URL('../data/matches_export.csv', import.meta.url);
const text = readFileSync(csvPath, 'utf8');
const rows = parseCsv(text);
const header = rows[0];
const winnerIdx = header.indexOf('WinnerId');
const loserIdx = header.indexOf('LoserId');

if (winnerIdx === -1 || loserIdx === -1) {
  console.error('Could not find WinnerId/LoserId columns in header:', header);
  process.exit(1);
}

const ids = new Set();
for (let r = 1; r < rows.length; r++) {
  const row = rows[r];
  if (row.length < 2) continue;
  const w = row[winnerIdx];
  const l = row[loserIdx];
  if (w) ids.add(Number(w));
  if (l) ids.add(Number(l));
}

const sorted = Array.from(ids).filter((n) => Number.isFinite(n)).sort((a, b) => a - b);
console.log(`Parsed ${rows.length - 1} match rows.`);
console.log(`Unique in-scope IgracId count: ${sorted.length}`);

writeFileSync(
  new URL('../data/inscope_player_ids.json', import.meta.url),
  JSON.stringify(sorted, null, 0)
);
console.log('Wrote ../data/inscope_player_ids.json');
