// One-off helper — rebuilds the final trenutci_scores.csv from a raw scores
// JSONL file (one score record per line) joined against the Trenutci export.
// Local-only, never deployed.
import fs from 'fs';
import { fileURLToPath } from 'node:url';

const DATA_DIR = fileURLToPath(new URL('../data/', import.meta.url));
const exportPath = DATA_DIR + 'trenutci_export.json';
const scoresPath = DATA_DIR + 'scores.jsonl';
const outPath = DATA_DIR + 'trenutci_scores.csv';

const players = JSON.parse(fs.readFileSync(exportPath, 'utf8'));
const playerById = new Map(players.map((p) => [p.IgracId, p]));

const scoreLines = fs.readFileSync(scoresPath, 'utf8').split('\n').filter((l) => l.trim().length > 0);
const scores = scoreLines.map((l) => JSON.parse(l));
const scoreById = new Map(scores.map((s) => [s.IgracId, s]));

// Check coverage
const missing = players.filter((p) => !scoreById.has(p.IgracId));
const extra = scores.filter((s) => !playerById.has(s.IgracId));
console.log('players:', players.length, 'scores:', scores.length);
console.log('missing scores for players:', missing.length, missing.map((p) => p.IgracId));
console.log('extra scores with no player:', extra.length, extra.map((s) => s.IgracId));

// Check duplicates in scores
const seen = new Set();
const dupes = [];
for (const s of scores) {
  if (seen.has(s.IgracId)) dupes.push(s.IgracId);
  seen.add(s.IgracId);
}
console.log('duplicate IgracId in scores:', dupes);

function csvEscape(val) {
  if (val === null || val === undefined) return '';
  const str = String(val);
  if (/[",\n]/.test(str)) {
    return '"' + str.replace(/"/g, '""') + '"';
  }
  return str;
}

const header = ['IgracId', 'Ime', 'Prezime', 'skill_1_10', 'ntrp_est', 'confidence', 'reasoning', 'trenutci_raw'];
const rows = [header.join(',')];

let count = 0;
for (const p of players) {
  const s = scoreById.get(p.IgracId);
  if (!s) continue;
  const row = [
    p.IgracId,
    p.Ime,
    p.Prezime,
    s.skill,
    s.ntrp,
    s.conf,
    s.reason,
    p.Trenutci,
  ].map(csvEscape);
  rows.push(row.join(','));
  count++;
}

fs.writeFileSync(outPath, rows.join('\n') + '\n', 'utf8');
console.log('wrote', count, 'rows to', outPath);
