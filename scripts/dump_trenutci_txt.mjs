// One-off debug helper — dumps the raw Trenutci export to a flat pipe-delimited
// text file for quick eyeballing before the Gemini scoring pass. Local-only.
import fs from 'fs';
import { fileURLToPath } from 'node:url';

const DATA_DIR = fileURLToPath(new URL('../data/', import.meta.url));
const data = JSON.parse(fs.readFileSync(DATA_DIR + 'trenutci_export.json', 'utf8'));
const out = data.map((d, i) => `${i + 1}|${d.IgracId}|${d.Ime}|${d.Prezime}|${JSON.stringify(d.Trenutci)}`).join('\n');
const outPath = DATA_DIR + 'trenutci_all.txt';
fs.writeFileSync(outPath, out);
console.log('done', data.length);
