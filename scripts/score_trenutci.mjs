// ML text-feature v2 — one-time offline Gemini scoring of Igrac.Trenutci self-descriptions.
// Converts free-text Croatian self-description (written at signup) into a numeric SKILL
// estimate, so Marin can eyeball the numbers before deciding whether to use it as a model
// feature. Reads GEMINI_API_KEY from project root .env.local. Never commits/writes to DB.
//
// Free-tier gemini-2.5-flash quota on this API key is 5 requests/minute (empirically
// confirmed) — this script paces requests sequentially (~13s apart) to stay under that,
// with retry/backoff on 429 (parses server-provided retryDelay when present).
//
// Resumable: appends each result to a JSONL checkpoint file as it goes, and skips players
// already present in the checkpoint on restart (so an interrupted run doesn't lose progress
// or waste quota re-scoring).
//
// Usage:
//   node score_trenutci.mjs --limit 10          (validation run, separate output file)
//   node score_trenutci.mjs                     (full batch, ~671 players, resumable)
//   node score_trenutci.mjs --finalize          (rebuild CSV from checkpoint only, no calls)
//
// Input:  ../data/trenutci_export.json      (from export_trenutci.mjs)
// Checkpoint: ../data/trenutci_scores_checkpoint.jsonl  (gitignored — data/*.jsonl not
//             matched by existing patterns, but contains PII; see .gitignore update)
// Output: ../data/trenutci_scores.csv       (gitignored — data/*.csv)

import { readFileSync, writeFileSync, appendFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const limitArg = args.indexOf('--limit');
const LIMIT = limitArg !== -1 ? Number(args[limitArg + 1]) : null;
const FINALIZE_ONLY = args.includes('--finalize');

const MIN_INTERVAL_MS = 13000; // ~4.6 req/min, safety margin under the 5 RPM free-tier cap
const TIMEOUT_MS = 20000;
const MAX_RETRIES = 3;
const MODEL = 'gemini-2.5-flash';

const DATA_DIR = fileURLToPath(new URL('../data/', import.meta.url));
const EXPORT_PATH = DATA_DIR + 'trenutci_export.json';
const CHECKPOINT_PATH = DATA_DIR + (LIMIT ? 'trenutci_scores_sample_checkpoint.jsonl' : 'trenutci_scores_checkpoint.jsonl');
const OUTPUT_CSV = DATA_DIR + (LIMIT ? 'trenutci_scores_sample.csv' : 'trenutci_scores.csv');

function loadEnvLocal() {
  const envPath = fileURLToPath(new URL('../.env.local', import.meta.url));
  const text = readFileSync(envPath, 'utf8');
  const match = text.match(/^GEMINI_API_KEY=(.+)$/m);
  if (!match) throw new Error('GEMINI_API_KEY not found in .env.local');
  return match[1].trim().replace(/^["']|["']$/g, '');
}

const GEMINI_API_KEY = loadEnvLocal();
const ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${GEMINI_API_KEY}`;

const RESPONSE_SCHEMA = {
  type: 'OBJECT',
  properties: {
    skill_1_10: { type: 'INTEGER' },
    ntrp_est: { type: 'NUMBER' },
    confidence: { type: 'NUMBER' },
    reasoning: { type: 'STRING' },
  },
  required: ['skill_1_10', 'ntrp_est', 'confidence', 'reasoning'],
};

const SYSTEM_PROMPT = `Ti si stručnjak za tenis koji procjenjuje razinu igrača na temelju kratkog
samoopisa koji je igrač napisao prilikom prijave u rekreativnu tenisku ligu (Tenis Liga Split).

Tekst je na hrvatskom jeziku, često kratak, katkad nepotpun ili nejasan.

Procijeni razinu igrača uzimajući u obzir (ako je spomenuto):
- natjecateljsku povijest (npr. "natjecao se", bivši juniorski igrač, klupski staž)
- samoopisanu razinu (npr. "početnik", "rekreativac", "srednja razina", "napredni")
- eksplicitno spomenut NTRP rejting
- volumen/učestalost treninga (koliko puta tjedno/mjesečno igra ili trenira)
- broj godina igranja tenisa
- eksplicitno samoopisano kao početnik ("početnik", "nikad nisam igrao", "rekreativno")

Za svaki tekst vrati STROGO strukturirani JSON s poljima:
- skill_1_10: cijeli broj 1-10, gdje je 1 = potpuni početnik bez ikakvog iskustva,
  10 = bivši natjecateljski igrač / vrlo jak igrač (npr. bivši profesionalac, dugogodišnji
  natjecatelj, visoki NTRP)
- ntrp_est: procjena po standardnoj NTRP skali (1.0-7.0, decimalni broj, npr. 3.5)
- confidence: 0-1, koliko teksta zapravo nosi signal o razini (kratak/prazan/nejasan tekst
  = niska pouzdanost, npr. 0.1-0.3; detaljan tekst s konkretnim podacima = visoka pouzdanost,
  npr. 0.7-0.95)
- reasoning: jedna kratka rečenica na hrvatskom koja objašnjava zaključak

Ako je tekst prekratak, prazan ili ne nosi nikakav signal o teniskoj razini (npr. samo ime,
šala, nepovezan tekst), dodijeli skill_1_10 blizu srednje vrijednosti (5) uz NISKU confidence
(<= 0.2) i to navedi u reasoning.

Odgovori ISKLJUČIVO JSON objektom prema shemi, bez dodatnog teksta.`;

function withTimeout(promiseFactory, ms) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), ms);
  return promiseFactory(controller.signal).finally(() => clearTimeout(timeout));
}

function parseRetryDelaySeconds(errText) {
  const m = errText.match(/"retryDelay":\s*"(\d+)s"/);
  return m ? Number(m[1]) : null;
}

async function scoreOne(text) {
  const body = {
    system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
    contents: [{ role: 'user', parts: [{ text: `Samoopis igrača:\n"""${text}"""` }] }],
    generationConfig: {
      responseMimeType: 'application/json',
      responseSchema: RESPONSE_SCHEMA,
      temperature: 0.2,
    },
  };

  let lastErr;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const res = await withTimeout(
        (signal) =>
          fetch(ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal,
          }),
        TIMEOUT_MS
      );
      if (!res.ok) {
        const errText = await res.text();
        if (res.status === 429) {
          const retrySec = parseRetryDelaySeconds(errText);
          const waitMs = (retrySec ? retrySec + 2 : 20 * (attempt + 1)) * 1000;
          await new Promise((r) => setTimeout(r, waitMs));
        }
        throw new Error(`HTTP ${res.status}: ${errText.slice(0, 300)}`);
      }
      const json = await res.json();
      const raw = json?.candidates?.[0]?.content?.parts?.[0]?.text;
      if (!raw) throw new Error('No text in Gemini response: ' + JSON.stringify(json).slice(0, 300));
      const parsed = JSON.parse(raw);
      parsed.skill_1_10 = Math.max(1, Math.min(10, Math.round(parsed.skill_1_10)));
      parsed.ntrp_est = Math.max(1.0, Math.min(7.0, Number(parsed.ntrp_est)));
      parsed.confidence = Math.max(0, Math.min(1, Number(parsed.confidence)));
      return parsed;
    } catch (err) {
      lastErr = err;
      if (attempt < MAX_RETRIES && !String(err.message).startsWith('HTTP 429')) {
        await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
      }
    }
  }
  throw lastErr;
}

function csvEscape(val) {
  if (val === null || val === undefined) return '';
  const s = String(val);
  if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
  return s;
}

function loadCheckpoint() {
  const map = new Map();
  if (existsSync(CHECKPOINT_PATH)) {
    const lines = readFileSync(CHECKPOINT_PATH, 'utf8').split('\n').filter(Boolean);
    for (const line of lines) {
      try {
        const rec = JSON.parse(line);
        map.set(rec.IgracId, rec);
      } catch {
        // skip malformed line
      }
    }
  }
  return map;
}

function writeFinalCsv(checkpointMap, orderedPlayers) {
  const header = 'IgracId,Ime,Prezime,Trenutci,skill_1_10,ntrp_est,confidence,reasoning,error';
  const lines = [header];
  for (const p of orderedPlayers) {
    const r = checkpointMap.get(p.IgracId) || { ...p, skill_1_10: '', ntrp_est: '', confidence: '', reasoning: '', error: 'NOT_SCORED' };
    lines.push(
      [
        r.IgracId,
        csvEscape(r.Ime),
        csvEscape(r.Prezime),
        csvEscape(r.Trenutci),
        r.skill_1_10,
        r.ntrp_est,
        r.confidence,
        csvEscape(r.reasoning),
        csvEscape(r.error || ''),
      ].join(',')
    );
  }
  writeFileSync(OUTPUT_CSV, lines.join('\n'), 'utf8');
  console.log(`Wrote ${OUTPUT_CSV}`);
}

async function main() {
  const players = JSON.parse(readFileSync(EXPORT_PATH, 'utf8'));
  const list = LIMIT ? players.slice(0, LIMIT) : players;
  const checkpointMap = loadCheckpoint();

  if (FINALIZE_ONLY) {
    console.log(`Finalizing CSV from checkpoint (${checkpointMap.size} scored)...`);
    writeFinalCsv(checkpointMap, list);
    return;
  }

  const todo = list.filter((p) => !checkpointMap.has(p.IgracId));
  console.log(
    `Total in-scope players with text: ${list.length}. Already scored (checkpoint): ${checkpointMap.size}. Remaining: ${todo.length}.`
  );
  console.log(`Model: ${MODEL}. Pacing: 1 request / ${MIN_INTERVAL_MS / 1000}s.`);

  let done = 0;
  let failed = 0;
  const total = todo.length;

  for (const p of todo) {
    const startedAt = Date.now();
    let record;
    try {
      const scored = await scoreOne(p.Trenutci);
      record = { ...p, ...scored, error: '' };
    } catch (err) {
      failed++;
      record = {
        ...p,
        skill_1_10: '',
        ntrp_est: '',
        confidence: '',
        reasoning: '',
        error: String(err.message || err),
      };
    }
    appendFileSync(CHECKPOINT_PATH, JSON.stringify(record) + '\n', 'utf8');
    checkpointMap.set(p.IgracId, record);
    done++;
    if (done % 10 === 0 || done === total) {
      console.log(`  progress: ${done}/${total} (failed so far: ${failed})`);
    }
    const elapsed = Date.now() - startedAt;
    const waitMore = MIN_INTERVAL_MS - elapsed;
    if (waitMore > 0 && done < total) {
      await new Promise((r) => setTimeout(r, waitMore));
    }
  }

  console.log(`Done. ${done - failed} scored OK this run, ${failed} failed this run.`);
  writeFinalCsv(checkpointMap, list);
}

main().catch((err) => {
  console.error('FATAL', err);
  process.exit(1);
});
