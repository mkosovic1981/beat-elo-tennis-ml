"""
Pseudonymise a RAW matches export (output of export_data.mjs) for research use.

  python anonimiziraj.py --raw <raw.csv> --map <_pseudonim_mapa.json> --out <matches_export.csv>

* IgracId -> stable pseudonym (>= 100001). Players already in --map keep their pseudonym; new
  players get fresh random pseudonyms continuing after max(map values). The map file is EXTENDED
  IN PLACE (a timestamped .bak copy is written next to it first). The map must live outside this
  repository (it is the re-identification key) and is never shipped anywhere.
* Birth year (Godiste) -> age at match year (same [1930, 2015] QC as notebook 02), then dropped.
  Free text (Trenutci), DatumRodjenja and Godiste columns are dropped.
* Final assertions: no free-text / birth columns left, every id >= 100001.
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--map', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    m = pd.read_csv(a.raw)
    mp_path = Path(a.map)
    mp = {int(k): int(v) for k, v in json.load(open(mp_path)).items()}
    ids = sorted(set(m.WinnerId) | set(m.LoserId))
    new = [int(i) for i in ids if int(i) not in mp]
    if new:
        shutil.copy(mp_path, mp_path.with_name(mp_path.name + f'.bak-{time.strftime("%Y%m%d%H%M%S")}'))
        start = max(mp.values()) + 1
        pool = list(range(start, start + len(new)))
        random.SystemRandom().shuffle(pool)
        mp.update(zip(new, pool))
        json.dump({str(k): v for k, v in mp.items()}, open(mp_path, 'w'))
    assert len(set(mp.values())) == len(mp), 'pseudonym collision'

    yr = pd.to_datetime(m.Datum).dt.year
    for w in ('Winner', 'Loser'):
        if w + 'Godiste' in m.columns:
            g = m[w + 'Godiste']
            m[w + 'Age'] = (yr - g).where((g >= 1930) & (g <= 2015))
        m[w + 'Id'] = m[w + 'Id'].map(lambda v: mp[int(v)])
    m = m.drop(columns=[c for c in m.columns if any(k in c for k in ('Trenutci', 'DatumRodjenja', 'Godiste'))])
    assert not any(any(k in c for k in ('Trenutci', 'Godiste', 'DatumRodjenja', 'Ime', 'Prezime', 'Mail')) for c in m.columns)
    assert min(m.WinnerId.min(), m.LoserId.min()) >= 100001
    m.to_csv(a.out, index=False)
    print(f'wrote {a.out}: {len(m)} matches, {len(set(m.WinnerId) | set(m.LoserId))} players, '
          f'{len(new)} new pseudonyms, {m.Datum.min()} .. {m.Datum.max()}')


if __name__ == '__main__':
    main()
