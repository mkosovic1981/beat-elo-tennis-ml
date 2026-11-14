"""
Re-run the notebook pipeline end-to-end on ANY (pseudonymised) matches_export.csv, without
touching the committed notebooks.

The notebooks read `../data` and write `../models`, so this runner builds a throw-away workspace

    <workdir>/notebooks/   copies of the chosen notebooks (executed in place -> outputs live ONLY here)
    <workdir>/scripts/     copy of scripts/ (notebook 12 imports wf_c3c4_common from it)
    <workdir>/serving/     copy of serving/ (notebooks 10/11 import serving.feature_pipeline)
    <workdir>/data/        ONLY matches_export.csv copied from --data-dir (everything else is rebuilt)
    <workdir>/models/      artefacts / charts

and executes the notebooks in order. The committed notebooks stay output-free.

Example (July snapshot, reproduces the published numbers):
    python scripts/run_pipeline.py --matches ../gdsi/ml/data_anon/matches_export.csv \
        --workdir ../gdsi/ml/runs/jul --notebooks 02 03 05 07 08 10

Keep <workdir> OUTSIDE this repository (it holds data derived from the match export).
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matches', required=True, help='path to (pseudonymised) matches_export.csv')
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--notebooks', nargs='+', required=True, help='notebook number prefixes, e.g. 02 03 05')
    ap.add_argument('--extra-data', nargs='*', default=[], help='extra files to copy into workdir/data')
    ap.add_argument('--timeout', type=int, default=-1)
    a = ap.parse_args()

    work = Path(a.workdir).resolve()
    if REPO in work.parents or work == REPO:
        sys.exit('workdir must be outside the repository')
    for d in ('notebooks', 'data', 'models'):
        (work / d).mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO / 'scripts', work / 'scripts', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('node_modules', '__pycache__', 'db-config.json'))
    shutil.copytree(REPO / 'serving', work / 'serving', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))  # notebook 10/11 import serving.*
    shutil.copy(a.matches, work / 'data' / 'matches_export.csv')
    for e in a.extra_data:
        shutil.copy(e, work / 'data' / Path(e).name)

    for prefix in a.notebooks:
        src = next((REPO / 'notebooks').glob(f'{prefix}_*.ipynb'))
        dst = work / 'notebooks' / src.name
        shutil.copy(src, dst)
        nb = nbformat.read(dst, as_version=4)
        t0 = time.time()
        print(f'[{time.strftime("%H:%M:%S")}] running {src.name}', flush=True)
        ep = ExecutePreprocessor(timeout=a.timeout, kernel_name='python3')
        try:
            ep.preprocess(nb, {'metadata': {'path': str(work / 'notebooks')}})
        finally:
            nbformat.write(nb, dst)
        print(f'[{time.strftime("%H:%M:%S")}] done {src.name} in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
