"""
Aggregate-only overview figures for notebooks/00_pregled.ipynb -> notebooks/figures/*.png
  python make_overview_figures.py <nov_data_dir> <jul_data_dir>
Read-only on the data dirs. No row-level data, IDs or names are plotted.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from summarize_metrics import MODELS, corp  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / 'notebooks' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
SHORT = {'Exp 0 ELO': 'ELO', 'Exp A': 'A', 'Exp B': 'B', 'Exp C (ELO-free GRU)': 'C', "Exp A'": "A'",
         "Exp B'": "B'", 'Exp C2': 'C2', 'Exp C3': 'C3', 'Exp C4': 'C4'}


def load(d):
    d = Path(d)
    base = pd.read_parquet(d / 'test_predictions_4way.parquet').reset_index(drop=True)
    key = base[['SusretId', 'Perspective']]
    y = base['Label'].values.astype(float)
    preds = {}
    for label, f, col in MODELS:
        fp = d / f'{f}.parquet'
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        if col is None:
            col = next(c for c in df.columns if c.startswith('p_') and c.endswith('_calibrated'))
        df = key.merge(df[['SusretId', 'Perspective', col]], on=['SusretId', 'Perspective'], how='left')
        preds[SHORT[label]] = df[col].values.astype(float)
    return y, preds


def main():
    nov, jul = sys.argv[1], sys.argv[2]
    yn, pn = load(nov)
    yj, pj = load(jul)
    brier = lambda y, p: float(np.mean((p - y) ** 2))

    # 1 Brier Nov vs Jul
    names = [k for k in pn if k in pj or True]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, [brier(yn, pn[k]) for k in names], w, label='Nov 2026')
    ax.bar(x + w / 2, [brier(yj, pj[k]) if k in pj else np.nan for k in names], w, label='Jul 2026')
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0.15, 0.21)
    ax.set_ylabel('Brier (lower is better)'); ax.legend(); ax.set_title('Test Brier by model')
    fig.tight_layout(); fig.savefig(OUT / 'brier_nov_vs_jul.png', dpi=130); plt.close(fig)

    # 2 calibration curves (equal-mass bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    for k in ('ELO', 'C2', 'C3'):
        p = pn[k]; o = np.argsort(p, kind='stable')
        bins = np.array_split(o, 10)
        ax.plot([p[i].mean() for i in bins], [yn[i].mean() for i in bins], 'o-', label=k)
    ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed win rate'); ax.legend()
    ax.set_title('Calibration, Nov (10 equal-mass bins)')
    fig.tight_layout(); fig.savefig(OUT / 'calibration_nov.png', dpi=130); plt.close(fig)

    # 3 walk-forward
    wf = pd.DataFrame(json.load(open(Path(nov) / 'walkforward_c3c4_results.json')))
    wf = wf[wf.experiment.str.match(r"Exp (0|B'|C2|C3)\b")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for e, g in wf.groupby('experiment'):
        g = g.sort_values('fold')
        ax.plot(g.fold, g.brier, 'o-', label=e.split(' - ')[0].replace('Exp 0', 'ELO').replace('Exp ', ''))
    ax.set_xticks(sorted(wf.fold.unique())); ax.set_xlabel('Fold'); ax.set_ylabel('Brier')
    ax.set_title('Walk-forward Brier by fold (Nov)'); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / 'walkforward_nov.png', dpi=130); plt.close(fig)

    # 4 MCB / DSC
    show = [k for k in ('ELO', 'A', 'B', 'C', "A'", "B'", 'C2', 'C3')]
    c = {k: corp(pn[k], yn) for k in show}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    axs[0].bar(show, [c[k][0] for k in show]); axs[0].set_title('MCB (miscalibration, lower better)')
    axs[1].bar(show, [c[k][1] for k in show], color='tab:green'); axs[1].set_title('DSC (discrimination, higher better)')
    axs[1].set_ylim(0.04, 0.09)
    fig.tight_layout(); fig.savefig(OUT / 'mcb_dsc_nov.png', dpi=130); plt.close(fig)
    print('saved to', OUT)


if __name__ == '__main__':
    main()
