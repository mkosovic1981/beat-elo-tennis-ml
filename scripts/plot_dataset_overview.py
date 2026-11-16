"""
Aggregate-only "Dataset overview" figure -> notebooks/figures/dataset_overview.png
  python scripts/plot_dataset_overview.py <workdir>      (or env GDSI_ML_WORKDIR)
<workdir> must contain data/matches_export.csv (pseudonymised export). Only aggregate
values are plotted: no row-level data, IDs or names. Read-only on the workdir.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_SM, C_SF, C_ELO = '#1f5fa8', '#d9532b', '#7f7f7f'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'], 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#444444',
                     'axes.labelcolor': '#222222', 'xtick.color': '#222222', 'ytick.color': '#222222',
                     'axes.axisbelow': True})
OUT = Path(__file__).resolve().parent.parent / 'notebooks' / 'figures' / 'dataset_overview.png'


def title(ax, t):
    ax.set_title(t, loc='left', fontsize=12, fontweight='bold')
    ax.grid(color='#e3e3e3', lw=0.7)


def main():
    wd = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('GDSI_ML_WORKDIR')
    if not wd:
        sys.exit('usage: plot_dataset_overview.py <workdir>  (or set GDSI_ML_WORKDIR)')
    f = Path(wd) / 'data' / 'matches_export.csv'
    if not f.exists():
        f = Path(wd) / 'matches_export.csv'
    d = pd.read_csv(f)
    d = d[d.LigaTip.isin(['SM', 'SF'])]
    # W.O. rows are stored as 6:0 6:0 (sets 6,6 vs 0,0) in the source system
    wo = (d.W_Set1 == 6) & (d.W_Set2 == 6) & (d.L_Set1 == 0) & (d.L_Set2 == 0)
    d = d[~wo].copy()
    d['Datum'] = pd.to_datetime(d.Datum, utc=True).dt.tz_localize(None)
    d['Year'] = d.Datum.dt.year
    n = len(d)
    players = pd.concat([d.WinnerId, d.LoserId]).nunique()
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.2), facecolor='white')
    pa, pb, pc_, pd_, pe, pf = axs.ravel()

    # A
    ct = d.groupby(['Year', 'LigaTip']).size().unstack(fill_value=0)
    ct = ct.reindex(range(int(d.Year.min()), int(d.Year.max()) + 1), fill_value=0)
    pa.bar(ct.index, ct.SM, color=C_SM, label='SM (men)')
    pa.bar(ct.index, ct.SF, bottom=ct.SM, color=C_SF, label='SF (women)')
    for x, t in zip(ct.index, ct.SM + ct.SF):
        pa.text(x, t + 25, str(t), ha='center', fontsize=8.5, color='#222222')
    pa.set_xticks(ct.index)
    pa.set_xticklabels([f"{y}" if y != 2017 else '2017*' for y in ct.index], rotation=45)
    pa.set_ylabel('Matches')
    pa.set_ylim(0, (ct.SM + ct.SF).max() * 1.25)
    pa.legend(frameon=False, loc='upper left')
    pa.text(0.99, 0.97, '* from May; 2026 partial (to Nov)', transform=pa.transAxes, ha='right', va='top',
            fontsize=8.5, color='#555555')
    title(pa, 'A  Matches per year')

    # B
    pc = pd.concat([d.WinnerId, d.LoserId]).value_counts()
    bins = np.arange(0, pc.max() + 10, 10)
    pb.hist(pc.values, bins=bins, color=C_SM, edgecolor='white')
    med = pc.median()
    pb.set_yscale('log')
    pb.axvline(med, color='black', ls='--', lw=1)
    pb.text(0.97, 0.95, f'median = {med:.0f}\nmean = {pc.mean():.1f}\nmax = {pc.max()}', transform=pb.transAxes,
            ha='right', va='top', fontsize=10)
    pb.set_xlabel('Matches played by a player')
    pb.set_ylabel('Players (log scale)')
    title(pb, 'B  Matches per player')

    # C
    eb = np.arange(1000, 2000, 25)
    for tp, col in [('SM', C_SM), ('SF', C_SF)]:
        m = d.LigaTip == tp
        x = pd.concat([d.WinnerEloBefore[m], d.LoserEloBefore[m]])
        pc_.hist(x, bins=eb, color=col, alpha=0.6, density=True, label=f'{tp} (n = {len(x):,} player-matches)')
    pc_.axvline(1400, color=C_ELO, ls=':', lw=1)
    pc_.set_ylim(0, pc_.get_ylim()[1] * 1.3)
    pc_.text(1410, pc_.get_ylim()[1] * 0.80, 'start rating 1400', fontsize=9, color='#555555', va='top', ha='left')
    pc_.set_xlabel('Pre-match ELO')
    pc_.set_ylabel('Density')
    pc_.legend(frameon=False, fontsize=9.5, loc='upper left')
    title(pc_, 'C  ELO distribution (pre-match)')

    # D
    three = d.W_Set3.notna()
    stb = three & (d[['W_Set3', 'L_Set3']].max(axis=1) >= 10)
    cats = ['2:0', '2:1\n(full 3rd set)', '2:1\n(match tie-break)']
    w = 0.38
    for i, (tp, col) in enumerate([('SM', C_SM), ('SF', C_SF)]):
        m = d.LigaTip == tp
        vals = [100 * (~three & m).sum() / m.sum(), 100 * (three & ~stb & m).sum() / m.sum(),
                100 * (stb & m).sum() / m.sum()]
        bars = pd_.bar(np.arange(3) + (i - 0.5) * w, vals, w, color=col, label=tp)
        for b_, v in zip(bars, vals):
            pd_.text(b_.get_x() + w / 2, v + 1, f'{v:.0f}%', ha='center', fontsize=9.5)
    pd_.set_xticks(range(3))
    pd_.set_xticklabels(cats)
    pd_.set_ylabel('Share of matches (%)')
    pd_.set_ylim(0, 90)
    pd_.legend(frameon=False, loc='upper right')
    pd_.text(0.98, 0.80, f'Three sets: {100 * three.mean():.0f}% of all matches', transform=pd_.transAxes,
             ha='right', fontsize=10)
    title(pd_, 'D  Set scores (winner perspective)')

    # E
    g = (d.W_Set1 + d.L_Set1 + d.W_Set2 + d.L_Set2)
    gm = g.median()
    gc = g.clip(upper=26)
    vc = gc.value_counts().sort_index().reindex(range(12, 27), fill_value=0)
    pe.bar(vc.index, 100 * vc.values / vc.sum(), color=C_SM, edgecolor='white')
    pe.set_xticks(range(12, 27))
    pe.set_xticklabels([str(i) if i < 26 else '26+' for i in range(12, 27)], fontsize=9.5)
    pe.set_xlabel('Total games in sets 1 + 2')
    pe.set_ylabel('Share of matches (%)')
    pe.text(0.98, 0.95, f'median = {gm:.0f} games', transform=pe.transAxes, ha='right', va='top', fontsize=10)
    title(pe, 'E  Total games in first two sets')

    # F
    diff = (d.WinnerEloBefore - d.LoserEloBefore).values
    keep = diff != 0
    fw = (diff > 0)[keep]
    ads = np.abs(diff)[keep]
    edges = np.array([0, 25, 50, 75, 100, 150, 200, 250, 300, 400, 10000])
    idx = np.digitize(ads, edges) - 1
    xs, ys, lo, hi, ex, cnt = [], [], [], [], [], []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum() < 30:
            continue
        p = fw[m].mean()
        nn = m.sum()
        z = 1.96
        c = (p + z * z / (2 * nn)) / (1 + z * z / nn)
        h = z * np.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / (1 + z * z / nn)
        xs.append(ads[m].mean()); ys.append(p); lo.append(c - h); hi.append(c + h)
        ex.append((1 / (1 + 10 ** (-ads[m] / 400))).mean()); cnt.append(nn)
    xs, ys, lo, hi, ex = map(np.array, (xs, ys, lo, hi, ex))
    gx = np.linspace(0, 500, 100)
    pf.plot(gx, 1 / (1 + 10 ** (-gx / 400)), color=C_ELO, ls='--', lw=1.5, label='ELO expectation')
    pf.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt='o', color=C_SM, capsize=3, lw=1.5,
                label='Observed (95% CI)')
    pf.set_xlabel('|ELO difference| (favourite - underdog), bin mean')
    pf.set_ylabel('Favourite win rate')
    pf.set_ylim(0.45, 1.0)
    pf.legend(frameon=False, loc='lower right', fontsize=9.5)
    pf.text(0.02, 0.97, f'{len(ads):,} matches with unequal ELO', transform=pf.transAxes, va='top', fontsize=9.5,
            color='#555555')
    title(pf, 'F  Raw ELO: favourite win rate by rating gap')

    d0, d1 = d.Datum.min(), d.Datum.max()
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.text(0.5, 0.012,
             f'N = {n:,} matches, {players:,} players, {d0:%Y-%m} – {d1:%Y-%m}, '
             'official singles (SM/SF), W.O. excluded',
             ha='center', fontsize=9, color='#444444')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor='white')
    print(f'N={n} players={players} range={d0:%Y-%m-%d}..{d1:%Y-%m-%d} SM={int((d.LigaTip == "SM").sum())} '
          f'SF={int((d.LigaTip == "SF").sum())} WO_removed={int(wo.sum())}')
    print(f'median/player={med} mean={pc.mean():.1f} three_sets={three.mean():.3f} stb={stb.mean():.3f} med_games={gm}')
    print('F bins', list(zip(np.round(xs), np.round(ys, 3), np.round(ex, 3), cnt)))


if __name__ == '__main__':
    main()
