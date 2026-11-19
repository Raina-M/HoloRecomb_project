"""Step 7. Why the R. rugosa shoulder is not resolved: local P(s) slope against distance scaled by chromosome length, per
chromosome, for R. rugosa (both libraries), R. cephalotes and R. corymbosa (libraries used in Fig. 4c).
Output: results/Suppl_rugosa_shoulder_diagnostic.{pdf,png,svg}"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hicps import load_library, ps_all_pixels, local_slope, RES, STY, FONT, results
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({**FONT, 'font.size': 6, 'axes.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False})
L = pd.read_csv(results('library_summary.tsv'), sep='\t').set_index('library_id'); SP = pd.read_csv(results('species_shoulder.tsv'), sep='\t').set_index('library')
panels = [('rugosa', 'R_rugosa_rep1'), ('rugosa', 'R_rugosa_rep2'), ('cephalotes', SP.loc['cephalotes', 'library_id']), ('corymbosa', SP.loc['corymbosa', 'library_id'])]
fig, axs = plt.subplots(1, 4, figsize=(180 / 25.4, 55 / 25.4), sharey=True)
for ax, (k, lib) in zip(axs, panels):
    Ls = []
    for c in load_library(lib):
        Lc = c['N'] * RES; Ls.append(Lc); x, p = ps_all_pixels([c]); s = local_slope(x, p); m = x < 0.95 * Lc
        ax.plot(x[m] / Lc, s[m], color=STY[k][0], lw=0.6, alpha=0.7)
    ax.axvspan(3e5 / np.median(Ls), 0.6, color='#EEEEEE', zorder=0); ax.set_xscale('log'); ax.set_ylim(-1.6, 0.6); ax.axhline(-1, color='#BDBDBD', lw=0.4)
    ax.set_title(f'R. {k}, library {lib[-1]}\n({min(Ls) / 1e6:.0f}–{max(Ls) / 1e6:.0f} Mb; resolved {L.loc[lib, "n_resolved"]}/{L.loc[lib, "n_chr"]})', fontstyle='italic', fontsize=6)
    ax.set_xlabel('Genomic distance / chromosome length')
axs[0].set_ylabel('Local slope of P(s), all pixel pairs')
fig.text(0.5, 0.005, 'Grey: interval searched for the shoulder (300 kb to 0.6 x chromosome length). In R. rugosa the slope rises without a trough into the chromosome-end increase.', ha='center', fontsize=5.5)
fig.subplots_adjust(bottom=0.25, top=0.82, wspace=0.08)
for e in ('pdf', 'png', 'svg'): fig.savefig(results(f'Suppl_rugosa_shoulder_diagnostic.{e}'), dpi=300 if e == 'png' else None)
