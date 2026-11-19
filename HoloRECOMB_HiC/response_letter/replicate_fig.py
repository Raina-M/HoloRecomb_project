"""Response letter only. Response Fig. [HiC replicates]: reproducibility of somatic Hi-C contact decay between two Arima
libraries from leaf tissue. Run after analysis/01-04. Output: results/response_letter/Response_Fig_HiC_replicates.{pdf,svg,png}"""
import os, sys, re, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from hicps import load_library, ps_all_pixels, local_slope, RES, STY, LAB, FONT, results
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
plt.rcParams.update({**FONT, 'font.size': 6.5, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
OUT = results('response_letter'); os.makedirs(OUT, exist_ok=True)
R = pd.read_csv(results('replicate_summary.tsv'), sep='\t'); F = pd.read_csv(results('replicate_fig4c_stats.tsv'), sep='\t')
SP = pd.read_csv(results('species_shoulder.tsv'), sep='\t').set_index('library'); txt = open(results('replicate_stats.txt')).read()
SPP = list(R.sort_values('species').species.unique()); g = lambda s, r, k: R[(R.species == s) & (R.replicate == r)][k].iloc[0]
fmt = FuncFormatter(lambda v, _: f'{v:g}')
fig = plt.figure(figsize=(180 / 25.4, 200 / 25.4))
gs = fig.add_gridspec(3, 4, left=0.07, right=0.985, top=0.965, bottom=0.55, hspace=0.6, wspace=0.32)
for i, s in enumerate(SPP):
    ax = fig.add_subplot(gs[i // 4, i % 4]); c = STY[s][0]
    for r, ls, lw, cc in (('rep1', '-', 1.0, c), ('rep2', '--', 0.9, '#333333')):
        ch = load_library(g(s, r, 'library_id')); x, p = ps_all_pixels(ch); sl = local_slope(x, p); Lm = np.mean([q['N'] * RES for q in ch]); m = x < 0.4 * Lm
        ax.plot(x[m], sl[m], ls=ls, lw=lw, color=cc, label=f'library {r[-1]}')
        if g(s, r, 'n_resolved') / g(s, r, 'n_chr') >= 0.5: ax.axvline(g(s, r, 'shoulder_median_Mb') * 1e6, color=cc, ls=':', lw=0.6)
    ax.axhline(-1, color='#BBBBBB', lw=0.5, zorder=0); ax.set_xscale('log'); ax.set_xlim(1e4, 1e8); ax.set_ylim(-1.6, -0.2)
    ttl = LAB(s) + (' (library 2 failed)' if (R[R.species == s].status != 'ok').any() else '')
    ax.set_title(ttl, fontstyle='italic', fontsize=6.3, pad=2, color=c)
    sh = lambda r: f"{g(s, r, 'shoulder_median_Mb'):.1f}" if g(s, r, 'n_resolved') / g(s, r, 'n_chr') >= 0.5 else 'nr'
    ax.text(0.03, 0.04, f"> 20 kb: {g(s, 'rep1', 'contacts_gt20kb_M'):.2g} / {g(s, 'rep2', 'contacts_gt20kb_M'):.2g} M\nshoulder: {sh('rep1')} / {sh('rep2')} Mb",
            transform=ax.transAxes, fontsize=5, va='bottom')
    ax.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8]); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: {1e4: '0.01', 1e5: '0.1', 1e6: '1', 1e7: '10', 1e8: '100'}.get(v, '')))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if i % 4 == 0: ax.set_ylabel('Local slope of P(s)')
    if i // 4 == 2: ax.set_xlabel('Genomic distance (Mbp)')
    if i == 0: ax.legend(frameon=False, fontsize=5, loc='upper left', handlelength=2)
fig.text(0.01, 0.975, 'a', fontsize=9, fontweight='bold')
# b
axb = fig.add_axes([0.07, 0.06, 0.24, 0.38])
for s in SPP:
    if (R[R.species == s].n_resolved / R[R.species == s].n_chr < 0.5).any(): continue
    x, y = g(s, 'rep1', 'shoulder_median_Mb'), g(s, 'rep2', 'shoulder_median_Mb'); c, m = STY[s]; fail = (R[R.species == s].status != 'ok').any()
    axb.errorbar(x, y, xerr=[[x - g(s, 'rep1', 'shoulder_min_Mb')], [g(s, 'rep1', 'shoulder_max_Mb') - x]], yerr=[[y - g(s, 'rep2', 'shoulder_min_Mb')], [g(s, 'rep2', 'shoulder_max_Mb') - y]],
                 fmt='none', ecolor=c, elinewidth=0.5, alpha=0.5)
    axb.scatter(x, y, s=26, marker=m, facecolor='white' if fail else c, edgecolor=c, linewidth=0.8 if fail else 0.3, zorder=3)
axb.plot([0.8, 9], [0.8, 9], ls=':', color='#888888', lw=0.7); axb.set_xscale('log'); axb.set_yscale('log'); axb.set_xlim(0.8, 9); axb.set_ylim(0.8, 9)
for a in (axb.xaxis, axb.yaxis): a.set_major_formatter(fmt); a.set_minor_formatter(NullFormatter())
axb.set_xticks([1, 2, 4, 8]); axb.set_yticks([1, 2, 4, 8]); axb.set_xlabel('Shoulder, library 1 (Mbp)'); axb.set_ylabel('Shoulder, library 2 (Mbp)')
n, rho, P = re.findall(r'\((\d+) species\): Spearman rho = ([0-9.]+), P = ([0-9.e-]+)', txt)[0]; fm, fx = re.findall(r'fold difference median ([0-9.]+), max ([0-9.]+)', txt)[0]
axb.text(0.04, 0.97, f'{n} species (open: R. ridleyi, library 2 failed;\nnot included)\nSpearman ρ = {float(rho):.2f}, $p$ < 0.001\nmedian fold difference {float(fm):.1f} (max {float(fx):.1f})',
         transform=axb.transAxes, va='top', fontsize=5.2)
fig.text(0.01, 0.465, 'b', fontsize=9, fontweight='bold')
# c
axc = fig.add_axes([0.42, 0.06, 0.20, 0.38]); y = np.arange(len(SPP))[::-1]
for yi, s in zip(y, SPP):
    r = R[R.species == s].iloc[0]; axc.barh(yi, r.scc_median, color=STY[s][0], height=0.65); axc.plot([r.scc_min, r.scc_max], [yi, yi], color='black', lw=0.6)
    d = R[R.species == s].contacts_gt20kb_M.min(); axc.text(1.02, yi, f'{d:.2g}', va='center', fontsize=5)
axc.set_yticks(y); axc.set_yticklabels([LAB(s) for s in SPP], fontstyle='italic'); axc.set_xlim(0, 1.0)
axc.set_xlabel('SCC, library 1 vs 2\n(100 kbp, 0.1-5 Mbp)'); axc.text(1.02, len(SPP) - 0.3, 'shallower\nlibrary,\nM > 20 kb', fontsize=4.8, va='bottom')
fig.text(0.31, 0.465, 'c', fontsize=9, fontweight='bold')
# d
L = pd.read_csv(results('library_summary.tsv'), sep='\t')
axd = fig.add_axes([0.76, 0.06, 0.225, 0.38])
for k, r in SP.iterrows():
    if not r.resolved: continue
    c, m = STY[k]; axd.scatter(r.mean_chr_Mb, r.shoulder_median_Mb, s=18, marker=m, color=c, edgecolor='white', linewidth=0.3, zorder=3)
    q = L[(L.species == k) & ~L.selected & (L.status == 'ok')]
    if len(q) and q.n_resolved.iloc[0] / q.n_chr.iloc[0] >= 0.5:
        y2 = q.shoulder_median_Mb.iloc[0]; axd.plot([r.mean_chr_Mb] * 2, [r.shoulder_median_Mb, y2], color=c, lw=0.6)
        axd.scatter(r.mean_chr_Mb, y2, s=18, marker=m, facecolor='white', edgecolor=c, linewidth=0.8, zorder=4)
axd.set_xscale('log'); axd.set_yscale('log'); axd.set_xlim(10, 450); axd.set_ylim(1, 16)
for a in (axd.xaxis, axd.yaxis): a.set_major_formatter(fmt); a.set_minor_formatter(NullFormatter())
axd.set_xticks([10, 20, 50, 100, 200, 400]); axd.set_yticks([1, 2, 4, 8]); axd.set_xlabel('Mean chromosome size (Mbp)'); axd.set_ylabel('P(s) shoulder (Mbp)')
a0, a1 = F.iloc[0], F.iloc[1]; pp = lambda p: '< 0.001' if p < 0.001 else f'= {p:.3f}' if p < 0.01 else f'= {p:.2f}'
axd.text(0.03, 0.97, f"filled: Fig. 4c (deeper library)\nopen: other library\nρ = {a0.rho:.2f} / {a1.rho:.2f}\nmixed model $p$ {pp(a0.P_mm)} / {pp(a1.P_mm)}\n"
         f"within species $p$ = {a0.P_within:.2f} / {a1.P_within:.2f}", transform=axd.transAxes, va='top', fontsize=5.2)
fig.text(0.68, 0.465, 'd', fontsize=9, fontweight='bold')
for e in ('pdf', 'svg', 'png'): fig.savefig(os.path.join(OUT, f'Response_Fig_HiC_replicates.{e}'), dpi=300)
