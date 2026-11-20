"""Step 6. Supplementary Fig. 12
 (a) local slope of P(s), observed and featureless null, five species; (b) five R. breviuscula libraries (two Arima Hi-C,
 three Omni-C); (c) per-chromosome shoulder against chromosome size; (d) shoulder of the two libraries of each species;
 (e) anther Hi-C enrichment over random ligation; (f, g) anther (background-corrected) and somatic local slope.
Inputs : results of steps 1, 2 and 4 and of anther/anther_hic.py; data/somatic_extra (Omni-C libraries of R. breviuscula)
Output : results/Suppl_Fig_HiC.{pdf,png,svg}"""
import os, sys, re, pickle, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); warnings.filterwarnings('ignore')
from hicps import ROOT, RES, STY, LAB, ORDER, FONT, load_library, ps_all_pixels, local_slope, null_chroms, results
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
plt.rcParams.update({**FONT, 'font.size': 6, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2,
                     'ytick.major.size': 2, 'xtick.minor.size': 1, 'axes.spines.top': False, 'axes.spines.right': False})
SP = pd.read_csv(results('species_shoulder.tsv'), sep='\t').set_index('library'); CH = pd.read_csv(results('chromosome_shoulder.tsv'), sep='\t')
REP = pd.read_csv(results('replicate_summary.tsv'), sep='\t')
NR = list(SP.index[~SP.resolved]); LOW = list(SP.index[SP.low_depth & SP.resolved]); OPEN = set(NR + LOW)
EXTRA = os.path.join(ROOT, 'data', 'somatic_extra')

def fmt(ax):
    f = FuncFormatter(lambda v, _: f'{v:g}')
    for a in (ax.xaxis, ax.yaxis):
        if a.get_scale() == 'log': a.set_major_formatter(f); a.set_minor_formatter(NullFormatter())
cache = {}
def curve(lib, data=None):
    if lib not in cache:
        ch = load_library(lib, data) if data else load_library(lib); x, p = ps_all_pixels(ch); cache[lib] = (x, p, local_slope(x, p), np.mean([c['N'] * RES for c in ch]), ch)
    return cache[lib]

fig = plt.figure(figsize=(180 / 25.4, 175 / 25.4)); Lt = lambda x, y, t: fig.text(x, y, t, fontsize=8, fontweight='bold')
# a
ax = fig.add_axes([0.07, 0.726, 0.40, 0.247])
for i, k in enumerate(['cephalotes', 'corymbosa', 'radicans', 'nervosa', 'rugosa']):
    x, p, s, Lm, ch = curve(SP.loc[k, 'library_id']); m = x < 0.4 * Lm; c = STY[k][0] if k != 'radicans' else '#1F78B4'
    ax.plot(x[m], s[m], color=c, lw=1, label=LAB(k))
    xn, pn = ps_all_pixels(null_chroms(ch, np.random.default_rng(7 + i))); sn = local_slope(xn, pn); mn = xn < 0.4 * Lm
    ax.plot(xn[mn], sn[mn], color=c, lw=0.5, ls='--')
ax.set_xscale('log'); ax.set_xlim(1e4, 4e7); ax.set_ylim(-1.6, -0.2); ax.axhline(-1, color='#BDBDBD', lw=0.4)
ax.set_xlabel('Genomic distance s (bp)'); ax.set_ylabel('Local slope of P(s), all pixel pairs')
leg = ax.legend(frameon=False, fontsize=5, loc='lower right', ncol=2); [t.set_fontstyle('italic') for t in leg.get_texts()]
ax.text(0.02, 0.98, 'solid: observed; dashed: featureless null', transform=ax.transAxes, va='top', fontsize=5); Lt(0.005, 0.985, 'a')
# b
ax = fig.add_axes([0.58, 0.726, 0.40, 0.247])
B = [('R_breviuscula_rep2', None, 'Arima Hi-C, library 2', '#1B9E77'), ('R_breviuscula_rep1', None, 'Arima Hi-C, library 1', '#66C2A5'),
     ('Rhynchospora_breviuscula_OmniC_4858A', EXTRA, 'Omni-C 4858A', '#A6D854'), ('Rhynchospora_breviuscula_OmniC_5122A', EXTRA, 'Omni-C 5122A', '#238B45'),
     ('Rhynchospora_breviuscula_OmniC_flowerBuds', EXTRA, 'Omni-C flower buds', '#7FBC41')]
for lib, d, lab, c in B:
    x, p, s, Lm, _ = curve(lib, d); m = x < 0.4 * Lm; ax.plot(x[m], s[m], color=c, lw=1, label=lab)
ax.set_xscale('log'); ax.set_xlim(1e4, 4e7); ax.set_ylim(-1.6, -0.2); ax.axhline(-1, color='#BDBDBD', lw=0.4)
ax.set_xlabel('Genomic distance s (bp)'); ax.set_ylabel('Local slope of P(s), all pixel pairs')
ax.legend(frameon=False, fontsize=5, loc='lower right', title='$\\it{R.\\ breviuscula}$ libraries', title_fontsize=5); Lt(0.51, 0.985, 'b')
# c
ax = fig.add_axes([0.07, 0.376, 0.52, 0.247]); KS = [k for k in SP.index if k not in NR]
cm = CH[CH.library.isin(KS) & CH.shoulder_Mb.notna()]
for k in KS:
    g = cm[cm.library == k]
    if k in OPEN: ax.scatter(g.chr_Mb, g.shoulder_Mb, s=9, marker=STY[k][1], facecolor='white', edgecolor=STY[k][0], linewidth=0.6)
    else: ax.scatter(g.chr_Mb, g.shoulder_Mb, s=9, marker=STY[k][1], color=STY[k][0], edgecolor='none', alpha=0.85)
import json; st = json.load(open(results('fig4c_stats.json')))['all']
xs = np.logspace(np.log10(cm.chr_Mb.min()), np.log10(cm.chr_Mb.max()), 20); ax.plot(xs, 10 ** (st['mm_intercept'] + st['mm_slope'] * np.log10(xs)), 'k-', lw=0.8)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Chromosome size (Mbp)'); ax.set_ylabel('P(s) shoulder position (Mbp)')
ax.set_yticks([0.5, 1, 2, 4, 8]); ax.set_xticks([10, 20, 50, 100, 200, 400]); fmt(ax)
pp = lambda p: '< 0.001' if p < 0.001 else f'= {p:.2f}'
ax.text(0.02, 0.98, f'{st["n_chr"]} chromosomes, {st["n_species"]} species\nmixed model slope = {st["mm_slope"]:.2f} ± {st["mm_se"]:.2f}, P {pp(st["P_mm"])}\n'
        f'within species slope = {st["within"]:.2f} ± {st["within_se"]:.2f}, P {pp(st["P_within"])}', transform=ax.transAxes, va='top', fontsize=5)
h = [Line2D([], [], marker=STY[k][1], ls='', markerfacecolor='white' if k in OPEN else STY[k][0], markeredgecolor=STY[k][0], markersize=3.5, label=LAB(k)) for k in KS]
leg = fig.legend(handles=h, loc='center', bbox_to_anchor=(0.675, 0.50), frameon=False, fontsize=4.8, handletextpad=0.1, labelspacing=0.12)
for t in leg.get_texts(): t.set_fontstyle('italic')
Lt(0.005, 0.64, 'c')
# d
ax = fig.add_axes([0.80, 0.376, 0.18, 0.247]); txt = open(results('replicate_stats.txt')).read()
res_both = [l for l in txt.split('\n') if 'shoulder resolved in both' in l][0]
for sp, g in REP.groupby('species'):
    if (g.status != 'ok').any() or (g.n_resolved / g.n_chr < 0.5).any(): continue
    a = g[g.replicate == 'rep1'].iloc[0]; b = g[g.replicate == 'rep2'].iloc[0]; c_, m_ = STY[sp]
    ax.errorbar(a.shoulder_median_Mb, b.shoulder_median_Mb, xerr=[[a.shoulder_median_Mb - a.shoulder_min_Mb], [a.shoulder_max_Mb - a.shoulder_median_Mb]],
                yerr=[[b.shoulder_median_Mb - b.shoulder_min_Mb], [b.shoulder_max_Mb - b.shoulder_median_Mb]], fmt='none', ecolor=c_, elinewidth=0.5, alpha=0.5)
    ax.scatter(a.shoulder_median_Mb, b.shoulder_median_Mb, s=16, marker=m_, color=c_, edgecolor='white', linewidth=0.3, zorder=3)
ax.plot([0.8, 9], [0.8, 9], ls=':', color='#888888', lw=0.6); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.8, 9); ax.set_ylim(0.8, 9)
ax.set_xticks([1, 2, 4, 8]); ax.set_yticks([1, 2, 4, 8]); fmt(ax)
ax.set_xlabel('Shoulder, library 1 (Mbp)'); ax.set_ylabel('Shoulder, library 2 (Mbp)')
n, rho, P = re.findall(r'\((\d+) species\): Spearman rho = ([0-9.]+), P = ([0-9.e-]+)', txt)[0]
ax.text(0.04, 0.97, f'{n} species\nρ = {float(rho):.2f}, P {pp(float(P))}', transform=ax.transAxes, va='top', fontsize=5); Lt(0.75, 0.64, 'd')
# e-g anther
AD = results('anther') + '/'; t = open(AD + 'anther_stats.txt').read(); RR = {}
for blk in t.split('== ')[1:]: RR[blk.split(':')[0]] = [float(v) for v in re.findall(r'ratio ([0-9.]+)', blk)]
AC = {'R_cephalotes': '#D95F02', 'R_corymbosa': '#7570B3', 'R_radicans': '#E6AB02'}
ax = fig.add_axes([0.085, 0.06, 0.225, 0.22]); w = 0.26
for i, sp in enumerate(AC): ax.bar(np.arange(3) + (i - 1) * w, RR[sp], w, color=AC[sp], label='R. ' + sp[2:])
ax.axhline(1, color='k', lw=0.5, ls='--'); ax.set_yscale('log'); fmt(ax); ax.set_yticks([1, 10, 100]); ax.set_xticks(range(3)); ax.set_xticklabels(['20–100 kb', '0.1–1 Mb', '1–10 Mb'], fontsize=5.5)
ax.set_ylabel('Observed / random ligation\n(anther, intrachromosomal)'); ax.set_xlabel('Genomic distance')
leg = ax.legend(frameon=False, fontsize=5, loc='upper right'); [t_.set_fontstyle('italic') for t_ in leg.get_texts()]; Lt(0.005, 0.30, 'e')
cur = pickle.load(open(AD + 'anther_curves.pkl', 'rb'))
for j, sp in enumerate(('R_corymbosa', 'R_radicans')):
    ax = fig.add_axes([0.42 + j * 0.30, 0.06, 0.25, 0.22]); x, p, s_, Lm, _ = curve(SP.loc[sp[2:], 'library_id']); m = x < 0.4 * Lm
    ax.plot(x[m], s_[m], color='k', lw=0.9, label='somatic (leaf)')
    x, p = cur[(sp, True)]; s2 = local_slope(x, p); m = x < 0.4 * Lm; ax.plot(x[m], s2[m], color=AC[sp], lw=0.9, label='anther, background-corrected')
    ax.set_xscale('log'); ax.set_xlim(1e4, 3e7); ax.set_ylim(-2, 0.2); ax.axhline(-1, color='#BDBDBD', lw=0.4)
    ax.set_xlabel('Genomic distance s (bp)'); ax.set_ylabel('Local slope of P(s)'); ax.set_title('R. ' + sp[2:], fontstyle='italic', fontsize=6.5, pad=2)
    ax.legend(frameon=False, fontsize=5, loc='lower left'); Lt(0.355 + j * 0.30, 0.30, 'fg'[j])
for e in ('pdf', 'png', 'svg'): fig.savefig(results(f'Suppl_Fig_HiC.{e}'), dpi=300 if e == 'png' else None)
