"""Step 8. Fig. 4 (a-e).
a  ZYP1 axis length per Mb (per nucleus) against mean chromosome size
b  double-crossover frequency against proportional axis length (per chromosome), mixed model
c  P(s) shoulder against chromosome size (as Fig4c_Ps_shoulder)
d  local P(s) slope and contact maps with Tyba arrays for three species
e  simulated against measured axis length per Mb (polymer model, companion study)
"""
import os, sys, json, numpy as np, pandas as pd, scipy.stats as ss, statsmodels.formula.api as smf, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter, FixedLocator
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hicps import ROOT, DATA, load_library, ps_all_pixels, local_slope, RES, STY, LAB, ORDER, FONT, results
IN = os.path.join(ROOT, 'data', 'fig4_inputs')     # axis lengths, DCO and simulation source data, micrographs
OUT = results()
plt.rcParams.update({**FONT, 'font.size': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5, 'pdf.fonttype': 42, 
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
MM = 1 / 25.4; W, H = 180, 176
fig = plt.figure(figsize=(W * MM, H * MM))
def ax_mm(x, y, w, h, **kw):          # x, y from top-left, in mm
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)
def label(x, y, t):
    fig.text(x / W, 1 - y / H, t, fontsize=9, fontweight='bold', va='top', ha='left')
pf = lambda p: '< 0.001' if p < 0.001 else f'= {p:.3f}' if p < 0.01 else f'= {p:.2f}'
gfmt = FuncFormatter(lambda v, _: f'{v:g}')
def k(sp): return sp.replace('R. ', '')
src = {}

# ---------------------------------------------------------------- a: axis per Mb
axl = pd.read_csv(os.path.join(IN, 'axis_lengths.txt'), sep='\t')
chl = pd.read_csv(os.path.join(IN, 'chrs_lengths.txt'), sep='\t')
chl['hap_Mb'] = (chl.h1_total + chl.h2_total) / 2 / 1e6; chl['chr_Mb'] = chl.mean_chr_length / 1e6
a = axl.merge(chl[['species', 'hap_Mb', 'chr_Mb']], on='species'); a['axis_per_Mb'] = a.total_axis_length / a.hap_Mb
sa = a.groupby('species').agg(chr_Mb=('chr_Mb', 'first'), hap_Mb=('hap_Mb', 'first'), med=('axis_per_Mb', 'median'),
                              q1=('axis_per_Mb', lambda v: v.quantile(.25)), q3=('axis_per_Mb', lambda v: v.quantile(.75)),
                              mean=('axis_per_Mb', 'mean'), n=('axis_per_Mb', 'size'))
ra, pa = ss.spearmanr(sa.chr_Mb, sa.med); oa = smf.ols('np.log10(med)~np.log10(chr_Mb)', sa).fit()
src['a'] = a; src['a_species'] = sa
# micrographs
for i, (nm, ttl) in enumerate((('raw', 'ZYP1 (white), DAPI (blue)'), ('skel', 'Skeletonised ZYP1'))):
    axi = ax_mm(6, 6 + i * 27.5, 25, 25); axi.imshow(plt.imread(os.path.join(IN, f'micro_{nm}.png'))); axi.set_axis_off()
    axi.text(0.5, -0.02, ttl, transform=axi.transAxes, ha='center', va='top', fontsize=5.5)
fig.text(18.5 / W, 1 - 2.2 / H, 'R. breviuscula, pachytene', fontsize=5.5, ha='center', va='top', fontstyle='italic')
ax = ax_mm(41, 5, 42, 46)
rng = np.random.default_rng(3)
for sp, g in a.groupby('species'):
    c, m = STY[k(sp)]; x = g.chr_Mb.iloc[0] * 10 ** rng.uniform(-0.035, 0.035, len(g))
    ax.scatter(x, g.axis_per_Mb, s=6, color=c, alpha=0.45, linewidth=0, marker=m, zorder=2)
    r = sa.loc[sp]; ax.errorbar(r.chr_Mb, r.med, yerr=[[r.med - r.q1], [r.q3 - r.med]], fmt='none', ecolor='k', elinewidth=0.6, capsize=1.2, zorder=3)
    ax.scatter(r.chr_Mb, r.med, s=30, color=c, marker=m, edgecolor='k', linewidth=0.5, zorder=4)
xs = np.logspace(np.log10(30), np.log10(100), 30); ax.plot(xs, 10 ** oa.predict(pd.DataFrame(dict(chr_Mb=xs))), color='#4D4D4D', ls='--', lw=1.0, zorder=1)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(28, 110); ax.set_ylim(0.09, 0.45)
ax.xaxis.set_major_locator(FixedLocator([30, 50, 100])); ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.3, 0.4]))
for A in (ax.xaxis, ax.yaxis): A.set_major_formatter(gfmt); A.set_minor_formatter(NullFormatter())
ax.set_xlabel('Mean chromosome size (Mbp)'); ax.set_ylabel('Axis length per Mbp (µm Mbp$^{-1}$)')
ax.text(0.97, 0.97, f'6 species, {len(a)} nuclei\nlog-log fit of species medians: $p$ {pf(oa.pvalues.iloc[1])}\nSpearman ρ = {ra:.2f}, $p$ {pf(pa)}', transform=ax.transAxes, ha='right', va='top', fontsize=5.8)
label(1, 2, 'a')

# ---------------------------------------------------------------- b: DCO vs axis
b = pd.read_csv(os.path.join(IN, 'panel_b_DCO_axis_source.tsv'), sep='\t')
mb = smf.mixedlm('dco_pct~axis_um', b, groups=b.species).fit(reml=True)
m1 = smf.mixedlm('dco_pct~axis_um', b, groups=b.species).fit(reml=False); m2 = smf.mixedlm('dco_pct~chr_Mb', b, groups=b.species).fit(reml=False)
ax = ax_mm(98, 5, 42, 46)
for sp, g in b.groupby('species'):
    c, m = STY[k(sp)]; ax.scatter(g.axis_um, g.dco_pct, s=16, color=c, marker=m, edgecolor='white', linewidth=0.3, zorder=3)
xs = np.linspace(b.axis_um.min(), b.axis_um.max(), 20); ax.plot(xs, mb.params.iloc[0] + mb.params.iloc[1] * xs, color='#4D4D4D', ls='--', lw=1.0)
ax.set_xlabel('Proportional axis length (µm)'); ax.set_ylabel('Double crossovers (%)'); ax.set_ylim(0, 6.3); ax.set_xlim(8.5, 23); ax.set_yticks(range(0, 6))
ax.text(0.03, 0.97, f'5 species, {len(b)} chromosomes\nmixed model slope = {mb.params.iloc[1]:.2f} ± {mb.bse.iloc[1]:.2f}, $p$ {pf(mb.pvalues.iloc[1])}\n'
        f'ΔAIC, axis vs chromosome length = {m2.aic - m1.aic:.1f}', transform=ax.transAxes, ha='left', va='top', fontsize=5.8)
label(88, 2, 'b')
src['b'] = b

# shared legend for a, b, e (species with cytology)
CY = ['cephalotes', 'corymbosa', 'breviuscula', 'colorata', 'nervosa', 'ciliata']
h = [Line2D([], [], marker=STY[s][1], ls='', color=STY[s][0], markersize=4, label=LAB(s)) for s in CY]
leg = fig.legend(handles=h, loc='upper left', bbox_to_anchor=(145 / W, 1 - 8 / H), fontsize=6, handletextpad=0.2, labelspacing=0.45, title='a, b, e', title_fontsize=6)
for t in leg.get_texts(): t.set_fontstyle('italic')
leg._legend_box.align = 'left'

# ---------------------------------------------------------------- c: shoulder vs chromosome size (as Fig4c_Ps_shoulder)
sp4 = pd.read_csv(results('species_shoulder.tsv'), sep='\t').set_index('library'); cres = pd.read_csv(results('chromosome_shoulder.tsv'), sep='\t')
NR = [s for s in sp4.index if not sp4.loc[s, 'resolved']]; LOW = [s for s in sp4.index if sp4.loc[s, 'low_depth'] and s not in NR]; KS = [s for s in sp4.index if s not in NR]
d4 = sp4.loc[KS]; cm = cres[cres.library.isin(KS) & cres.shoulder_Mb.notna()]
mm = smf.mixedlm('np.log10(shoulder_Mb)~np.log10(chr_Mb)', cm, groups=cm.library).fit(reml=True)
rc, pc = ss.spearmanr(d4.mean_chr_Mb, d4.shoulder_median_Mb)
ax = ax_mm(13, 65, 52, 50)
for s, r in d4.iterrows():
    c, m = STY[s]; ax.errorbar(r.mean_chr_Mb, r.shoulder_median_Mb, yerr=[[max(r.shoulder_median_Mb - r.shoulder_min_Mb, 0)], [max(r.shoulder_max_Mb - r.shoulder_median_Mb, 0)]],
                               fmt='none', ecolor=c, elinewidth=0.6, alpha=0.6)
    if s in LOW: ax.scatter(r.mean_chr_Mb, r.shoulder_median_Mb, marker=m, s=22, facecolor='white', edgecolor=c, linewidth=0.9, zorder=3)
    else: ax.scatter(r.mean_chr_Mb, r.shoulder_median_Mb, marker=m, s=22, color=c, edgecolor='white', linewidth=0.3, zorder=3)
xs = np.logspace(np.log10(d4.mean_chr_Mb.min()), np.log10(d4.mean_chr_Mb.max()), 50)
ax.plot(xs, 10 ** (mm.params.iloc[0] + mm.params.iloc[1] * np.log10(xs)), color='#4D4D4D', ls='--', lw=1.0)
for s in NR:
    ax.scatter(sp4.loc[s, 'mean_chr_Mb'], 0.95, marker=STY[s][1], s=22, facecolor='white', edgecolor=STY[s][0], linewidth=0.8, zorder=3, clip_on=False)
    ax.annotate('not resolved', (sp4.loc[s, 'mean_chr_Mb'], 0.95), xytext=(4, 0), textcoords='offset points', fontsize=5.5, va='center')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.88, 9); ax.set_xlim(10, 450)
ax.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 400])); ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 6, 8]))
for A in (ax.xaxis, ax.yaxis): A.set_major_formatter(gfmt); A.set_minor_formatter(NullFormatter())
ax.set_xlabel('Mean chromosome size (Mbp)'); ax.set_ylabel('P(s) shoulder position (Mbp)')
ax.text(0.03, 0.97, f'{len(d4)} species: Spearman ρ = {rc:.2f}, $p$ {pf(pc)}\n{len(cm)} chromosomes: mixed model $p$ {pf(mm.pvalues.iloc[1])}', transform=ax.transAxes, ha='left', va='top', fontsize=5.8, bbox=dict(fc='white', ec='none', alpha=0.85, pad=0.5))
h = [Line2D([], [], marker=STY[s][1], ls='', color=STY[s][0], markersize=3.8, label=LAB(s)) if s not in NR + LOW else
     Line2D([], [], marker=STY[s][1], ls='', markerfacecolor='white', markeredgecolor=STY[s][0], markersize=3.8, label=LAB(s)) for s in sp4.index]
leg = fig.legend(handles=h, loc='upper left', bbox_to_anchor=(66 / W, 1 - 62 / H), fontsize=5.2, handletextpad=0.1, labelspacing=0.12, ncol=1)
for t in leg.get_texts(): t.set_fontstyle('italic')
label(1, 62, 'c')

# ---------------------------------------------------------------- d: slope curves and maps
AGG = {}          # all maps at 100 kbp
# (species, Hi-C chromosome = Tyba sequence name, Tyba annotation file prefix); library = the one used in Fig. 4c
D3 = [('cephalotes', 'chr1_h1', 'Rhync_cephalotes.hap1'),
      ('nervosa', 'HiC_scaffold_1_h1', 'Rhync_nervosa_6321B.asm.hic.hap1'),
      ('tenuis', 'Chr1_h2', 'Rhync_tenuis.hap2')]
tyba = pd.read_csv(os.path.join(ROOT, 'data', 'tyba', 'tyba_arrays_merged10kb.tsv'), sep='\t')
ax = ax_mm(103, 65, 55, 42)
for s, _, _ in D3:
    ch = load_library(sp4.loc[s, 'library_id']); x, p = ps_all_pixels(ch); sl = local_slope(x, p); Lm = np.mean([c['N'] * RES for c in ch]); kk = x < 0.4 * Lm
    c = STY[s][0]; ax.plot(x[kk] / 1e6, sl[kk], color=c, lw=1.2, label=f'{LAB(s)} ({Lm / 1e6:.0f} Mbp)')
    sh = sp4.loc[s, 'shoulder_median_Mb']; ax.axvline(sh, color=c, ls=':', lw=0.8)
    ax.text(sh * {'cephalotes': 0.93, 'nervosa': 0.93, 'tenuis': 1.08}[s], {'cephalotes': 0.1, 'nervosa': -0.06, 'tenuis': 0.02}[s], f'{sh:.1f}', color=c, fontsize=5.8, ha='right' if s != 'tenuis' else 'left', va='bottom', fontweight='bold')
    src[f'd_slope_{s}'] = pd.DataFrame(dict(distance_Mb=x[kk] / 1e6, local_slope=sl[kk]))
ax.axhline(-1, color='#BDBDBD', lw=0.5, zorder=0)
ax.set_xscale('log'); ax.set_xlim(0.01, 40); ax.set_ylim(-1.5, 0.2)
ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10])); ax.xaxis.set_major_formatter(gfmt); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('Genomic distance s (Mbp)'); ax.set_ylabel('Local slope of P(s)')
lg = ax.legend(loc='upper left', fontsize=5.6, handlelength=1.5, title='species (mean chromosome size)', title_fontsize=5.4); lg._legend_box.align = 'left'
for t in lg.get_texts(): t.set_fontstyle('italic')
ax.text(0.99, 1.0, 'dotted: P(s) shoulder (Mbp)', transform=ax.transAxes, ha='right', va='bottom', fontsize=5.4, color='#4D4D4D')
label(95, 62, 'd')

# maps: chromosome 1, 20-Mbp window, 100-kbp bins, distance up to 10 Mbp
WIN0, WIN = 15e6, 20e6; BIN = 1e5
for j, (s, seq, tf) in enumerate(D3):
    z = np.load(os.path.join(DATA, sp4.loc[s, 'library_id'], seq + '.npz')); M = z['M'].astype(float); M = np.triu(M) + np.triu(M, 1).T
    ag = AGG.get(s, 1); BIN = 1e5 * ag
    if ag > 1: m_ = (len(M) // ag) * ag; M = M[:m_, :m_].reshape(m_ // ag, ag, m_ // ag, ag).sum(axis=(1, 3))
    i0 = int(WIN0 / BIN); n = int(WIN / BIN); S = M[i0:i0 + n, i0:i0 + n]; S = np.triu(S) + np.triu(S, 1).T
    dd = np.abs(np.subtract.outer(np.arange(n), np.arange(n))); ref = np.nanmean(S[dd == int(2e5 / BIN)]); S = S / ref
    e = np.arange(n + 1) * BIN / 1e6 + WIN0 / 1e6; Ei, Ej = np.meshgrid(e, e, indexing='ij')
    X = (Ei + Ej) / 2; Y = (Ej - Ei); C = np.ma.masked_where(np.tril(np.ones((n, n)), -1).astype(bool) | (S <= 0), S)
    axm = ax_mm(3 + j * 38.5, 129, 34, 21)
    pm = axm.pcolormesh(X, Y, C, cmap='RdYlBu_r', norm=LogNorm(vmin=10 ** -2.0, vmax=10 ** 0.5), shading='flat', rasterized=True)
    axm.set_xlim(WIN0 / 1e6, (WIN0 + WIN) / 1e6); axm.set_ylim(0, 10); axm.set_aspect('auto')
    sh = sp4.loc[s, 'shoulder_median_Mb']; axm.axhline(sh, color='k', ls='--', lw=0.6)
    axm.text((WIN0 + WIN) / 1e6 - 0.3, sh + 0.25, f'{sh:.1f} Mbp', fontsize=5.4, ha='right', va='bottom')
    for sp_ in ('top', 'right', 'left'): axm.spines[sp_].set_visible(False)
    axm.set_yticks([]); axm.set_xticks([])
    axm.set_title(f'{LAB(s)}', fontstyle='italic', fontsize=6.3, pad=1.5, color=STY[s][0])
    # Tyba track
    t = tyba[tyba.file.str.startswith(tf) & (tyba.seqID == seq)]; tw = t[(t.end > WIN0) & (t.start < WIN0 + WIN)]
    axt = ax_mm(3 + j * 38.5, 151, 34, 3.5)
    for r in tw.itertuples(): axt.axvspan(max(r.start, WIN0) / 1e6, min(r.end, WIN0 + WIN) / 1e6, color='#2E7D32', lw=0.25)
    mid = np.sort(((t.start + t.end) / 2).values); spc = np.median(np.diff(mid)) / 1e3
    axt.set_xlim(WIN0 / 1e6, (WIN0 + WIN) / 1e6); axt.set_ylim(0, 1); axt.set_yticks([])
    for sp_ in ('top', 'right', 'left'): axt.spines[sp_].set_visible(False)
    axt.set_xticks([15, 20, 25, 30, 35]); axt.set_xticklabels(['15', '20', '25', '30', '35'], fontsize=5.6)
    axt.text(0.5, -2.1, f'{len(t)} arrays, median spacing {spc:.0f} kbp', transform=axt.transAxes, fontsize=5.4, va='top', ha='center', color='#2E7D32')
    src[f'd_tyba_{s}'] = tw
cax = ax_mm(117, 131, 1.6, 15); cb = fig.colorbar(pm, cax=cax); cb.ax.tick_params(labelsize=5.2, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)
cb.set_label('relative contacts', fontsize=5.2, labelpad=1)
fig.text(58 / W, 1 - 169.5 / H, 'Position on chromosome 1 (Mbp). Green ticks, $\\it{Tyba}$ arrays; dashed line, P(s) shoulder', fontsize=5.6, ha='center', va='top')

# ---------------------------------------------------------------- e: simulated vs measured axis
e = pd.read_csv(os.path.join(IN, 'panel_e_source_corrected.csv'))   # measured values of R. ciliata and R. nervosa swapped back to match the per-nucleus data (axis_lengths.txt)
re_, pe = ss.pearsonr(e.axis_sim, e.axis_meas); rs, ps_ = ss.spearmanr(e.axis_sim, e.axis_meas)
loo = [ss.pearsonr(e.axis_sim.drop(i), e.axis_meas.drop(i))[0] for i in e.index]; bfit = np.polyfit(e.axis_sim, e.axis_meas, 1)
ax = ax_mm(144, 129, 32, 31)
ax.plot([0.1, 0.42], [0.1, 0.42], color='#BDBDBD', ls=':', lw=0.8); ax.text(0.39, 0.405, '1:1', fontsize=5.4, color='#9E9E9E', ha='right', va='bottom', rotation=45)
xs = np.linspace(e.axis_sim.min(), e.axis_sim.max(), 10); ax.plot(xs, np.polyval(bfit, xs), color='#4D4D4D', ls='--', lw=1.0)
for r in e.itertuples():
    s = k(r.species); c, m = STY[s]
    ax.errorbar(r.axis_sim, r.axis_meas, xerr=[[r.axis_sim - r.axis_sim_lo], [r.axis_sim_hi - r.axis_sim]], yerr=[[r.axis_meas - r.axis_meas_lo], [r.axis_meas_hi - r.axis_meas]],
                fmt='none', ecolor=c, elinewidth=0.6, alpha=0.6)
    ax.scatter(r.axis_sim, r.axis_meas, s=22, color=c, marker=m, edgecolor='white', linewidth=0.3, zorder=3)
ax.set_xlim(0.2, 0.42); ax.set_ylim(0.08, 0.42); ax.set_xticks([0.2, 0.3, 0.4]); ax.set_yticks([0.1, 0.2, 0.3, 0.4])
ax.set_xlabel('Simulated axis length (µm Mbp$^{-1}$)'); ax.set_ylabel('Measured axis length (µm Mbp$^{-1}$)')
ax.text(0.03, 0.97, f'r = {re_:.2f}, $p$ = {pe:.3f}\nleave-one-out r = {min(loo):.2f}-{max(loo):.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=5.8)
label(129, 125, 'e')
src['e'] = e

for ext in ('pdf', 'svg', 'png'): fig.savefig(results(f'Fig4_revised.{ext}'), dpi=600 if ext == 'png' else 300)
# statistics and source data
st = {'a': dict(spearman_rho=ra, P=pa, loglog_slope=oa.params.iloc[1], loglog_P=oa.pvalues.iloc[1], n_nuclei=int(len(a)), per_species=sa.round(4).reset_index().to_dict('records')),
      'b': dict(n_chr=int(len(b)), mixed_slope=mb.params.iloc[1], mixed_se=mb.bse.iloc[1], mixed_P=mb.pvalues.iloc[1], AIC_axis=m1.aic, AIC_chrMb=m2.aic,
                pooled_R2_axis=smf.ols('dco_pct~axis_um', b).fit().rsquared_adj, pooled_R2_chrMb=smf.ols('dco_pct~chr_Mb', b).fit().rsquared_adj),
      'c': dict(spearman_rho=rc, P=pc, mixed_slope=mm.params.iloc[1], mixed_P=mm.pvalues.iloc[1], n_chr=int(len(cm))),
      'e': dict(pearson_r=re_, P=pe, spearman_rho=rs, spearman_P=ps_, loo_min=min(loo), loo_max=max(loo), fit_slope=bfit[0], fit_intercept=bfit[1],
                measured_over_simulated=dict(zip(e.species, (e.axis_meas / e.axis_sim).round(2))))}
json.dump(st, open(results('Fig4_stats.json'), 'w'), indent=1, default=float)
with pd.ExcelWriter(results('Fig4_source_data.xlsx')) as w:
    for kk, v in src.items(): v.to_excel(w, sheet_name=kk[:31], index=False)
print(json.dumps(st, indent=1, default=float)[:3000])
