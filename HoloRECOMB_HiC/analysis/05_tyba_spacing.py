"""Step 5. Extended Data Fig. [TybaSpacing]: Tyba inter-array spacing (distance between midpoints of consecutive arrays),
mean chromosome size and P(s) shoulder position, per species and per chromosome.
Inputs : data/tyba/tyba_arrays_merged10kb.tsv (Tyba monomer runs < 10 kb apart merged; haplotype used for Hi-C, see
         config/libraries.tsv, column tyba_annotation); results/species_shoulder.tsv, chromosome_shoulder.tsv (step 1)
Outputs: results/tyba_spacing_species.tsv, tyba_spacing_chromosomes.tsv, ExtData_TybaSpacing_stats.json,
         ExtData_Fig_TybaSpacing.{pdf,svg,png}"""
import os, sys, re, json, warnings, numpy as np, pandas as pd, scipy.stats as ss, statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); warnings.filterwarnings('ignore')
from hicps import ROOT, STY, LAB, ORDER, FONT, LIBRARIES, within_slope, results
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter, FixedLocator

T = pd.read_csv(os.path.join(ROOT, 'data', 'tyba', 'tyba_arrays_merged10kb.tsv'), sep='\t')
SP = pd.read_csv(results('species_shoulder.tsv'), sep='\t').set_index('library'); CH = pd.read_csv(results('chromosome_shoulder.tsv'), sep='\t')
EXCLUDE = {'filiformis': 'only 16 Tyba arrays annotated'}
ASSEMBLY_MISMATCH = {'barbata'}           # Hi-C map and Tyba annotation from different assemblies: species level only
key = lambda n: (lambda m: m.group(1) if m else n)(re.search(r'scaffold_(\d+)', n))   # R. austrobrasiliensis: chr1_scaffold_1 <-> scaffold_1_a1

rows, crow = [], []
for k in SP.index:
    if k in EXCLUDE: continue
    pre = LIBRARIES[LIBRARIES.library_id == SP.loc[k, 'library_id']].tyba_annotation.iloc[0]
    g = T[T.file.str.startswith(pre)]; hic = CH[CH.library == k]
    names = {key(n): n for n in hic.chrom} if k == 'austrobrasiliensis' else {n: n for n in hic.chrom}
    gk = g.assign(k=g.seqID.map(key) if k == 'austrobrasiliensis' else g.seqID)
    matched = gk[gk.k.isin(names)] if k not in ASSEMBLY_MISMATCH else gk
    sp_all = []
    for s, h in matched.groupby('k'):
        mid = np.sort(((h.start + h.end) / 2).values)
        if len(mid) < 3: continue
        d = np.diff(mid); sp_all += list(d)
        if k not in ASSEMBLY_MISMATCH and s in names:
            r = hic[hic.chrom == names[s]].iloc[0]
            crow.append(dict(library=k, chrom=r.chrom, chr_Mb=r.chr_Mb, n_arrays=len(mid), median_spacing_kb=np.median(d) / 1e3, shoulder_Mb=r.shoulder_Mb))
    rows.append(dict(library=k, n_arrays=len(matched), n_chr_matched=matched.k.nunique(), median_spacing_kb=np.median(sp_all) / 1e3,
                     q1_kb=np.percentile(sp_all, 25) / 1e3, q3_kb=np.percentile(sp_all, 75) / 1e3, mean_chr_Mb=SP.loc[k, 'mean_chr_Mb'],
                     shoulder_Mb=SP.loc[k, 'shoulder_median_Mb'], resolved=bool(SP.loc[k, 'resolved']), low_depth=bool(SP.loc[k, 'low_depth']), annotation=pre))
S = pd.DataFrame(rows); C = pd.DataFrame(crow); S['intervals_per_domain'] = S.shoulder_Mb * 1e3 / S.median_spacing_kb
S.to_csv(results('tyba_spacing_species.tsv'), sep='\t', index=False); C.to_csv(results('tyba_spacing_chromosomes.tsv'), sep='\t', index=False)

R = S[S.resolved]; st = {}
st['a'] = dict(zip(('rho', 'P'), ss.spearmanr(S.mean_chr_Mb, S.median_spacing_kb)), n=len(S), spacing_range_kb=(S.median_spacing_kb.min(), S.median_spacing_kb.max()))
mp = smf.ols('np.log10(shoulder_Mb)~np.log10(median_spacing_kb)+np.log10(mean_chr_Mb)', R).fit()
st['b'] = dict(zip(('rho', 'P'), ss.spearmanr(R.median_spacing_kb, R.shoulder_Mb)), n=len(R), partial_spacing_P=mp.pvalues.iloc[1], partial_chr_P=mp.pvalues.iloc[2],
               intervals_per_domain=(R.intervals_per_domain.min(), R.intervals_per_domain.max()))
R2 = R[~R.low_depth]
st['b_no_low'] = dict(zip(('rho', 'P'), ss.spearmanr(R2.median_spacing_kb, R2.shoulder_Mb)), n=len(R2))
Cr = C[C.shoulder_Mb.notna() & C.library.isin(R.library)].copy()
mm = smf.mixedlm('np.log10(shoulder_Mb)~np.log10(median_spacing_kb)', Cr, groups=Cr.library).fit(reml=True); b, se, pw = within_slope(Cr, 'median_spacing_kb', 'shoulder_Mb', 'library')
st['c'] = dict(n_chr=len(Cr), n_species=Cr.library.nunique(), mixed_slope=mm.params.iloc[1], mixed_P=mm.pvalues.iloc[1], within_slope=b, within_se=se, within_P=pw)
json.dump(st, open(results('ExtData_TybaSpacing_stats.json'), 'w'), indent=1, default=float); print(json.dumps(st, indent=1, default=float))
print(S.round(2).to_string())

# figure
plt.rcParams.update({**FONT, 'font.size': 6.5, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
pf = lambda p: '< 0.001' if p < 0.001 else f'= {p:.2f}' if p >= 0.01 else f'= {p:.3f}'
OPEN = set(S[S.low_depth | ~S.resolved].library)
fmt = FuncFormatter(lambda v, _: f'{v:g}')
def logax(ax, xt, yt):
    ax.set_xscale('log'); ax.set_yscale('log'); ax.xaxis.set_major_locator(FixedLocator(xt)); ax.yaxis.set_major_locator(FixedLocator(yt))
    for a in (ax.xaxis, ax.yaxis): a.set_major_formatter(fmt); a.set_minor_formatter(NullFormatter())
def pt(ax, k, x, y, s=24, alpha=1):
    c, m = STY[k]; o = k in OPEN
    ax.scatter(x, y, marker=m, s=s, facecolor='white' if o else c, edgecolor=c if o else 'white', linewidth=0.9 if o else 0.3, zorder=3, alpha=alpha)
W, H = 180, 62; fig = plt.figure(figsize=(W / 25.4, H / 25.4))
A = lambda x, y, w, h: fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H]); Lt = lambda x, y, t: fig.text(x / W, 1 - y / H, t, fontsize=9, fontweight='bold', va='top')
ax = A(12, 5, 37, 42)
for r in S.itertuples():
    ax.errorbar(r.mean_chr_Mb, r.median_spacing_kb, yerr=[[r.median_spacing_kb - r.q1_kb], [r.q3_kb - r.median_spacing_kb]], fmt='none', ecolor=STY[r.library][0], elinewidth=0.6, alpha=0.5)
    pt(ax, r.library, r.mean_chr_Mb, r.median_spacing_kb)
logax(ax, [10, 20, 50, 100, 200], [50, 100, 200, 500, 1000]); ax.set_xlim(10, 400); ax.set_ylim(40, 1500)
ax.set_xlabel('Mean chromosome size (Mbp)'); ax.set_ylabel('$\\it{Tyba}$ inter-array spacing (kbp)')
ax.text(0.03, 0.97, f'{st["a"]["n"]} species\nSpearman ρ = {st["a"]["rho"]:.2f}, $p$ {pf(st["a"]["P"])}', transform=ax.transAxes, va='top', fontsize=5.8); Lt(1, 2, 'a')
ax = A(64, 5, 37, 42)
for r in R.itertuples(): pt(ax, r.library, r.median_spacing_kb, r.shoulder_Mb)
for f_ in (5, 10, 15):
    xs = np.array([150, 800]); ax.plot(xs, xs * f_ / 1e3, color='#BDBDBD', lw=0.5, ls=':'); xl = min(900, 7.2e3 / f_)
    ax.text(xl, xl * f_ / 1e3, f'{f_}×', fontsize=5.2, color='#9E9E9E', va='bottom', ha='right')
logax(ax, [200, 500, 1000], [1, 2, 3, 4, 6, 8, 12]); ax.set_xlim(150, 1000); ax.set_ylim(1.0, 14)
ax.set_xlabel('$\\it{Tyba}$ inter-array spacing (kbp)'); ax.set_ylabel('P(s) shoulder position (Mbp)')
ax.text(0.03, 0.97, f'{st["b"]["n"]} species\nSpearman ρ = {st["b"]["rho"]:.2f}, $p$ {pf(st["b"]["P"])}\nadjusted for chromosome size:\nspacing $p$ {pf(st["b"]["partial_spacing_P"])}',
        transform=ax.transAxes, va='top', ha='left', fontsize=5.6); Lt(53, 2, 'b')
ax = A(116, 5, 37, 42)
for r in Cr.itertuples(): pt(ax, r.library, r.median_spacing_kb, r.shoulder_Mb, s=12, alpha=0.85)
logax(ax, [100, 200, 500, 1000], [1, 2, 4, 8]); ax.set_xlim(60, 1500); ax.set_ylim(0.5, 16)
ax.set_xlabel('$\\it{Tyba}$ inter-array spacing (kbp)'); ax.set_ylabel('P(s) shoulder position (Mbp)')
ax.text(0.03, 0.97, f'{st["c"]["n_chr"]} chromosomes, {st["c"]["n_species"]} species\nmixed model $p$ {pf(st["c"]["mixed_P"])}\nwithin species $p$ {pf(st["c"]["within_P"])}',
        transform=ax.transAxes, va='top', fontsize=5.8); Lt(105, 2, 'c')
h = [Line2D([], [], marker=STY[k][1], ls='', markerfacecolor='white' if k in OPEN else STY[k][0], markeredgecolor=STY[k][0], markersize=3.6, label=LAB(k))
     for k in ORDER if k in set(S.library)]
lg = fig.legend(handles=h, loc='upper left', bbox_to_anchor=(155 / W, 1 - 2 / H), fontsize=5.0, handletextpad=0.1, labelspacing=0.12)
for t_ in lg.get_texts(): t_.set_fontstyle('italic')
for e in ('pdf', 'svg', 'png'): fig.savefig(results(f'ExtData_Fig_TybaSpacing.{e}'), dpi=600 if e == 'png' else None)
