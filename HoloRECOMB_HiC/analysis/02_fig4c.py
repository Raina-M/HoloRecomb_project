"""Step 2. Fig. 4c: P(s) shoulder position (contact-domain size) against mean chromosome size.
Statistics: Spearman correlation and log-log least squares across species (species medians), linear mixed model of
log10 shoulder on log10 chromosome size across chromosomes (species as random intercept), and the pooled within-species
slope after centring on species means; each with all resolved species and without low-depth libraries.
Inputs : results/chromosome_shoulder.tsv, results/species_shoulder.tsv (step 1)
Outputs: results/fig4c_stats.txt, results/fig4c_stats.json, results/Fig4c_Ps_shoulder.{pdf,svg,png}"""
import os, sys, json, numpy as np, pandas as pd, statsmodels.formula.api as smf, scipy.stats as ss, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); warnings.filterwarnings('ignore')
from hicps import STY, LAB, ORDER, FONT, pf, within_slope, results
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sp = pd.read_csv(results('species_shoulder.tsv'), sep='\t').set_index('library'); ch = pd.read_csv(results('chromosome_shoulder.tsv'), sep='\t')
NR = [k for k in sp.index if not sp.loc[k, 'resolved']]; LOW = [k for k in sp.index if sp.loc[k, 'low_depth'] and k not in NR]
KS = [k for k in sp.index if k not in NR]


def analyse(keep):
    d = sp.loc[keep]; o = smf.ols('np.log10(shoulder_median_Mb)~np.log10(mean_chr_Mb)', d).fit(); r, p = ss.spearmanr(d.mean_chr_Mb, d.shoulder_median_Mb)
    cm = ch[ch.library.isin(keep) & ch.shoulder_Mb.notna()].copy()
    mm = smf.mixedlm('np.log10(shoulder_Mb)~np.log10(chr_Mb)', cm, groups=cm.library).fit(reml=True)
    b, se, pw = within_slope(cm, 'chr_Mb', 'shoulder_Mb', 'library')
    return dict(n_species=len(d), n_chr=len(cm), n_chr_total=int(ch.library.isin(keep).sum()), rho=r, P_rho=p, adjR2=o.rsquared_adj, P_ols=o.pvalues.iloc[1],
                ols_slope=o.params.iloc[1], mm_intercept=mm.params.iloc[0], mm_slope=mm.params.iloc[1], mm_se=mm.bse.iloc[1], P_mm=mm.pvalues.iloc[1],
                within=b, within_se=se, P_within=pw, shoulder_range=(d.shoulder_median_Mb.min(), d.shoulder_median_Mb.max()),
                chr_shoulder_range=(cm.shoulder_Mb.min(), cm.shoulder_Mb.max()))


A = analyse(KS); X = analyse([k for k in KS if k not in LOW])
txt = [f'Species shoulder = median of resolved chromosomes; not resolved (< 50% of chromosomes): {NR} '
       + ', '.join(f'{k} {sp.loc[k, "n_resolved"]}/{sp.loc[k, "n_chr"]}' for k in NR),
       f'Low-depth libraries (featureless null produces a shoulder; open symbols): {LOW}']
for lab, s in (('all resolved species', A), ('without low-depth libraries', X)):
    txt += [f'## {lab}: {s["n_species"]} species, {s["n_chr"]} of {s["n_chr_total"]} chromosomes resolved',
            f'   species medians {s["shoulder_range"][0]:.2f}-{s["shoulder_range"][1]:.2f} Mb; chromosomes {s["chr_shoulder_range"][0]:.2f}-{s["chr_shoulder_range"][1]:.2f} Mb',
            f'   Spearman rho = {s["rho"]:.2f}, P = {s["P_rho"]:.2g}; log-log OLS adj. R2 = {s["adjR2"]:.2f}, P = {s["P_ols"]:.2g}',
            f'   mixed model slope = {s["mm_slope"]:.2f} +/- {s["mm_se"]:.2f}, P = {s["P_mm"]:.2g}',
            f'   within-species slope = {s["within"]:.2f} +/- {s["within_se"]:.2f}, P = {s["P_within"]:.3f}']
txt.append(sp[['library_id', 'mean_chr_Mb', 'shoulder_median_Mb', 'shoulder_min_Mb', 'shoulder_max_Mb', 'n_resolved', 'n_chr', 'contacts_gt20kb_M', 'low_depth']].round(2).to_string())
open(results('fig4c_stats.txt'), 'w').write('\n'.join(txt) + '\n'); print('\n'.join(txt))
json.dump(dict(all=A, no_low=X, NR=NR, LOW=LOW), open(results('fig4c_stats.json'), 'w'), indent=1, default=float)


def draw(ax, legend_ax=None, fs=6):
    """Fig. 4c panel; used here and by 09_fig4.py."""
    d = sp.loc[KS]
    for k, r in d.iterrows():
        c, m = STY[k]
        ax.errorbar(r.mean_chr_Mb, r.shoulder_median_Mb, yerr=[[max(r.shoulder_median_Mb - r.shoulder_min_Mb, 0)], [max(r.shoulder_max_Mb - r.shoulder_median_Mb, 0)]],
                    fmt='none', ecolor=c, elinewidth=0.6, alpha=0.6)
        if k in LOW: ax.scatter(r.mean_chr_Mb, r.shoulder_median_Mb, marker=m, s=22, facecolor='white', edgecolor=c, linewidth=0.9, zorder=3)
        else: ax.scatter(r.mean_chr_Mb, r.shoulder_median_Mb, marker=m, s=22, color=c, edgecolor='white', linewidth=0.3, zorder=3)
    xs = np.logspace(np.log10(d.mean_chr_Mb.min()), np.log10(d.mean_chr_Mb.max()), 50)
    ax.plot(xs, 10 ** (A['mm_intercept'] + A['mm_slope'] * np.log10(xs)), color='#4D4D4D', ls='--', lw=1.1)
    for k in NR:
        ax.scatter(sp.loc[k, 'mean_chr_Mb'], 0.95, marker=STY[k][1], s=22, facecolor='white', edgecolor=STY[k][0], linewidth=0.8, zorder=3, clip_on=False)
        ax.annotate('not resolved', (sp.loc[k, 'mean_chr_Mb'], 0.95), xytext=(5, -1), textcoords='offset points', fontsize=fs - 0.5, va='center')
    ax.set_yscale('log'); ax.set_ylim(0.88, 9); ax.set_yticks([1, 2, 3, 4, 5, 6, 8]); ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '8'])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xscale('log'); ax.set_xlim(10, 450); ax.set_xticks([10, 20, 50, 100, 200, 400]); ax.set_xticklabels(['10', '20', '50', '100', '200', '400'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Chromosome size (Mbp)'); ax.set_ylabel('P(s) shoulder position (Mbp)')
    ax.text(0.03, 0.98, f'{A["n_species"]} species: Spearman ρ = {A["rho"]:.2f}, $p$ {pf(A["P_rho"])}\n{A["n_chr"]} chromosomes: mixed model $p$ {pf(A["P_mm"])}',
            transform=ax.transAxes, ha='left', va='top', fontsize=fs)
    h = [Line2D([], [], marker=STY[k][1], ls='', color=STY[k][0], markersize=3.8, label=LAB(k)) if k not in NR + LOW else
         Line2D([], [], marker=STY[k][1], ls='', markerfacecolor='white', markeredgecolor=STY[k][0], markersize=3.8, label=LAB(k)) for k in sp.index]
    return h


if __name__ == '__main__':
    plt.rcParams.update({**FONT, 'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(100 / 25.4, 66 / 25.4)); ax = fig.add_axes([0.11, 0.16, 0.52, 0.80])
    h = draw(ax)
    leg = ax.legend(handles=h, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6, handletextpad=0.2, labelspacing=0.2)
    for t in leg.get_texts(): t.set_fontstyle('italic')
    fig.text(0.01, 0.97, 'c', fontsize=9, fontweight='bold', va='top')
    for e in ('pdf', 'svg', 'png'): fig.savefig(results(f'Fig4c_Ps_shoulder.{e}'), dpi=600 if e == 'png' else None)
