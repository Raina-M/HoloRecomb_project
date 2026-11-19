"""Step 4. Species with two Arima leaf libraries: agreement of the P(s) shoulder between libraries, stratum-adjusted correlation
coefficient (SCC, HiCRep) between libraries, and the Fig. 4c statistics with the other library of each species in place of
the selected one. Libraries that failed QC (config/libraries.tsv, status 'excluded') are reported but not compared.
Inputs : results/library_summary.tsv, library_chromosome_shoulder.tsv, chromosome_shoulder.tsv (step 1)
Outputs: results/replicate_summary.tsv, replicate_fig4c_stats.tsv, replicate_stats.txt"""
import os, sys, warnings, numpy as np, pandas as pd, scipy.stats as ss, statsmodels.formula.api as smf
from scipy.signal import convolve2d
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); warnings.filterwarnings('ignore')
from hicps import load_library, within_slope, results

L = pd.read_csv(results('library_summary.tsv'), sep='\t'); C = pd.read_csv(results('library_chromosome_shoulder.tsv'), sep='\t')
base = pd.read_csv(results('chromosome_shoulder.tsv'), sep='\t')


def scc(A, B, maxd=50, h=1):
    """Stratum-adjusted correlation coefficient (HiCRep; Yang et al. 2017) of two 100-kb matrices, distances 1..maxd bins,
    after a (2h+1) x (2h+1) mean filter."""
    K = np.ones((2 * h + 1, 2 * h + 1)) / (2 * h + 1) ** 2; n = min(len(A), len(B))
    f = lambda M: convolve2d(np.triu(M[:n, :n]) + np.triu(M[:n, :n], 1).T, K, mode='same'); A, B = f(A), f(B); num = den = 0
    for d in range(1, maxd + 1):
        x, y = np.diagonal(A, d), np.diagonal(B, d); ok = (x > 0) | (y > 0); x, y = x[ok], y[ok]
        if len(x) < 10: continue
        rx, ry = ss.rankdata(x) / len(x), ss.rankdata(y) / len(y); r = np.corrcoef(x, y)[0, 1]; w = len(x) * np.sqrt(np.var(rx) * np.var(ry))
        if np.isfinite(r): num += w * r; den += w
    return num / den


pairs = L.groupby('species').filter(lambda g: len(g) == 2)
for sp, g in pairs.groupby('species'):
    a, b = [load_library(x) for x in g.sort_values('replicate').library_id]
    v = [scc(c1['M'], [c2 for c2 in b if c2['name'] == c1['name']][0]['M']) for c1 in a]
    pairs.loc[g.index, ['scc_median', 'scc_min', 'scc_max']] = [np.median(v), np.min(v), np.max(v)]
pairs = pairs.assign(both_deep=pairs.species.map(pairs.groupby('species').contacts_gt20kb_M.min() > 5))
pairs.to_csv(results('replicate_summary.tsv'), sep='\t', index=False)
ok = pairs[~pairs.species.isin(pairs[pairs.status != 'ok'].species)]
# shoulder agreement: species whose shoulder is resolved (>= 50% of chromosomes) in both libraries
res_both = ok.assign(r=ok.n_resolved / ok.n_chr >= 0.5).groupby('species').r.all()
p = ok[ok.species.isin(res_both[res_both].index)].pivot(index='species', columns='replicate', values='shoulder_median_Mb'); fold = np.maximum(p.rep1 / p.rep2, p.rep2 / p.rep1)
rho, P = ss.spearmanr(p.rep1, p.rep2)
q = C[C.species.isin(p.index)].pivot_table(index=['species', 'chrom'], columns='replicate', values='shoulder_Mb').dropna(); rc, pc = ss.spearmanr(q.rep1, q.rep2)
deep = ok[ok.both_deep].groupby('species').scc_median.first()


def stats(c):
    c = c[c.library != 'rugosa']; r = c.dropna(subset=['shoulder_Mb']); spm = c.groupby('library').chr_Mb.mean().to_frame('L').join(r.groupby('library').shoulder_Mb.median())
    rh, ph = ss.spearmanr(spm.L, spm.shoulder_Mb); m = smf.mixedlm('np.log10(shoulder_Mb)~np.log10(chr_Mb)', r, groups=r.library).fit()
    b, se, pw = within_slope(r, 'chr_Mb', 'shoulder_Mb', 'library')
    return dict(n_species=spm.shape[0], n_chr=len(r), rho=rh, P_rho=ph, mm_slope=m.params.iloc[1], mm_se=m.bse.iloc[1], P_mm=m.pvalues.iloc[1], within=b, within_se=se, P_within=pw)


other = L[(L.status == 'ok') & ~L.selected & L.species.isin(base.library)]
alt = C[C.library_id.isin(other.library_id)].assign(library=lambda d: d.species)
swap = pd.concat([base[~base.library.isin(other.species)], alt[base.columns.intersection(alt.columns)]])
O = pd.DataFrame([dict(version='Fig. 4c (deepest library per species)', **stats(base)),
                  dict(version='other library for ' + ', '.join(other.species), **stats(swap))])
O.to_csv(results('replicate_fig4c_stats.tsv'), sep='\t', index=False)
with open(results('replicate_stats.txt'), 'w') as fh:
    fh.write(f'species with two libraries: {pairs.species.nunique()}; both passed QC: {ok.species.nunique()}; shoulder resolved in both: {len(p)} '
             f'(not resolved in at least one library: {list(res_both[~res_both].index)})\n')
    fh.write(f'species shoulder, library 1 vs 2 ({len(p)} species): Spearman rho = {rho:.2f}, P = {P:.2g}; fold difference median {fold.median():.2f}, '
             f'max {fold.max():.2f} ({fold.idxmax()})\n')
    fh.write(f'per chromosome ({len(q)} chromosomes resolved in both): rho = {rc:.2f}, P = {pc:.2g}\n')
    fh.write(f'SCC where both libraries have > 5 M contacts > 20 kb ({len(deep)} species): {deep.min():.2f}-{deep.max():.2f}; all pairs: '
             f'{ok.groupby("species").scc_median.first().min():.2f}-{ok.groupby("species").scc_median.first().max():.2f}\n')
    fh.write(pairs[['species', 'replicate', 'status', 'selected', 'contacts_gt20kb_M', 'n_resolved', 'n_chr', 'shoulder_median_Mb', 'scc_median', 'scc_min', 'scc_max', 'both_deep']].round(3).to_string(index=False) + '\n')
    fh.write(fold.round(2).to_string() + '\n' + O.round(4).to_string(index=False) + '\n')
print(open(results('replicate_stats.txt')).read())
