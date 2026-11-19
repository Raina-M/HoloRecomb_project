"""Step 3. Robustness of the Fig. 4c association (selected library of each species).
 (1) analysis settings: distance bins, slope window, slope-decrease threshold, trough and search intervals
 (2) leave-one-species-out
 (3) library depth (contacts > 20 kb) versus shoulder position
 (4) Poisson thinning of deep libraries to 1.0, 0.4, 0.2 and 0.1 M contacts > 20 kb (5 replicates)
The featureless null is in step 1 (results/null_model.txt). Output: results/robustness.txt"""
import os, sys, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, scipy.stats as ss
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); warnings.filterwarnings('ignore')
from hicps import load_library, shoulder, chrom_length, thin, contacts_beyond, within_slope, results

SP = pd.read_csv(results('species_shoulder.tsv'), sep='\t').set_index('library')
LIB = {k: load_library(SP.loc[k, 'library_id']) for k in SP.index}
out = open(results('robustness.txt'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.write(s + '\n')

def table(**kw):
    return pd.DataFrame([dict(library=k, chr_Mb=chrom_length(c) / 1e6, sh=shoulder([c], chrom_length(c), **kw)) for k in LIB for c in LIB[k]])

def stats(t, agg='median', drop=()):
    fr = t.groupby('library').sh.apply(lambda v: v.notna().mean()); keep = [k for k in LIB if fr[k] >= 0.5 and k not in drop]
    sp = pd.DataFrame(dict(L=[t[t.library == k].chr_Mb.mean() for k in keep], sh=[getattr(t[t.library == k].sh.dropna(), agg)() for k in keep]), index=keep)
    o = smf.ols('np.log10(sh)~np.log10(L)', sp).fit(); r, p = ss.spearmanr(sp.L, sp.sh)
    cm = t[t.library.isin(keep) & t.sh.notna()].copy()
    mm = smf.mixedlm('np.log10(sh)~np.log10(chr_Mb)', cm, groups=cm.library).fit(reml=True)
    b, se, pw = within_slope(cm, 'chr_Mb', 'sh', 'library')
    return dict(n_species=len(keep), not_resolved=','.join(k for k in LIB if fr[k] < 0.5), n_chr=len(cm), R2_loglog=o.rsquared_adj,
                P_loglog=o.pvalues.iloc[1], rho=r, P_rho=p, mm_slope=mm.params.iloc[1], P_mm=mm.pvalues.iloc[1], within_slope=b, P_within=pw), sp

base = table(); s0, sp0 = stats(base)
say('## Baseline'); say(pd.Series(s0).to_string())
say('\n## 1. Analysis settings (species value = median or mean of resolved chromosomes)')
grid = [dict(), dict(nb=60), dict(nb=120), dict(w=0.2), dict(w=0.4), dict(prom=0.05), dict(prom=0.15), dict(tlo=2e6), dict(tlo=4e6),
        dict(thi=0.5), dict(thi=0.7), dict(slo=2e5), dict(slo=5e5)]
rows = []
for g in grid:
    t = table(**g)
    for agg in ('median', 'mean'):
        s, _ = stats(t, agg); s['setting'] = (str(g) if g else 'baseline') + ' ' + agg; rows.append(s)
R = pd.DataFrame(rows).set_index('setting'); say(R.round(4).to_string())
say(f'range over settings: Spearman P {R.P_rho.min():.2g}-{R.P_rho.max():.2g}; mixed model P {R.P_mm.min():.2g}-{R.P_mm.max():.2g}')
say('\n## 2. Leave one species out')
rows = []
for k in sp0.index:
    s, _ = stats(base, drop=(k,)); s['dropped'] = k; rows.append(s)
R = pd.DataFrame(rows).set_index('dropped'); say(R[['n_species', 'n_chr', 'R2_loglog', 'P_loglog', 'rho', 'P_rho', 'mm_slope', 'P_mm', 'within_slope', 'P_within']].round(4).to_string())
say(f'range: Spearman P {R.P_rho.min():.2g}-{R.P_rho.max():.2g}; mixed model P {R.P_mm.min():.2g}-{R.P_mm.max():.2g}')
say('\n## 3. Library depth (contacts > 20 kb) versus shoulder position and chromosome size (species level)')
dep = pd.Series({k: contacts_beyond(LIB[k]) / 1e6 for k in sp0.index})
r1 = ss.spearmanr(dep, sp0.sh); r2 = ss.spearmanr(dep, sp0.L)
say(f'shoulder ~ depth: rho = {r1[0]:.2f}, P = {r1[1]:.3f}; chromosome size ~ depth: rho = {r2[0]:.2f}, P = {r2[1]:.3f}')
sp0['depth'] = dep; say(smf.ols('np.log10(sh)~np.log10(L)+np.log10(depth)', sp0).fit().summary().tables[1])
say('\n## 4. Poisson thinning to target contacts > 20 kb: species median (number of resolved chromosomes), 5 replicates')
rng = np.random.default_rng(5); fold = []
for k in [k for k in LIB if contacts_beyond(LIB[k]) > 5e6]:
    ch = LIB[k]; full = np.array([shoulder([c], chrom_length(c)) for c in ch]); line = f'{k:16s} full {np.nanmedian(full):.2f} ({np.isfinite(full).sum()}/{len(ch)})'
    for tgt in (1.0, 0.4, 0.2, 0.1):
        f = tgt * 1e6 / contacts_beyond(ch); r = []
        for _ in range(5):
            v = np.array([shoulder([c], chrom_length(c)) for c in thin(ch, f, rng)]); med = np.nanmedian(v) if np.isfinite(v).mean() >= 0.5 else np.nan
            r.append(f'{med:.2f}({np.isfinite(v).sum()})')
            if tgt == 1.0 and np.isfinite(med): fold.append(max(med / np.nanmedian(full), np.nanmedian(full) / med))
        line += f' | {tgt} M: ' + ' '.join(r)
    say(line)
say(f'thinning to 1 M contacts > 20 kb: species estimates changed by up to {max(fold):.2f}-fold (median {np.median(fold):.2f})')
out.close()
