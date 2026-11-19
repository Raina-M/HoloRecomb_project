#!/usr/bin/env python3
"""R3.9: CO rate versus chromosome size across seven Rhynchospora species.
Input: <species>_CO_num_per_chr.txt (chr, size_bp, CO count) from raw_files/Figure_3/CO_rates.
CO rate (cM/Mb) = CO count * 1e8 / (size_bp * n meioses), as in CO_rate_vs_chrsize.R,
with the R. breviuscula denominator corrected to its own chromosome sizes (the R script used R. barbata sizes)."""
import pandas as pd, numpy as np, statsmodels.formula.api as smf, scipy.stats as ss, warnings
warnings.filterwarnings('ignore')
MEI={'R_barbata':2495,'R_breviuscula':1641,'R_cephalotes':2719,'R_ciliata':597,'R_colorata':429,'R_nervosa':1173,'R_pubera':489*2}
rows=[]
for sp,n in MEI.items():
    t=pd.read_csv(f'{sp}_CO_num_per_chr.txt',sep=r'\s+',header=None,names=['chr','size','co'])
    t['species']=sp; t['n_meioses']=n; t['co_rate']=t.co*1e8/(t['size']*n); rows.append(t)
df=pd.concat(rows,ignore_index=True)
df['lx']=np.log10(df['size']); df['ly']=np.log10(df.co_rate)
# buggy breviuscula value for reference
barb=df[df.species=='R_barbata']['size'].values
bug=df.copy(); m=bug.species=='R_breviuscula'; bug.loc[m,'ly']=np.log10(bug.loc[m,'co']*1e8/(barb*1641))
# MLH1-based genome-wide points (not in fits)
mlh=pd.DataFrame({'species':['R_pubera','R_breviuscula'],'size':[1472673118,368174147],'co':[639,921],'cells':[84,152]})
mlh['co_rate']=mlh.co*1e8/(mlh['size']*mlh.cells*2)
out=[]; P=lambda s:out.append(s)
def ols(d,lab):
    f=smf.ols('ly~lx',d).fit(); P(f"{lab}: slope {f.params.lx:.3f} ± {f.bse.lx:.3f} s.e., adj. R2 = {f.rsquared_adj:.3f}, P = {f.pvalues.lx:.2e}, n = {len(d)}"); return f
P('== OLS log10(CO rate) ~ log10(chromosome size)')
f_all=ols(df,'All 39 chromosomes (corrected)')
ols(bug,'All 39 chromosomes, original script (R. breviuscula divided by R. barbata sizes)')
ols(df[df.species!='R_pubera'],'Excluding R. pubera (34 chromosomes)')
wm=pd.concat([df[['lx','ly']],pd.DataFrame({'lx':np.log10(mlh['size']),'ly':np.log10(mlh.co_rate)})])
ols(wm,'Including the two MLH1-based genome-wide points (as drawn in submitted Fig. 3b)')
P('\n== Linear mixed model, species random intercept (REML)')
mm=smf.mixedlm('ly~lx',df,groups=df.species).fit(reml=True)
P(f"slope {mm.params.lx:.3f} ± {mm.bse.lx:.3f} s.e., z = {mm.tvalues.lx:.2f}, P = {mm.pvalues.lx:.2e}; species variance = {float(mm.cov_re.iloc[0,0]):.2e}, residual variance = {mm.scale:.4f}")
mm2=smf.mixedlm('ly~lx',df[df.species!='R_pubera'],groups=df[df.species!='R_pubera'].species).fit(reml=True)
P(f"excluding R. pubera: slope {mm2.params.lx:.3f} ± {mm2.bse.lx:.3f} s.e., P = {mm2.pvalues.lx:.2e}")
P('\n== Within-species (variables centred on species means)')
for c in ('lx','ly'): df[c+'_c']=df[c]-df.groupby('species')[c].transform('mean')
w=smf.ols('ly_c~lx_c-1',df).fit()
# correct df: n - 1 slope - 7 species means
dfree=len(df)-1-df.species.nunique(); se=np.sqrt(w.ssr/dfree/np.sum(df.lx_c**2)); tt=w.params.lx_c/se; pw=2*ss.t.sf(abs(tt),dfree)
P(f"pooled within-species slope {w.params.lx_c:.3f} ± {se:.3f} s.e., t({dfree}) = {tt:.2f}, P = {pw:.2e}")
fe=smf.ols('ly~lx+C(species)',df).fit(); P(f"(equivalent fixed-effects model: slope {fe.params.lx:.3f}, P = {fe.pvalues.lx:.2e})")
P('\n== Per species')
tab=[]
for sp,g in df.groupby('species'):
    r,p=ss.spearmanr(g['size'],g.co_rate); o=smf.ols('ly~lx',g).fit(); cv=g['size'].std(ddof=1)/g['size'].mean()*100
    tab.append(dict(species=sp,n_chromosomes=len(g),size_CV_pct=round(cv,1),spearman_rho=round(r,2),spearman_P=round(p,4),ols_slope=round(o.params.lx,2),ols_slope_se=round(o.bse.lx,2),ols_P=round(o.pvalues.lx,4)))
tab=pd.DataFrame(tab); P(tab.to_string(index=False))
P('\nMLH1-based genome-wide points (plotted, not fitted):\n'+mlh.assign(co_rate=mlh.co_rate.round(3)).to_string(index=False))
open('r39_stats.txt','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
tab.to_csv('Suppl_Table_MixedModel_per_species.tsv',sep='\t',index=False)
df.drop(columns=['lx_c','ly_c']).to_csv('CO_rate_per_chromosome.tsv',sep='\t',index=False)
mlh.to_csv('MLH1_genomewide_points.tsv',sep='\t',index=False)
import json; json.dump({'ols':[f_all.params.Intercept,f_all.params.lx,f_all.rsquared_adj],'mm':[mm.params.Intercept,mm.params.lx,mm.pvalues.lx],'w':[w.params.lx_c,pw]},open('fits.json','w'))
