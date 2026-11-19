"""Response letter only. Re-implementation of the published contact-domain estimate (3_Ps_plot.R, first turning point of the
P(s) slope from the mean of non-zero pixels) and its behaviour on featureless (null) contact decay.
Outputs (results/response_letter/): published_pipeline_validation.tsv (reported versus re-implemented values; only if the maps of the
submitted version are available in data/original_maps), null_model_turning_points.tsv (turning points returned for featureless
contact decay of the libraries used in the revised Fig. 4c, 3 replicates per library)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from hicps import ROOT, load_library, null_chroms, RES, results
RESULTS = results('response_letter'); os.makedirs(RESULTS, exist_ok=True)
ORIGINAL = os.environ.get('HICPS_ORIGINAL_MAPS', os.path.join(ROOT, 'data', 'original_maps'))   # contact summaries of the maps of the submitted version (optional)
# contact-domain sizes (bp) reported in the original Fig. 4c
SOMATIC_REPORTED={'Rhynchospora_breviuscula':6054928,'Rhynchospora_cephalotes':2044970,'Rhynchospora_ciliata':2864613,'Rhynchospora_corymbosa':3054949,
 'Rhynchospora_filiformis':2734488,'Rhynchospora_gaudichaudii':3989992,'Rhynchospora_holoschoenoides':2184761,'Rhynchospora_nervosa':6014659,
 'Rhynchospora_radicans':4084369,'Rhynchospora_ridleyi':5069720,'Rhynchospora_rugosa':1824891,'Rhynchospora_tenuis_PECP2':5769139,'Rhynchospora_watsonii':5874596}
def loess(x,y,x0,span,deg=2):
    """Nearest-neighbour local polynomial regression with tricube weights (as R loess, family gaussian, exact evaluation)."""
    o=np.argsort(x); x=x[o]; y=y[o]; n=len(x); k=int(np.ceil(span*n)); out=np.empty(len(x0))
    for t,xx in enumerate(x0):
        # find contiguous window of k nearest points
        i=np.searchsorted(x,xx); a=max(0,min(i-k//2,n-k))
        while a>0 and xx-x[a-1]<x[a+k-1]-xx: a-=1
        while a<n-k and x[a+k]-xx<xx-x[a]: a+=1
        xs=x[a:a+k]; ys=y[a:a+k]; h=max(abs(xs[0]-xx),abs(xs[-1]-xx))*1.0000001+1e-12
        w=(1-(np.abs(xs-xx)/h)**3)**3
        X=np.vander(xs-xx,deg+1); W=np.sqrt(w)
        beta=np.linalg.lstsq(X*W[:,None],ys*W,rcond=None)[0]; out[t]=beta[-1]
    return out
def published_pipeline(chroms,grid=None,span=0.1,lo=1e5,hi=8e6):
    """Re-implementation of 3_Ps_plot.R: per chromosome mean of non-zero pixels per distance, normalised to sum 1; pooled;
    loess(log10 Ps ~ log10 s, span 0.1); slope by differences; slope smoothed again with loess span 0.1 (geom_smooth);
    turning point = first local maximum of the smoothed slope in [lo, hi]."""
    X=[];Y=[]
    for c in chroms:
        k=c['C']>0; d=np.arange(len(c['S']))[k]*RES+1; m=c['S'][k]/c['C'][k]; X.append(d); Y.append(m/m.sum())
    x=np.log10(np.concatenate(X)); y=np.log10(np.concatenate(Y))
    g=np.linspace(np.log10(5e3),x.max(),400) if grid is None else grid
    fy=loess(x,y,g,span); sl=np.diff(fy)/np.diff(g); gm=(g[1:]+g[:-1])/2
    ss=loess(gm,sl,gm,span)
    w=(gm>=np.log10(lo))&(gm<=np.log10(hi)); xs=gm[w]; s2=ss[w]; tp=np.nan
    for i in range(1,len(s2)-1):
        if s2[i]>s2[i-1] and s2[i]>=s2[i+1]: tp=10**xs[i]; break
    return gm,ss,tp

if __name__ == '__main__':
    if os.path.isdir(ORIGINAL):
        rows = []
        for sp, rep in SOMATIC_REPORTED.items():
            gm, ss, tp = published_pipeline(load_library(sp, ORIGINAL))
            rows.append(dict(species=sp, reported_Mb=rep / 1e6, reimplemented_Mb=tp / 1e6)); print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'published_pipeline_validation.tsv'), sep='\t', index=False)
    rows = []
    sel = pd.read_csv(results('species_shoulder.tsv'), sep='\t')
    for i, sp in enumerate(sel.library_id):
        ch = load_library(sp); rng = np.random.default_rng(2000 + i)
        for r in range(3):
            gm, ss, tp = published_pipeline(null_chroms(ch, rng))
            w = (gm >= 5) & (gm <= np.log10(8e6)); s = ss[w]; x = gm[w]
            lm = [10 ** x[k] / 1e6 for k in range(1, len(s) - 1) if s[k] > s[k - 1] and s[k] >= s[k + 1]]
            rows.append(dict(library=sp, rep=r, null_global_max_Mb=10 ** x[np.argmax(s)] / 1e6,
                             null_local_maxima_Mb=';'.join(f'{v:.2f}' for v in lm))); print(rows[-1], flush=True)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(RESULTS, 'null_model_turning_points.tsv'), sep='\t', index=False)
    print(f'first turning point on featureless decay: {T.null_local_maxima_Mb.str.split(";").str[0].replace("", np.nan).astype(float).min():.2f}-'
          f'{T.null_local_maxima_Mb.str.split(";").str[0].replace("", np.nan).astype(float).max():.2f} Mb; global maximum {T.null_global_max_Mb.min():.2f}-{T.null_global_max_Mb.max():.2f} Mb')
