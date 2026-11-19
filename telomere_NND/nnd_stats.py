import pandas as pd, numpy as np, itertools
from scipy.stats import kruskal, rankdata, norm
df=pd.read_csv('telomere_NND.tsv',sep='\t',header=None,names=['species','NND'])
ar=pd.read_csv('cell_areas.tsv',sep='\t',header=None,names=['species','area'])
ma=ar.groupby('species').area.mean()
df['norm']=df.NND/np.sqrt(df.species.map(ma))
def dunn(df,col):
    x=df[col].values; g=df.species.values; r=rankdata(x); N=len(x)
    _,cnt=np.unique(x,return_counts=True); tie=np.sum(cnt**3-cnt)
    sp=sorted(set(g)); res={}
    for a,b in itertools.combinations(sp,2):
        ra,rb=r[g==a].mean(),r[g==b].mean(); na,nb=(g==a).sum(),(g==b).sum()
        se=np.sqrt((N*(N+1)/12 - tie/(12*(N-1)))*(1/na+1/nb)); z=(ra-rb)/se
        res[(a,b)]=2*norm.sf(abs(z))
    keys=list(res); p=np.array([res[k] for k in keys]); m=len(p)
    o=np.argsort(p); q=np.empty(m); prev=1
    for i in range(m-1,-1,-1):
        prev=min(prev,p[o[i]]*m/(i+1)); q[o[i]]=prev
    return dict(zip(keys,q))
for col in ('NND','norm'):
    print('=====',col)
    print(df.groupby('species')[col].agg(['count','median','mean']).round(3))
    H,p=kruskal(*[g[col].values for _,g in df.groupby('species')]); print(f'Kruskal-Wallis H={H:.2f} P={p:.2e}')
    q=dunn(df,col)
    for (a,b),v in sorted(q.items(),key=lambda x:x[1]):
        if 'R_pubera' in (a,b) or 'R_nervosa' in (a,b): print(f'  {a} vs {b}: BH P={v:.4f}')
print('mean nucleus area', ma.round(1).to_dict())
# pooled group comparisons (nucleus level, two-sided Mann-Whitney U)
from scipy.stats import mannwhitneyu
D=['R_breviuscula','R_cephalotes']; I=['R_nervosa','R_ciliata','R_colorata']
for col in ('NND','norm'):
    p=df[df.species=='R_pubera'][col]
    print(col,'R. pubera vs R. breviuscula+R. cephalotes P=%.3f'%mannwhitneyu(p,df[df.species.isin(D)][col]).pvalue,
          '| vs R. nervosa+R. ciliata+R. colorata P=%.3f'%mannwhitneyu(p,df[df.species.isin(I)][col]).pvalue)
