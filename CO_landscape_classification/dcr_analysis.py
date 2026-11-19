#!/usr/bin/env python3
"""Quantitative classification of CO landscapes (revision point R3.2).
Input: <species>_CO_rates.txt from Suppl. Dataset 3; R_pubera_CO_rates.txt from pubera_rates_from_marey.py (F2 linkage map) (chr, pos, CI_start, co_rate, CI_end; 100-kb steps).
Per chromosome: distal-to-central ratio (DCR) = mean CO rate at 5-20% of chromosome length from each end /
mean CO rate in the central 40% (30-70%). Length-scaled profile: 20 equal bins along the chromosome, folded
into 10 bins from end to centre (mean of mirror bins), divided by the chromosome mean (shape only).
PCA and k-means (k = 2) on the 10-bin profiles. Outputs: dcr_per_chromosome.tsv, dcr_stats.txt.
"""
import glob, os, math, itertools, statistics as st, random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
SP=['R_cephalotes','R_breviuscula','R_pubera','R_nervosa','R_ciliata','R_colorata','R_barbata']   # Fig. 2a order; R. pubera (F2 map) added
rows=[]
for sp in SP:
    d={}
    for i,l in enumerate(open(f'{sp}_CO_rates.txt')):
        if i==0: continue
        x=l.split(); d.setdefault(x[0],[]).append((float(x[1]),float(x[3])))
    for c,v in d.items():
        v.sort(); L=v[-1][0]                     # chromosome length taken as the last window position
        m=lambda a,b:st.mean([r for p,r in v if a*L<=p<=b*L])
        dist=st.mean([r for p,r in v if 0.05*L<=p<=0.2*L or 0.8*L<=p<=0.95*L]); cen=m(0.3,0.7)
        bins=[st.mean([r for p,r in v if k*L/20<=p<(k+1)*L/20] or [0]) for k in range(20)]
        fold=[(bins[k]+bins[19-k])/2 for k in range(10)]
        mu=st.mean(fold); sd=st.pstdev(fold)
        rows.append(dict(species=sp.replace('_',' ').replace('R ','R. '),chr=c,length_Mb=round(L/1e6,1),log2DCR=math.log2(dist/cen),profile=[(f-mu)/sd for f in fold],profile_meanscaled=[f/st.mean(r for p,r in v) for f in fold]))
X=np.array([r['profile'] for r in rows])
pca=PCA().fit(X); pcs=pca.transform(X)
km=KMeans(2,n_init=100,random_state=1).fit(X); lab=km.labels_
# name clusters by mean DCR
hi=int(np.argmax([np.mean([r['log2DCR'] for r,l in zip(rows,lab) if l==k]) for k in (0,1)]))
for r,l,p in zip(rows,lab,pcs): r['cluster']='distal-biased' if l==hi else 'irregular'; r['PC1']=p[0]; r['PC2']=p[1]
out=[]
out.append(f"Chromosomes: {len(rows)}; species: {len(SP)}")
for sp in SP:
    s=sp.replace('_',' ').replace('R ','R. '); rr=[r for r in rows if r['species']==s]
    out.append(f"{s}: n = {len(rr)}, mean log2 DCR = {st.mean(r['log2DCR'] for r in rr):.2f} (range {min(r['log2DCR'] for r in rr):.2f} to {max(r['log2DCR'] for r in rr):.2f}); k-means clusters: {sorted(set(r['cluster'] for r in rr))}")
out.append(f"PCA variance explained: PC1 {100*pca.explained_variance_ratio_[0]:.1f}%, PC2 {100*pca.explained_variance_ratio_[1]:.1f}%")
pure=all(len(set(r['cluster'] for r in rows if r['species']==s))==1 for s in set(r['species'] for r in rows))
out.append(f"k-means assigns all chromosomes of each species to one cluster: {pure}")
D=[r for r in rows if r['cluster']=='distal-biased']; I=[r for r in rows if r['cluster']=='irregular']
out.append(f"Distal-biased cluster: {sorted(set(r['species'] for r in D))}; chromosome log2 DCR {min(r['log2DCR'] for r in D):.2f} to {max(r['log2DCR'] for r in D):.2f}")
out.append(f"Irregular cluster: {sorted(set(r['species'] for r in I))}; chromosome log2 DCR {min(r['log2DCR'] for r in I):.2f} to {max(r['log2DCR'] for r in I):.2f}")
out.append(f"Gap between clusters in chromosome log2 DCR: {min(r['log2DCR'] for r in D)-max(r['log2DCR'] for r in I):.2f}")
# all 2-vs-4 partitions of species: difference in mean chromosome log2 DCR
names=sorted(set(r['species'] for r in rows))
def diff(g):
    a=[r['log2DCR'] for r in rows if r['species'] in g]; b=[r['log2DCR'] for r in rows if r['species'] not in g]
    return st.mean(a)-st.mean(b)
parts=sorted(((abs(diff(set(g))),g) for k in (1,2,3) for g in itertools.combinations(names,k)),reverse=True)
out.append("Top species bipartitions by |difference in mean log2 DCR|: "+'; '.join(f"{'+'.join(g)} {v:.2f}" for v,g in parts[:3])+f" (of {len(parts)})")
# chromosome-level permutation test for the observed grouping
obs=diff(set(r['species'] for r in D)); vals=[r['log2DCR'] for r in rows]; nD=len(D); random.seed(1); N=100000; ge=0
for _ in range(N):
    random.shuffle(vals); ge+= (st.mean(vals[:nD])-st.mean(vals[nD:]))>=obs
out.append(f"Permutation test (chromosome labels, {N} permutations): observed difference {obs:.2f}, P = {(ge+1)/(N+1):.1e}")
Xm=np.array([r['profile_meanscaled'] for r in rows]); km2=KMeans(2,n_init=100,random_state=1).fit(Xm)
hi2=int(np.argmax([np.mean([r['log2DCR'] for r,l in zip(rows,km2.labels_) if l==k]) for k in (0,1)]))
dist_sp=set(r['species'] for r in D)
mis=[f"{r['species']} chr{r['chr']}" for r,l in zip(rows,km2.labels_) if (r['species'] in dist_sp)!=(l==hi2)]
out.append(f"Sensitivity, k-means on mean-scaled profiles (amplitude retained): chromosomes assigned against their species group: {mis or 'none'}")
open('dcr_stats.txt','w').write('\n'.join(out)+'\n'); print('\n'.join(out))
with open('dcr_per_chromosome.tsv','w') as f:
    f.write('species\tchr\tlength_Mb\tlog2DCR\tcluster\tPC1\tPC2\t'+'\t'.join(f'bin{k+1}' for k in range(10))+'\n')
    for r in rows: f.write(f"{r['species']}\t{r['chr']}\t{r['length_Mb']}\t{r['log2DCR']:.3f}\t{r['cluster']}\t{r['PC1']:.3f}\t{r['PC2']:.3f}\t"+'\t'.join(f'{x:.3f}' for x in r['profile'])+'\n')
import pickle; pickle.dump((rows,pca.explained_variance_ratio_),open('dcr.pkl','wb'))
