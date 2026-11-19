#!/usr/bin/env python3
"""CO rates at R. cephalotes chromosome ends classified by synteny with R. breviuscula.
Inputs (same folder): R_cephalotes_CO_rates.txt, R_breviuscula_CO_rates.txt (Suppl. Dataset 3),
ceph_brev_synteny_blocks_from_Fig2a.tsv (DEEPSPACE blocks >=0.5 Mb, read from the Fig. 2a vector graphic).
Outputs: ceph_end_classes.tsv, stats.txt, Fig2_new_panels.{pdf,svg,png}
"""
import csv, statistics as st
from collections import defaultdict
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr, binomtest
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CL={1:46646626,2:39764246,3:35948395,4:32796034,5:30371766,6:30785530,7:30061336,8:28880904,9:28351248}
BL={1:91632052,2:70953004,3:69585868,4:66447897,5:69555326}
FIG_ORDER=[3,9,4,6,7,8,5,2,1]   # order in Fig. 2a
WIN=0.2                          # terminal fraction of chromosome length
def load(f):
    d=defaultdict(list)
    for i,l in enumerate(open(f)):
        if i==0: continue
        c,p,lo,r,hi=l.split(); d[int(c)].append((float(p),float(r)))
    for c in d: d[c].sort()
    return d
C=load('R_cephalotes_CO_rates.txt'); Bv=load('R_breviuscula_CO_rates.txt')
blk=[dict(cc=int(r['ceph_chr']),cs=int(r['ceph_start']),ce=int(r['ceph_end']),bc=int(r['brev_chr']),bs=int(r['brev_start']),be=int(r['brev_end']),o=r['orientation']) for r in csv.DictReader(open('ceph_brev_synteny_blocks_from_Fig2a.tsv'),delimiter='\t')]

def end_class(c,end):
    b=[x for x in blk if x['cc']==c]
    t=min(b,key=lambda x:x['cs']) if end=='start' else max(b,key=lambda x:x['ce'])
    # breviuscula coordinate of the block edge that faces the cephalotes chromosome end
    if end=='start': bp=t['bs'] if t['o']=='+' else t['be']
    else: bp=t['be'] if t['o']=='+' else t['bs']
    dist=min(bp, BL[t['bc']]-bp)
    cls='fission-derived' if dist>10e6 else ('conserved' if dist<2e6 else 'terminal rearrangement')
    return t['bc'],bp,dist,cls
def internal_frac(c,end,frac=WIN,thr=10e6):
    # fraction of the terminal 20% syntenic to R. breviuscula positions >10 Mb from a chromosome end
    L=CL[c]; a,b=(0,frac*L) if end=='start' else ((1-frac)*L,L); tot=0; intl=0
    for x in blk:
        if x['cc']!=c: continue
        s_,e_=max(a,x['cs']),min(b,x['ce'])
        if e_<=s_: continue
        n=int((e_-s_)/1e5)+1
        for k in range(n):
            p=s_+(e_-s_)*(k+0.5)/n; f=(p-x['cs'])/(x['ce']-x['cs'])
            bp=x['bs']+f*(x['be']-x['bs']) if x['o']=='+' else x['be']-f*(x['be']-x['bs'])
            tot+=1; intl+= min(bp,BL[x['bc']]-bp)>thr
    return intl/tot
def term(c,end):
    x=[r for p,r in C[c] if (p<WIN*CL[c] if end=='start' else p>=(1-WIN)*CL[c])]
    return st.mean(x), max(x)
rows=[]
for c in range(1,10):
    for end in ('start','end'):
        bc,bp,dist,cls=end_class(c,end); m,mx=term(c,end)
        fr=internal_frac(c,end)
        rows.append(dict(chr=c,end=end,brev_chr=bc,brev_pos_Mb=round(bp/1e6,2),dist_to_brev_end_Mb=round(dist/1e6,2),cls=cls,internal_frac_terminal20=round(fr,2),rearr_derived=fr>=0.3,mean_rate=round(m,2),max_rate=round(mx,2)))
# dominant peak per chromosome
for c in range(1,10):
    p,r=max(C[c],key=lambda x:x[1]); side='start' if p<CL[c]/2 else 'end'
    for row in rows:
        if row['chr']==c: row['dominant_peak_here']= (row['end']==side); row['peak_Mb']=round(p/1e6,1) if row['end']==side else ''; row['peak_rate']=round(r,2) if row['end']==side else ''
with open('ceph_end_classes.tsv','w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),delimiter='\t'); w.writeheader(); w.writerows(rows)

out=[]
F=[r for r in rows if r['rearr_derived']]; O=[r for r in rows if not r['rearr_derived']]; K=[r for r in rows if r['cls']=='conserved']
out.append(f"Ends: {len(F)} rearrangement-derived (terminal 20% with >=30% sequence syntenic to internal R. breviuscula positions; fraction {min(r['internal_frac_terminal20'] for r in F)}-{max(r['internal_frac_terminal20'] for r in F)} vs <= {max(r['internal_frac_terminal20'] for r in O)} for other ends), of which {len([r for r in F if r['cls']=='fission-derived'])} fission-derived;, {len([r for r in rows if r['cls']=='terminal rearrangement'])} terminal rearrangement, {len(K)} conserved")
pk=[r for r in F if r['dominant_peak_here']]; out.append(f"Rearrangement-derived ends carrying the chromosome's highest CO rate: {len(pk)}/{len(F)}; binomial (p=0.5, one-sided) P = {binomtest(len(pk),len(F),0.5,alternative='greater').pvalue:.4f}")
pairs=[(a,b) for a in F for b in O if a['chr']==b['chr']]
out.append("Paired (same chromosome) mean CO rate in terminal 20%%: rearrangement-derived vs opposite end: " + ', '.join(f"chr{a['chr']} {a['mean_rate']} vs {b['mean_rate']}" for a,b in pairs))
out.append(f"Wilcoxon signed-rank (one-sided) mean: P = {wilcoxon([a['mean_rate'] for a,b in pairs],[b['mean_rate'] for a,b in pairs],alternative='greater').pvalue:.4f}; max: P = {wilcoxon([a['max_rate'] for a,b in pairs],[b['max_rate'] for a,b in pairs],alternative='greater').pvalue:.4f}")
out.append(f"Median terminal mean rate: rearrangement-derived {st.median([r['mean_rate'] for r in F]):.2f}, other {st.median([r['mean_rate'] for r in O]):.2f}, conserved {st.median([r['mean_rate'] for r in K]):.2f} cM/Mb")
out.append(f"Mann-Whitney (one-sided) rearrangement-derived vs all other ends: P = {mannwhitneyu([r['mean_rate'] for r in F],[r['mean_rate'] for r in O],alternative='greater').pvalue:.4f}; vs conserved: P = {mannwhitneyu([r['mean_rate'] for r in F],[r['mean_rate'] for r in K],alternative='greater').pvalue:.4f}")
s=spearmanr([r['internal_frac_terminal20'] for r in rows],[r['mean_rate'] for r in rows]); out.append(f"Spearman, fraction of terminal 20% syntenic to internal R. breviuscula positions vs terminal mean CO rate (18 ends): rho = {s.statistic:.2f}, P = {s.pvalue:.4f}")
# peak distance from fission end, and breviuscula rate at syntenic position
def tobrev(c,p):
    for x in blk:
        if x['cc']==c and x['cs']<=p<=x['ce']:
            f=(p-x['cs'])/(x['ce']-x['cs']); return x['bc'], x['bs']+f*(x['be']-x['bs']) if x['o']=='+' else x['be']-f*(x['be']-x['bs'])
    return None,None
def brate(c,p): x=[r for q,r in Bv[c] if abs(q-p)<=5e5]; return st.mean(x)
dl=[]; lines=[]
for r in rows:
    if r['dominant_peak_here']:
        p=r['peak_Mb']*1e6; d=p if r['end']=='start' else CL[r['chr']]-p
        bc,bp=tobrev(r['chr'],p); br=brate(bc,bp); bm=st.mean(x for q,x in Bv[bc])
        lines.append(f"chr{r['chr']} peak {r['peak_rate']} cM/Mb at {r['peak_Mb']} Mb ({d/1e6:.1f} Mb from the {r['cls']} end) -> R. breviuscula chr{bc} {bp/1e6:.1f} Mb, {br:.2f} cM/Mb (chromosome mean {bm:.2f})")
        if r['rearr_derived']: dl.append(d/1e6)
out.append(f"Distance of dominant peak from rearrangement-derived end: {min(dl):.1f} to {max(dl):.1f} Mb (median {st.median(dl):.1f})")
out+=lines
open('stats.txt','w').write('\n'.join(out)+'\n'); print('\n'.join(out))

