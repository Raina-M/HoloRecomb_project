#!/usr/bin/env python3
"""Sequence features at R. cephalotes chromosome ends, by end class (ceph_end_classes.tsv).
Inputs: Rcephalotes.hap1.scaffolded.chr.fasta, ceph_end_classes.tsv and the R. cephalotes tracks of
Suppl. Dataset 3 (Tyba, TE, genes, other tandem repeats) in TRACKS.
Output: ceph_end_sequence_features.tsv
"""
import re, sys, csv, statistics as st
from collections import defaultdict
FA='Rcephalotes.hap1.scaffolded.chr.fasta'; TRACKS=sys.argv[1]
seqs={};name=None;buf=[]
for l in open(FA):
    if l[0]=='>':
        if name: seqs[name]=''.join(buf).upper()
        name=l[1:].split()[0]; buf=[]
    else: buf.append(l.strip())
seqs[name]=''.join(buf).upper()
G=re.compile(r'T{2,3}AGGG'); C=re.compile(r'CCCTA{2,3}')
def cov(s,rx):
    m=bytearray(len(s))
    for x in rx.finditer(s): m[x.start():x.end()]=b'\x01'*(x.end()-x.start())
    return m
def terminal_telomere(s,end):
    # telomeric array length from the chromosome end: 100-bp windows with >=50% motif coverage,
    # allowing up to 2 consecutive windows below threshold
    W=100; reg=s[:200000] if end=='start' else s[-200000:][::-1]
    rx=C if end=='start' else re.compile(r'GGGAT{2,3}')   # reversed G-strand motif when reading from the end
    m=cov(reg,rx); L=0; miss=0; i=0
    while i+W<=len(m):
        if sum(m[i:i+W])>=W*0.5: L=i+W; miss=0
        else:
            miss+=1
            if miss>2: break
        i+=W
    # also report motif bp in the terminal 20 kb regardless of contiguity
    return L, sum(m[:20000])
def its(s,a,b):
    # interstitial telomeric repeats: >=5 tandem copies of either strand motif
    rx=re.compile(r'(?:T{3}AGGG){5,}|(?:CCCTA{3}){5,}')
    hits=[(x.start()+a,x.end()-x.start()) for x in rx.finditer(s[a:b])]
    return hits
def gc(s): s=s.replace('N',''); return 100*(s.count('G')+s.count('C'))/max(1,len(s))
def load_intervals(path,cols=(0,3,4),gff=True,keep=None):
    d=defaultdict(list)
    for l in open(path):
        if l.startswith('#') or not l.strip(): continue
        r=l.rstrip('\n').split('\t')
        if keep and not keep(r): continue
        c,s,e=(r[0],int(r[3]),int(r[4])) if gff else (r[0],int(r[1]),int(r[2]))
        d[c].append((s,e,r[8] if gff and len(r)>8 else ''))
    return d
T=TRACKS
tyba=load_intervals(f'{T}/R_cephalotes_Tyba_arrays.gff3')
te=load_intervals(f'{T}/R_cephalotes_TE.gff3')
genes=load_intervals(f'{T}/R_cephalotes_genes.bed',gff=False)
trc=load_intervals(f'{T}/R_cephalotes_other_repeat_clustering.gff3')
def inwin(d,c,a,b):
    return [(s,e,x) for s,e,x in d.get(c,[]) if e>=a and s<=b]
def bp(iv,a,b): return sum(min(e,b)-max(s,a)+1 for s,e,x in iv)
rows=list(csv.DictReader(open('ceph_end_classes.tsv'),delimiter='\t'))
out=[]
for r in rows:
    c=f"chr{r['chr']}_h1"; s=seqs[c]; L=len(s); end=r['end']
    tel,tel20=terminal_telomere(s,end)
    rec=dict(chr=r['chr'],end=end,cls=r['cls'],mean_CO_terminal20=r['mean_rate'],telomere_array_bp=tel,telomere_motif_bp_terminal20kb=tel20)
    for lab,w in (('1Mb',1_000_000),('5Mb',5_000_000)):
        a,b=(0,w) if end=='start' else (L-w,L)
        reg=s[a:b]
        h=its(s,a+ (tel if end=='start' else 0), b-(0 if end=='start' else tel))
        rec[f'ITS_n_{lab}']=len(h); rec[f'ITS_bp_{lab}']=sum(x[1] for x in h)
        rec[f'GC_{lab}']=round(gc(reg),2); rec[f'N_bp_{lab}']=reg.count('N')
        ty=inwin(tyba,c,a,b); rec[f'Tyba_n_{lab}']=len(ty); rec[f'Tyba_kb_{lab}']=round(bp(ty,a,b)/1e3,1)
        rec[f'TE_pct_{lab}']=round(100*bp(inwin(te,c,a,b),a,b)/w,1)
        rec[f'genes_n_{lab}']=len(inwin(genes,c,a,b))
        tr=inwin(trc,c,a,b); rec[f'otherTR_{lab}']=';'.join(sorted({re.search(r'Name=([^;]+)',x).group(1) for s_,e_,x in tr})) or '-'
    out.append(rec)
with open('ceph_end_sequence_features.tsv','w') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0].keys()),delimiter='\t'); w.writeheader(); w.writerows(out)
for o in out: print(o)
