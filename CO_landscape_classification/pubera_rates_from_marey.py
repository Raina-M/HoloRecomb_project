#!/usr/bin/env python3
"""Convert the R. pubera F2 linkage map (Marey map) into a CO rate table in the format of Suppl. Dataset 3.
The map is made monotone (cumulative maximum of cM along physical position) and linearly interpolated;
the CO rate at each 100-kb step is the slope (cM/Mb) over a 1-Mb window centred on that position.
Chromosome lengths from raw_files/Figure_3/CO_rates/R_pubera_CO_num_per_chr.txt.
"""
import numpy as np
L={'1':375421704,'2':361203633,'3':265146654,'4':240819442,'5':230081685}
m={}
for l in open('merged_linkage_brev_ceph_cili_pube.txt'):
    c,cm,p=l.split()
    if c.startswith('pube_'): m.setdefault(c.replace('pube_Chr',''),[]).append((float(p),float(cm)))
nonmono=0
with open('R_pubera_CO_rates.txt','w') as f:
    f.write('chr\tpos\tCI_start\tco_rate\tCI_end\n')
    for c in sorted(m):
        v=sorted(m[c]); pos=np.array([p for p,_ in v]); cm=np.array([x for _,x in v])
        nonmono+=int(np.sum(np.diff(cm)<0)); cm=np.maximum.accumulate(cm)
        # extend flat to the chromosome ends (no information beyond terminal markers)
        pos=np.r_[0,pos,L[c]]; cm=np.r_[cm[0],cm,cm[-1]]
        for x in np.arange(5e5,L[c]-5e5+1,1e5):
            r=(np.interp(x+5e5,pos,cm)-np.interp(x-5e5,pos,cm))/1.0
            f.write(f'{c}\t{x:.0f}\tNA\t{r:.6f}\tNA\n')
print('non-monotone steps corrected:',nonmono)
