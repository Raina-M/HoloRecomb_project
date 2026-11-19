"""Anther Hi-C re-evaluation figure: (a) contact enrichment over random ligation by distance class; (b, c) local P(s) slope of the
background-corrected anther profile versus the somatic profile (map used in Fig. 4c), R. corymbosa and R. radicans.
Only background-corrected anther profiles are shown. Run after anther_hic.py (reads results/anther/anther_stats.txt
and results/anther_curves.pkl)."""
import os, sys, re, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
import pandas as _pd
from hicps.core import RESULTS as _R
_SEL = _pd.read_csv(os.path.join(_R, 'species_shoulder.tsv'), sep='\t').set_index('library').library_id   # somatic library used in Fig. 4c
from hicps.core import load_library, ps_all_pixels, local_slope, RES
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':['Arial','Liberation Sans'],'font.size':6.5,'axes.linewidth':0.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
RESD = os.path.join(HERE, '..', 'results', 'anther')
txt = open(os.path.join(RESD, 'anther_stats.txt')).read(); R = {}
for blk in txt.split('== ')[1:]: R[blk.split(':')[0]] = [float(v) for v in re.findall(r'ratio ([0-9.]+)', blk)]
COL = {'R_cephalotes':'#D95F02','R_corymbosa':'#7570B3','R_radicans':'#E6AB02'}
fig = plt.figure(figsize=(180/25.4, 60/25.4))
ax = fig.add_axes([0.075,0.2,0.235,0.7]); w = 0.26
for i, sp in enumerate(COL): ax.bar(np.arange(3)+(i-1)*w, R[sp], w, color=COL[sp], label='R. '+sp[2:])
ax.axhline(1, color='k', lw=0.5, ls='--'); ax.set_yscale('log'); ax.set_xticks(range(3)); ax.set_xticklabels(['20–100 kb','0.1–1 Mb','1–10 Mb'])
ax.set_ylabel('Observed / expected from random ligation\n(intrachromosomal contacts)'); ax.set_xlabel('Genomic distance')
leg = ax.legend(frameon=False, fontsize=6); [t.set_fontstyle('italic') for t in leg.get_texts()]
fig.text(0.005, 0.95, 'a', fontsize=8, fontweight='bold')
cur = pickle.load(open(os.path.join(RESD, 'anther_curves.pkl'), 'rb'))
for j, (sp, ssp) in enumerate((('R_corymbosa',_SEL['corymbosa']), ('R_radicans',_SEL['radicans']))):
    ax = fig.add_axes([0.40+j*0.31, 0.2, 0.25, 0.7])
    ch = load_library(ssp); x, p = ps_all_pixels(ch); L = np.mean([c['N']*RES for c in ch]); s = local_slope(x, p); k = x < 0.4*L
    ax.plot(x[k], s[k], color='k', lw=1, label='somatic')
    x, p = cur[(sp, True)]; ok = p > 0; x, p = x[ok], p[ok]; s = local_slope(x, p); k = x < 0.4*L
    ax.plot(x[k], s[k], color=COL[sp], lw=1, label='anther, background-corrected')
    ax.set_xscale('log'); ax.set_xlim(1e4, 3e7); ax.set_ylim(-2, 0.2); ax.axhline(-1, color='#BDBDBD', lw=0.4)
    ax.set_xlabel('Genomic distance s (bp)'); ax.set_ylabel('Local slope of P(s)'); ax.set_title('R. '+sp[2:], fontstyle='italic', fontsize=7)
    ax.legend(frameon=False, fontsize=5.5, loc='lower left'); fig.text(0.34+j*0.31, 0.95, 'bc'[j], fontsize=8, fontweight='bold')
for e in ('pdf','png','svg'): fig.savefig(os.path.join(RESD, f'Anther_HiC_reevaluation.{e}'), dpi=300 if e=='png' else None)
