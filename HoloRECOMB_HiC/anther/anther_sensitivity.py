"""Sensitivity of the anther-versus-somatic shoulder comparison: pooled slope maximum (0.3 Mb to f x mean chromosome length)
of the somatic profile and of the background-corrected anther profile, over distance-bin number (60, 80, 120), slope window
(0.2, 0.3, 0.4 log10 units), upper search limit (0.2, 0.3, 0.4 x mean length) and background scaling (0.5, 0.75, 1, 1.25; background-corrected profiles only),
and for VC_SQRT-normalised instead of raw anther contacts (vcsqrt/<chr>.npz: S, C, L; extraction/aggregate_vcsqrt.py).
Output: results/anther_sensitivity.txt"""
import os, glob, sys, numpy as np
from anther_hic import load_raw, trans_background, ps_corrected, slope_max, DATA, RESULTS, RES
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pandas as _pd
from hicps.core import RESULTS as _R
_SEL = _pd.read_csv(os.path.join(_R, 'species_shoulder.tsv'), sep='\t').set_index('library').library_id   # somatic library used in Fig. 4c
from hicps.core import load_library, ps_all_pixels

out = open(os.path.join(RESULTS, 'anther_sensitivity.txt'), 'w')
def say(s): print(s, flush=True); out.write(s + '\n')
PAIRS = (('R_corymbosa', _SEL['corymbosa']), ('R_radicans', _SEL['radicans']), ('R_cephalotes', _SEL['cephalotes']))
SC = (0.5, 0.75, 1, 1.25)
for sp, som in PAIRS:
    ch = load_raw(sp); bg, _ = trans_background(sp, ch); Lm = np.mean([c['N'] * RES for c in ch]); sch = load_library(som)
    say(f'== {sp}  mean chromosome length {Lm / 1e6:.1f} Mb')
    ratios = []
    for hf in (0.2, 0.3, 0.4):
        for nb in (60, 80, 120):
            for w in (0.2, 0.3, 0.4):
                so = slope_max(*ps_all_pixels(sch, nb=nb), hi=hf * Lm, w=w)
                an = [slope_max(*ps_corrected(ch, bg * f, nb=nb), hi=hf * Lm, w=w) for f in SC]
                if hf == 0.4: ratios += [max(a / so, so / a) for a, f in zip(an, SC) if f in (0.75, 1)]
                say(f'  search 0.3 Mb-{hf}L, {nb} bins, window {w}: somatic {so:.2f} | anther, background x ' + ', '.join(f'{f}: {a:.2f}' for f, a in zip(SC, an)))
    say(f'  anther/somatic ratio (search to 0.4L, background x0.75 and x1): {sum(r <= 1.3 for r in ratios)} of {len(ratios)} within 1.3-fold; max {max(ratios):.2f}')
say('\n== VC_SQRT instead of raw anther contacts (80 bins, window 0.3, search to 0.4L), VC_SQRT rescaled to the raw total')
for sp, som in PAIRS[:2]:
    raw = load_raw(sp); bg, _ = trans_background(sp, raw); Lm = np.mean([c['N'] * RES for c in raw])
    vc = []
    for c in raw:
        z = np.load(os.path.join(DATA, sp, 'vcsqrt', c['name'] + '.npz')); vc.append(dict(S=z['S'][:c['N']].astype(float), pairs=c['pairs'], N=c['N']))
    f = sum(c['S'].sum() for c in raw) / sum(c['S'].sum() for c in vc)
    for c in vc: c['S'] = c['S'] * f
    for m in (0.75, 1.0):
        say(f'{sp} background x{m}: raw {slope_max(*ps_corrected(raw, bg * m), hi=0.4 * Lm):.2f}, VC_SQRT {slope_max(*ps_corrected(vc, bg * m), hi=0.4 * Lm):.2f}')
out.close()
