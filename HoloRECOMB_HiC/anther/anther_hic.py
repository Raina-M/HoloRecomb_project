"""Anther (meiocyte-enriched) Arima Hi-C: library QC, enrichment over random ligation and background-corrected P(s).

Input per species, in <DATA>/<species>/ (DATA defaults to ../data/anther, env HICPS_ANTHER_DATA):
  raw/<chr>.npz        raw (NONE) intrachromosomal contacts summed per 5-kb distance (S) and chromosome length (L)
                       (extraction/aggregate_raw.py)
  gw/<c1>__<c2>.txt    raw contacts at 500 kb for every chromosome pair (extraction/dump_anther.sh)
  juicer_stats.txt     Juicer statistics block of inter_30.hic (extraction/hic_header.py --stats)
Outputs (results/): anther_QC.tsv, anther_stats.txt, anther_curves.pkl"""
import glob, os, re, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from hicps.core import local_slope, shoulder

RES = 5000
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('HICPS_ANTHER_DATA', os.path.join(HERE, '..', 'data', 'anther'))
RESULTS = os.path.join(HERE, '..', 'results', 'anther'); os.makedirs(RESULTS, exist_ok=True)
SPECIES = ('R_cephalotes', 'R_corymbosa', 'R_radicans')


def load_raw(sp):
    """Chromosomes of the anther library: raw contacts per distance; pixel pairs per distance = n - d (all bins)."""
    ch = []
    for f in sorted(glob.glob(os.path.join(DATA, sp, 'raw', '*.npz'))):
        z = np.load(f); L = int(z['L']); n = L // RES + 1
        ch.append(dict(name=os.path.basename(f)[:-4], S=z['S'][:n].astype(float), pairs=np.maximum(n - np.arange(n), 0).astype(float), N=n))
    return ch


def trans_background(sp, ch):
    """Mean interchromosomal contacts per 5-kb pixel pair (random-ligation background)."""
    tot = 0.0
    for f in glob.glob(os.path.join(DATA, sp, 'gw', '*.txt')):
        a, b = os.path.basename(f)[:-4].split('__')
        if a == b: continue
        try: tot += np.loadtxt(f, ndmin=2)[:, 2].sum()
        except Exception: pass
    L = np.array([c['N'] * RES for c in ch], float); pairs = (L.sum() ** 2 - (L ** 2).sum()) / 2 / RES ** 2
    return tot / pairs, tot


def enrichment_by_distance(ch, bg, classes=((2e4, 1e5), (1e5, 1e6), (1e6, 1e7))):
    rows = []
    for lo, hi in classes:
        o = sum(c['S'][int(lo / RES):int(hi / RES)].sum() for c in ch); e = sum(c['pairs'][int(lo / RES):int(hi / RES)].sum() for c in ch) * bg
        rows.append((lo, hi, o, e, o / e))
    return rows


def ps_corrected(ch, bg=0.0, lo=1e4, hi=1e8, nb=80):
    """Mean contacts per pixel pair minus background, in nb log bins; only bins with positive values are returned."""
    e = np.logspace(np.log10(lo), np.log10(hi), nb + 1); m = np.sqrt(e[1:] * e[:-1]); S = np.zeros(nb); P = np.zeros(nb)
    for c in ch:
        d = np.arange(len(c['S'])) * RES; i = np.digitize(d, e) - 1; ok = (i >= 0) & (i < nb)
        np.add.at(S, i[ok], c['S'][ok]); np.add.at(P, i[ok], c['pairs'][ok])
    k = P > 0; y = S[k] / P[k] - bg; x = m[k]; ok = y > 0
    return x[ok], y[ok]


def slope_max(x, p, lo=3e5, hi=None, w=0.3):
    """Position (Mb) of the maximum local slope between lo and hi; used for profiles pooled over all chromosomes,
    where the chromosome-end trough is blurred by pooling chromosomes of different lengths (hi = 0.4 x mean length)."""
    s = local_slope(x, p, w); k = (x >= lo) & (x <= hi) & np.isfinite(s); i = np.argmax(s[k]); return x[k][i] / 1e6


def juicer_qc(sp):
    t = open(os.path.join(DATA, sp, 'juicer_stats.txt')).read()
    num = lambda key: (lambda m: int(m.group(1).replace(',', '')) if m else np.nan)(re.search(re.escape(key) + r'\s*([\d,]+)', t))
    d = dict(species=sp.replace('R_', 'R. '), read_pairs=num('Sequenced Read Pairs:'), unique=num('Total Unique:'), duplicates=num('Total Duplicates:'),
             hic_contacts=num('Hi-C Contacts:'), inter=num('Inter-chromosomal:'), intra=num('Intra-chromosomal:'), lt500=num('<500BP:'),
             b500_5k=num('500BP-5kB:'), b5k_20k=num('5kB-20kB:'), gt20k=num('Long Range (>20Kb):'))
    d['duplicate_pct'] = 100 * d['duplicates'] / (d['unique'] + d['duplicates'])
    d['lt500_pct_of_contacts'] = 100 * d['lt500'] / d['hic_contacts']; d['gt20k_pct_of_contacts'] = 100 * d['gt20k'] / d['hic_contacts']
    return d


if __name__ == '__main__':
    out = []; curves = {}; qc = []; enr = []
    for sp in SPECIES:
        ch = load_raw(sp); bg, trans = trans_background(sp, ch); qc.append(juicer_qc(sp))
        out.append(f'== {sp}: trans contacts {trans / 1e6:.2f} M; background per 5-kb pixel pair (from trans) {bg:.4f}')
        row = dict(species=sp.replace('R_', 'R. '), background_per_pixel_pair=bg, trans_contacts=trans)
        for lo, hi, o, e, r in enrichment_by_distance(ch, bg):
            row.update({f'observed_{lo / 1e6:g}_{hi / 1e6:g}Mb': o, f'expected_{lo / 1e6:g}_{hi / 1e6:g}Mb': e})
            out.append(f'   cis {lo / 1e6:g}-{hi / 1e6:g} Mb: observed {o / 1e6:.2f} M, expected from random ligation {e / 1e6:.2f} M, ratio {r:.1f}')
        Lm = np.mean([c['N'] * RES for c in ch])
        for corr in (True,):              # background-corrected profiles only
            b = bg
            res = [shoulder([c], c['N'] * RES, curve=ps_corrected([c], b)) for c in ch]
            x, p = ps_corrected(ch, b); curves[(sp, corr)] = (x, p)
            pooled = slope_max(x, p, hi=0.4 * Lm)
            row.update(n_resolved=int(np.isfinite(res).sum()), n_chr=len(res), shoulder_pooled_Mb=pooled)
            out.append('   shoulder per chromosome (background-corrected): ' + ', '.join('nr' if np.isnan(v) else f'{v:.2f}' for v in res)
                       + f'; resolved {np.isfinite(res).sum()}/{len(res)}; median {np.nanmedian(res) if np.isfinite(res).any() else float("nan"):.2f} Mb; pooled slope maximum (0.3 Mb to 0.4 x mean length) {pooled:.2f} Mb')
        enr.append(row)
    pd.DataFrame(enr).to_csv(os.path.join(RESULTS, 'anther_enrichment.tsv'), sep='\t', index=False)
    print('\n'.join(out)); open(os.path.join(RESULTS, 'anther_stats.txt'), 'w').write('\n'.join(out) + '\n')
    pd.DataFrame(qc).to_csv(os.path.join(RESULTS, 'anther_QC.tsv'), sep='\t', index=False)
    pickle.dump(curves, open(os.path.join(RESULTS, 'anther_curves.pkl'), 'wb'))
