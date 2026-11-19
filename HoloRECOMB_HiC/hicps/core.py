"""Core functions for the contact-decay (P(s)) analysis of Hi-C libraries.

Input per chromosome (written by extraction/aggregate_contacts.py): an .npz file with
  S      summed contacts per distance (in 5-kb bins, index = distance / 5 kb)
  C      number of non-zero pixels per distance
  pairs  number of valid pixel pairs per distance (autocorrelation of bins with >= 1 contact)
  cov    per-bin coverage;  M  100-kb matrix;  N  number of 5-kb bins
Libraries are folders <DATA>/<library>/<chromosome>.npz; DATA defaults to <repository>/data/contact_summaries
(environment variable HICPS_DATA)."""
import glob, os
import numpy as np

RES = 5000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('HICPS_DATA', os.path.join(ROOT, 'data', 'contact_summaries'))
RESULTS = os.environ.get('HICPS_RESULTS', os.path.join(ROOT, 'results')); os.makedirs(RESULTS, exist_ok=True)


def load_library(library, data=DATA):
    """List of chromosomes (dict with name, S, C, pairs, cov, M, N) of one library."""
    out = []
    for f in sorted(glob.glob(os.path.join(data, library, '*.npz'))):
        z = np.load(f)
        out.append(dict(name=os.path.basename(f)[:-4], S=z['S'], C=z['C'], pairs=z['pairs'], cov=z['cov'], M=z['M'], N=int(z['N'])))
    if not out: raise FileNotFoundError(f'no .npz files for {library} in {data}')
    return out


def chrom_length(c):
    return c['N'] * RES


def ps_all_pixels(chroms, lo=1e4, hi=1e8, nb=80):
    """P(s) = summed contacts / number of valid pixel pairs (pixels without contacts included), pooled over the
    chromosomes given, in nb logarithmic distance bins between lo and hi. Returns bin centres (bp) and P(s)."""
    e = np.logspace(np.log10(lo), np.log10(hi), nb + 1); m = np.sqrt(e[1:] * e[:-1]); S = np.zeros(nb); P = np.zeros(nb)
    for c in chroms:
        d = np.arange(len(c['S'])) * RES; i = np.digitize(d, e) - 1; ok = (i >= 0) & (i < nb)
        np.add.at(S, i[ok], c['S'][ok]); np.add.at(P, i[ok], c['pairs'][ok])
    k = (P > 0) & (S > 0)
    return m[k], S[k] / P[k]


def local_slope(x, p, w=0.3):
    """Slope of log10 P(s) on log10 s by linear regression in windows of w log10 units centred on each bin."""
    lx = np.log10(x); ly = np.log10(p); out = []
    for x0 in lx:
        k = np.abs(lx - x0) <= w / 2 + 1e-9          # tolerance: windows contain the same number of bins everywhere
        out.append(np.polyfit(lx[k], ly[k], 1)[0] if k.sum() >= 3 else np.nan)
    return np.array(out)


def shoulder(chroms, L, prom=0.1, tlo=3e6, thi=0.6, slo=3e5, nb=80, w=0.3, curve=None):
    """Position (Mb) of the P(s) shoulder: maximum of the local slope between slo and the slope minimum (trough) that
    precedes the chromosome-end increase. The trough is searched between tlo and thi x L. Returns NaN (not resolved) if
    the trough lies at the upper end of its interval or if the slope decreases by less than prom from shoulder to trough.
    curve: optional (x, P) to use instead of ps_all_pixels(chroms)."""
    x, p = curve if curve is not None else ps_all_pixels(chroms, nb=nb)
    s = local_slope(x, p, w); ok = np.isfinite(s)
    t = ok & (x >= tlo) & (x <= thi * L)
    if t.sum() < 3: return np.nan
    xt, st = x[t], s[t]; j = int(np.argmin(st))
    if j == len(st) - 1: return np.nan
    k = ok & (x >= slo) & (x < xt[j])
    if k.sum() < 3: return np.nan
    i = int(np.argmax(s[k]))
    if s[k][i] - st[j] < prom: return np.nan
    return x[k][i] / 1e6


def null_chroms(chroms, rng):
    """Featureless control: per chromosome, a single power law fitted to log10 P(s) (all valid pixel pairs) between
    50 kb and L/2; contacts per distance ~ Poisson(fitted P(s) x pairs); non-zero pixels ~ Binomial(pairs, 1 - exp(-P))."""
    out = []
    for c in chroms:
        n = len(c['S']); d = np.arange(n) * RES; pr = c['pairs'][:n]
        k = (d >= 5e4) & (d <= n * RES / 2) & (pr > 0) & (c['S'][:n] > 0)
        b = np.polyfit(np.log10(d[k]), np.log10(c['S'][:n][k] / pr[k]), 1)
        lam = np.zeros(n); lam[1:] = 10 ** np.polyval(b, np.log10(d[1:])); lam[0] = c['S'][0] / max(pr[0], 1)
        S = rng.poisson(lam * pr).astype(float); C = rng.binomial(pr.astype(np.int64), 1 - np.exp(-lam)).astype(float)
        S[C == 0] = 0; out.append(dict(S=S, C=C, pairs=pr, N=c['N']))
    return out


def thin(chroms, f, rng):
    """Poisson thinning of per-distance sums to a fraction f of the contacts."""
    return [dict(S=rng.poisson(np.clip(c['S'], 0, None) * f).astype(float), pairs=c['pairs'], N=c['N']) for c in chroms]


def contacts_beyond(chroms, d_bp=2e4):
    return sum(c['S'][int(d_bp / RES):].sum() for c in chroms)
