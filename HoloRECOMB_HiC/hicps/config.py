"""Library table, species order, plotting style and helpers shared by all analysis scripts."""
import os, re, logging, numpy as np, pandas as pd
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from .core import ROOT, RESULTS

LIBRARIES = pd.read_csv(os.path.join(ROOT, 'config', 'libraries.tsv'), sep='\t', keep_default_na=False)

# species colours and markers (circles: species with crossover maps; triangles: other species)
STY = {'alba': ('#17BECF', '^'), 'austrobrasiliensis': ('#6A3D9A', '^'), 'barbata': ('#7570B3', 'o'), 'breviuscula': ('#1B9E77', 'o'),
       'cephalotes': ('#D95F02', 'o'), 'ciliata': ('#3399FF', 'o'), 'colorata': ('#66A61E', 'o'), 'corymbosa': ('#7570B3', '^'),
       'filiformis': ('#1B9E77', '^'), 'gaudichaudii': ('#D95F02', '^'), 'holoschoenoides': ('#3399FF', '^'), 'nervosa': ('#E6AB02', 'o'),
       'pubera': ('#E7298A', 'o'), 'radicans': ('#E6AB02', '^'), 'ridleyi': ('#66A61E', '^'), 'riparia': ('#1F78B4', 'v'),
       'rugosa': ('#E7298A', '^'), 'tenerrima': ('#E31A1C', 'v'), 'tenuis': ('#A6761D', '^'), 'watsonii': ('#666666', '^')}
ORDER = list(STY)
LAB = lambda k: 'R. ' + k
FONT = {'font.family': ['Arial', 'Liberation Sans'], 'pdf.fonttype': 42, 'svg.fonttype': 'none'}


def pf(p):
    """P value as printed in the figures."""
    return '< 0.001' if p < 0.001 else f'= {p:.3f}' if p < 0.01 else f'= {p:.2f}'


def juicer_stats(path):
    """Numbers from the Juicer statistics block (inter_30.txt or the 'statistics' attribute of the .hic header)."""
    t = open(path).read() if os.path.exists(path) else ''
    g = lambda pat: (lambda m: int(m.group(1).replace(',', '')) if m else np.nan)(re.search(pat, t))
    return dict(read_pairs=g(r'Sequenced Read Pairs:\s+([\d,]+)'), unique=g(r'Total Unique:\s+([\d,]+)'),
                duplicates=g(r'Total Duplicates:\s+([\d,]+)'), complexity=g(r'Library Complexity Estimate\*?:\s+([\d,]+)'),
                hic_contacts=g(r'Hi-C Contacts:\s+([\d,]+)'), inter=g(r'Inter-chromosomal:\s+([\d,]+)'),
                intra=g(r'Intra-chromosomal:\s+([\d,]+)'), lt500=g(r'<500BP:\s+([\d,]+)'), gt20k=g(r'Long Range \(>20Kb\):\s+([\d,]+)'),
                unmapped_pct=(lambda m: float(m.group(1)) if m else np.nan)(re.search(r'One or both reads unmapped:\s+[\d,]+\s+\(([\d.]+)%', t)))


def within_slope(df, x, y, group):
    """Pooled within-group slope of log10 y on log10 x after centring both on group means; returns slope, s.e., P."""
    import scipy.stats as ss
    d = df[[x, y, group]].dropna().copy()
    for v in (x, y): d[v + '_c'] = np.log10(d[v]) - np.log10(d[v]).groupby(d[group]).transform('mean')
    b = (d[x + '_c'] * d[y + '_c']).sum() / (d[x + '_c'] ** 2).sum(); dfree = len(d) - 1 - d[group].nunique()
    se = np.sqrt(((d[y + '_c'] - b * d[x + '_c']) ** 2).sum() / dfree / (d[x + '_c'] ** 2).sum())
    return b, se, 2 * ss.t.sf(abs(b / se), dfree)


def results(*p):
    return os.path.join(RESULTS, *p)
