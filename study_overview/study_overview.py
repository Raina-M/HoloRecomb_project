"""Study-design overview figure (referee 3, m12). Species x data matrix (a) and analysis workflow (b).
Style as Fig. 4: Arial (Liberation Sans fallback for rendering), 6-6.5 pt text, 9-pt bold panel letters, 180 mm width."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': ['Arial', 'Liberation Sans'], 'font.size': 6.5, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.linewidth': 0.6,
                     'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic'})
MM = 1 / 25.4; W, H = 180, 129
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

# colour per data group
GEN, CHR, REC, CYT, MOD = '#33658A', '#2E7D32', '#C0562A', '#6D4C9A', '#6B6B6B'
LIGHT = {GEN: '#E3ECF3', CHR: '#E2EFE3', REC: '#F6E3DA', CYT: '#ECE6F3', MOD: '#ECECEC'}

def letter(x, y, t): ax.text(x, y, t, fontsize=9, fontweight='bold', va='top', ha='left')

# ---------------- a, species x data matrix ----------------
letter(1, 2, 'a')
SP = [  # name, 2n, haplotypes
    ('R. rugosa', '2n = 2x = 36', '1'), ('R. alba', '2n = 2x = 26', '1'), ('R. cephalotes', '2n = 2x = 18', '2'),
    ('R. ridleyi', '2n = 2x = 12', '2'), ('R. gaudichaudii', '2n = 3x = 15', '3'), ('R. watsonii', '2n = 2x = 10', '1'),
    ('R. radicans', '2n = 2x = 10', '1'), ('R. pubera', '2n = 2x = 10', '1'), ('R. breviuscula', '2n = 2x = 10', '2'),
    ('R. nervosa', '2n = 2x = 10', '2'), ('R. ciliata', '2n = 2x = 10', '2'), ('R. colorata', '2n = 2x = 10', '2'),
    ('R. tenerrima', '2n = 4x = 20', '2'), ('R. filiformis', '2n = 2x = 10', '2'), ('R. austrobrasiliensis', '2n = 6x = 18', '6'),
    ('R. tenuis', '2n = 2x = 4', '18'), ('R. riparia', '2n = 2x = 10', '2'), ('R. barbata', '2n = 2x = 10', '2'),
    ('R. holoschoenoides', '2n = 2x = 10', '2'), ('R. corymbosa', '2n = 2x = 18', '2')]
# columns: key, header, group colour
COLS = [('asm', 'Assembly', GEN), ('hicL', 'Leaf', CHR), ('hicR', 'Leaf, 2nd library', CHR),
        ('hicA', 'Anthers', CHR), ('co', 'CO map', REC), ('mlh', 'MLH1 foci', REC),
        ('cyt', 'Synapsis, telomeres', CYT), ('axis', 'Axis length', CYT), ('sim', 'Simulation', MOD)]
HICL = ['R. alba', 'R. austrobrasiliensis', 'R. barbata', 'R. breviuscula', 'R. cephalotes', 'R. ciliata', 'R. colorata', 'R. corymbosa',
        'R. filiformis', 'R. gaudichaudii', 'R. holoschoenoides', 'R. nervosa', 'R. pubera', 'R. radicans', 'R. ridleyi', 'R. riparia',
        'R. tenerrima', 'R. tenuis', 'R. watsonii']
OMNIC = ['R. alba', 'R. colorata', 'R. tenerrima']             # Omni-C; all other Hi-C libraries Arima Hi-C
D = {s[0]: {} for s in SP}
for s, *_ in SP: D[s]['asm'] = 'f'
for s in HICL: D[s]['hicL'] = 'f'
for s in OMNIC: D[s]['hicL'] = 'OC'
D['R. rugosa']['hicL'] = 'o'                                   # shoulder not resolved
for s in ['R. breviuscula', 'R. cephalotes', 'R. ciliata', 'R. corymbosa', 'R. holoschoenoides', 'R. nervosa', 'R. pubera', 'R. radicans',
          'R. tenuis', 'R. watsonii']: D[s]['hicR'] = 'f'
D['R. ridleyi']['hicR'] = 'o'                                  # library failed
D['R. rugosa']['hicR'] = 'o'                                   # shoulder not resolved
for s in ['R. corymbosa', 'R. radicans']: D[s]['hicA'] = 'f'
D['R. cephalotes']['hicA'] = 'o'                               # no enrichment beyond 1 Mb
for s in ['R. barbata', 'R. cephalotes', 'R. ciliata', 'R. colorata', 'R. nervosa']: D[s]['co'] = 'f'
D['R. breviuscula']['co'] = 'p'                                # published map
D['R. pubera']['co'] = 'F2'                                    # F2 genetic map
for s in ['R. breviuscula', 'R. pubera']: D[s]['mlh'] = 'f'
for s in ['R. breviuscula', 'R. cephalotes', 'R. pubera', 'R. nervosa', 'R. ciliata', 'R. colorata', 'R. holoschoenoides', 'R. corymbosa']: D[s]['cyt'] = 'f'
for s in ['R. breviuscula', 'R. cephalotes', 'R. ciliata', 'R. colorata', 'R. corymbosa', 'R. nervosa']: D[s]['axis'] = 'f'; D[s]['sim'] = 'f'

x0, xk, xh = 2.5, 29.0, 43.8      # species name, karyotype, haplotypes
xc0, dx = 57.0, 7.2               # first data column, spacing
y0, dy = 25.0, 4.55               # first row, row height
r = 1.25                          # dot radius
# column group bands and headers
groups = [('Genome', GEN, 0, 0), ('Chromatin (Hi-C)', CHR, 1, 3), ('Recombination', REC, 4, 5), ('Cytology', CYT, 6, 7), ('Model', MOD, 8, 8)]
for g, c, a_, b_ in groups:
    xa, xb = xc0 + a_ * dx - dx / 2 + 0.3, xc0 + b_ * dx + dx / 2 - 0.3
    ax.add_patch(FancyBboxPatch((xa, y0 - 3.2), xb - xa, len(SP) * dy + 1.0, boxstyle='round,pad=0,rounding_size=1.2',
                                fc=LIGHT[c], ec='none', zorder=0))
    ax.text((xa + xb) / 2, 4.2, g, ha='center', va='center', fontsize=6, fontweight='bold', color=c)
    ax.plot([xa + 0.6, xb - 0.6], [6.2, 6.2], color=c, lw=0.6)
for j, (k, h, c) in enumerate(COLS):
    ax.text(xc0 + j * dx - 0.6, 20.0, h, ha='left', va='bottom', fontsize=5.6, rotation=42, rotation_mode='anchor')
ax.text(x0, 18.6, 'Species', ha='left', va='center', fontsize=6, fontweight='bold')
ax.text(xk, 18.6, 'Karyotype', ha='left', va='center', fontsize=6, fontweight='bold')
ax.text(xh + 2.2, 18.6, 'Haplotypes', ha='center', va='center', fontsize=6, fontweight='bold')
ax.plot([x0, xc0 + 8 * dx + dx / 2], [21.0, 21.0], color='k', lw=0.6)
for i, (s, kar, hap) in enumerate(SP):
    y = y0 + i * dy
    if i % 2 == 1:
        ax.add_patch(plt.Rectangle((x0 - 1, y - dy / 2), xc0 - x0 - dx / 2 + 0.7, dy, fc='#F4F4F4', ec='none', zorder=0))
    ax.text(x0, y, s, ha='left', va='center', fontstyle='italic', fontsize=6.2)
    ax.text(xk, y, kar, ha='left', va='center', fontsize=6)
    ax.text(xh + 2.2, y, hap, ha='center', va='center', fontsize=6)
    for j, (k, h, c) in enumerate(COLS):
        v = D[s].get(k); x = xc0 + j * dx
        if v is None: continue
        if v == 'o':
            ax.add_patch(Circle((x, y), r, fc='white', ec=c, lw=0.8, zorder=3))
        else:
            ax.add_patch(Circle((x, y), r, fc=c, ec='none', zorder=3))
            if v in ('p', 'F2', 'OC'):
                ax.text(x + 1.6, y - 1.1, {'p': '†', 'F2': 'F$_2$', 'OC': '*'}[v], fontsize=5, ha='left', va='center', color=c)
yb = y0 + len(SP) * dy - 1.0
ax.plot([x0, xc0 + 8 * dx + dx / 2], [yb, yb], color='k', lw=0.6)
# totals
TOT = {'asm': '20', 'hicL': '20', 'hicR': '12', 'hicA': '3', 'co': '7', 'mlh': '2', 'cyt': '8', 'axis': '6', 'sim': '6'}
yt = yb + 2.6
ax.text(x0, yt, 'Total', ha='left', va='center', fontsize=6, fontweight='bold')
ax.text(xh + 2.2, yt, '56', ha='center', va='center', fontsize=6, fontweight='bold')
for j, (k, h, c) in enumerate(COLS):
    ax.text(xc0 + j * dx, yt, TOT[k], ha='center', va='center', fontsize=6, fontweight='bold', color=c)
# key
yk = yt + 4.6
ax.add_patch(Circle((x0 + 1.2, yk), r, fc='#4D4D4D', ec='none'))
ax.text(x0 + 3.2, yk, 'used in the analyses', va='center', fontsize=5.6)
ax.add_patch(Circle((x0 + 30.5, yk), r, fc='white', ec='#4D4D4D', lw=0.8))
ax.text(x0 + 32.5, yk, 'generated, excluded (see Methods)', va='center', fontsize=5.6)
ax.text(x0 + 66.0, yk, '†, published map;  F$_2$, F$_2$ genetic map', va='center', fontsize=5.6)
ax.text(x0, yk + 3.6, '*, Omni-C library; all other Hi-C libraries, Arima Hi-C', va='center', fontsize=5.6)

# ---------------- b, workflow ----------------
XB = 125.5
letter(XB - 4.0, 2, 'b')
def box(x, y, w, h, title, body, c, fill=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc=LIGHT[c] if fill else 'white',
                                ec=c, lw=0.7, zorder=2))
    ax.text(x + 1.6, y + 1.5, title, ha='left', va='top', fontsize=6, fontweight='bold', color=c, zorder=3)
    ax.text(x + 1.6, y + 4.9, body, ha='left', va='top', fontsize=5.4, linespacing=1.15, zorder=3)
def arrow(x1, y1, x2, y2, c='k'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=5, lw=0.6, color=c, zorder=1,
                                 shrinkA=0, shrinkB=0))
bw, bh = 52.0, 14.0
rows = [
    ('Genome assembly and synteny', 'PacBio HiFi, Hi-C scaffolding; GENESPACE,\nsyngraph, AGORA; fission and fusion\njunctions, $\\it{Tyba}$ and TE overlap (Fig. 1)', GEN),
    ('Single-gamete CO maps', 'Pollen nuclei, scRNA-seq and scATAC-seq;\nCO landscapes, distal-to-central ratio,\nCO rate and interference (Figs. 2a, 3)', REC),
    ('Meiotic cytology', 'ZYP1, ASY1 and MLH1 immunostaining,\ntelomere FISH; synapsis initiation,\ntelomere clustering, axis length (Figs. 2, 4a,b)', CYT),
    ('Hi-C contact decay', 'Leaf (Arima Hi-C, Omni-C) and anther\n(Arima Hi-C) libraries; P(s) shoulder, two\nlibraries per species, $\\it{Tyba}$ spacing (Fig. 4c,d)', CHR),
    ('Polymer simulations', 'Companion study; simulated against\nmeasured axis length per Mb (Fig. 4e)', MOD),
]
yb0, gap = 8.0, 3.2
ys = []
for i, (t, b, c) in enumerate(rows):
    y = yb0 + i * (bh + gap); ys.append(y)
    box(XB, y, bw, bh, t, b, c)
# reference genome feeds all downstream analyses
xr = XB - 2.2
ax.plot([xr, xr], [ys[0] + bh / 2, ys[3] + bh / 2], color=GEN, lw=0.6)
ax.plot([xr, XB], [ys[0] + bh / 2, ys[0] + bh / 2], color=GEN, lw=0.6)
for i in (1, 3):
    arrow(xr, ys[i] + bh / 2, XB, ys[i] + bh / 2, GEN)
# integration box
yi = ys[-1] + bh + 4.2
ax.add_patch(FancyBboxPatch((XB, yi), bw, 14.0, boxstyle='round,pad=0,rounding_size=1.2', fc='white', ec='k', lw=0.8, zorder=2))
ax.text(XB + 1.6, yi + 1.5, 'Integration', ha='left', va='top', fontsize=6, fontweight='bold')
ax.text(XB + 1.6, yi + 4.9, 'Chromosome number and size, CO number\nand density, axis length, contact-domain size;\nproposed framework (Fig. 5)', ha='left', va='top', fontsize=5.4, linespacing=1.15)
xs = XB + bw + 1.6
ax.plot([xs, xs], [ys[0] + bh / 2, ys[-1] + bh / 2], color='k', lw=0.6)
for y in ys: ax.plot([XB + bw, xs], [y + bh / 2, y + bh / 2], color='k', lw=0.6)
ax.plot([xs, xs], [ys[-1] + bh / 2, yi + 7.0], color='k', lw=0.6)
arrow(xs, yi + 7.0, XB + bw, yi + 7.0)

for ext in ('pdf', 'svg', 'png'):
    fig.savefig(f'Study_overview.{ext}', dpi=600 if ext == 'png' else None)
