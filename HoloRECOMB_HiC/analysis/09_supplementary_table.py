"""Step 9. Supplementary Dataset 4: (a) all Hi-C libraries from leaf tissue, (b) anther Hi-C libraries, (c) P(s) shoulder
of every chromosome of every leaf library. Run after steps 1-4 and anther/anther_hic.py.
Output: results/Supplementary_Table_HiC_libraries.xlsx"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'anther'))
from hicps import LIBRARIES, RES, load_library, ps_all_pixels, results
from anther_hic import slope_max, juicer_qc
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

L = pd.read_csv(results('library_summary.tsv'), sep='\t').merge(LIBRARIES[['library_id', 'tissue', 'protocol', 'reference']], on='library_id')
REP = pd.read_csv(results('replicate_summary.tsv'), sep='\t').set_index('library_id'); C = pd.read_csv(results('library_chromosome_shoulder.tsv'), sep='\t')
SP = pd.read_csv(results('species_shoulder.tsv'), sep='\t').set_index('library')
chrs = C.groupby('library_id').chr_Mb.agg(['min', 'max', 'mean'])
wb = Workbook(); F = Font(name='Arial', size=9); B = Font(name='Arial', size=9, bold=True); T = Font(name='Arial', size=10, bold=True)
I = Font(name='Arial', size=9, italic=True); HDR = PatternFill('solid', fgColor='DDE5F0')

def sheet(ws, title, header, rows, italic_cols=(0,), widths=None, notes=()):
    ws['A1'] = title; ws['A1'].font = T
    for j, h in enumerate(header, 1):
        c = ws.cell(3, j, h); c.font = B; c.fill = HDR; c.alignment = Alignment(wrap_text=True, vertical='top')
    for i, r in enumerate(rows, 4):
        for j, v in enumerate(r, 1):
            if isinstance(v, float) and not np.isfinite(v): v = None
            if isinstance(v, (np.integer,)): v = int(v)
            if isinstance(v, (np.floating,)): v = float(v)
            if isinstance(v, (np.bool_,)): v = bool(v)
            c = ws.cell(i, j, v.replace('{r}', str(i)) if isinstance(v, str) else v); c.font = I if j - 1 in italic_cols else F
    for j in range(1, len(header) + 1): ws.column_dimensions[get_column_letter(j)].width = (widths or {}).get(j, 13)
    ws.row_dimensions[3].height = 54; ws.freeze_panes = 'C4'
    n = 4 + len(rows) + 1
    for k, t in enumerate(notes): ws.cell(n + k, 1, t).font = F

# a: leaf libraries
H = ['Species', 'Library', 'Tissue', 'Library protocol', 'Reference', 'Status', 'Used in Fig. 4c', 'Sequenced read pairs', 'One or both reads unmapped (%)',
     'Unique read pairs', 'Duplicate read pairs', 'Duplicates (%)', 'Library complexity estimate', 'Hi-C contacts (MAPQ ≥ 30)', 'Intrachromosomal contacts',
     'Interchromosomal contacts', 'Intrachromosomal (%)', 'Contacts > 20 kb (Juicer)', 'Intrachromosomal contacts > 20 kb, VC_SQRT (millions)',
     'Contacts at 1-10 Mb per chromosome, median (thousands)', 'Chromosomes analysed', 'Chromosome size, min (Mb)', 'Chromosome size, max (Mb)',
     'Chromosome size, mean (Mb)', 'Chromosomes with resolved shoulder', 'Fraction resolved', 'P(s) shoulder, median (Mb)', 'P(s) shoulder, min (Mb)',
     'P(s) shoulder, max (Mb)', 'Featureless null: simulated chromosomes with a shoulder', 'Low depth (null produces a shoulder)', 'SCC with the other library, median',
     'SCC, min', 'SCC, max', 'Note']
rows = []
for r in L.itertuples():
    sp = 'R. ' + r.species; lib = f'{r.species} library {r.replicate[-1]}'; rp = REP.loc[r.library_id] if r.library_id in REP.index else None
    rows.append([sp, lib, r.tissue.capitalize(), r.protocol, r.reference, r.status, 'yes' if r.selected else 'no', r.read_pairs, r.unmapped_pct, r.unique, r.duplicates,
                 '=IFERROR(K{r}/(J{r}+K{r})*100,"")', r.complexity, r.hic_contacts, r.intra, r.inter, '=IFERROR(O{r}/N{r}*100,"")', r.gt20k, r.contacts_gt20kb_M,
                 r.contacts_1_10Mb_per_chr_k, r.n_chr, chrs.loc[r.library_id, 'min'], chrs.loc[r.library_id, 'max'], chrs.loc[r.library_id, 'mean'],
                 r.n_resolved, '=Y{r}/U{r}', r.shoulder_median_Mb, r.shoulder_min_Mb, r.shoulder_max_Mb, f'{r.null_with_shoulder} of {r.null_total}',
                 'yes' if r.low_depth else 'no', None if rp is None else rp.scc_median, None if rp is None else rp.scc_min, None if rp is None else rp.scc_max,
                 r.note if isinstance(r.note, str) else None])
ws = wb.active; ws.title = 'a Leaf libraries'
sheet(ws, 'Supplementary Dataset 4a. Hi-C libraries from leaf tissue (Arima Hi-C and Omni-C): library statistics, depth, P(s) shoulder and reproducibility', H, rows,
      widths={1: 18, 2: 22, 5: 34, 35: 40},
      notes=['Library protocol: Omni-C (Dovetail) for R. alba, R. colorata and R. tenerrima (one library per species); Arima Hi-C for all other libraries. '
             'Juicer statistics from inter_30.hic (header statistics block); empty cells, not reported in the header. Contacts > 20 kb (VC_SQRT) and the P(s) shoulder were '
             'calculated from intrachromosomal contacts at 5-kb resolution (Methods). Fig. 4c uses, for each species, the library with most intrachromosomal contacts > 20 kb '
             'among those that passed QC. Featureless null: single power law fitted to each chromosome, with the depth, chromosome length and valid bins of the library, '
             '3 simulations per chromosome. SCC, stratum-adjusted correlation coefficient (HiCRep) of 100-kb matrices at 0.1-5 Mb between the two libraries of a species.'])
# b: anther
E_ = pd.read_csv(results('anther', 'anther_enrichment.tsv'), sep='\t'); rows = []
for _, e in E_.iterrows():
    q = juicer_qc(e.species.replace('R. ', 'R_')); som = SP.loc[e.species[3:], 'library_id']; ch = load_library(som); x, p = ps_all_pixels(ch)
    s_som = slope_max(x, p, hi=0.4 * np.mean([c['N'] * RES for c in ch])); ok = e.species != 'R. cephalotes'
    rows.append([e.species, 'Anthers at meiotic stages', 'Arima Hi-C', q['read_pairs'], q['unique'], q['duplicates'], '=F{r}/(E{r}+F{r})*100', q['hic_contacts'], q['intra'], q['inter'],
                 q['lt500'], '=K{r}/H{r}*100', q['gt20k'], '=M{r}/H{r}*100', e.background_per_pixel_pair,
                 e['observed_0.02_0.1Mb'], e['expected_0.02_0.1Mb'], '=P{r}/Q{r}', e['observed_0.1_1Mb'], e['expected_0.1_1Mb'], '=S{r}/T{r}', e['observed_1_10Mb'], e['expected_1_10Mb'], '=V{r}/W{r}',
                 f'{e.n_resolved} of {e.n_chr}' if ok else 'not analysed (no enrichment beyond 1 Mb)', round(e.shoulder_pooled_Mb, 2) if ok else 'not analysed',
                 round(s_som, 2) if ok else 'not analysed', f'leaf library: {som}'])
Hb = ['Species', 'Tissue', 'Library protocol', 'Sequenced read pairs', 'Unique read pairs', 'Duplicate read pairs', 'Duplicates (%)', 'Hi-C contacts (MAPQ ≥ 30)',
      'Intrachromosomal contacts', 'Interchromosomal contacts', 'Contacts < 500 bp', 'Contacts < 500 bp (%)', 'Contacts > 20 kb', 'Contacts > 20 kb (%)',
      'Random-ligation background (contacts per 5-kb pixel pair)', 'Observed 20-100 kb', 'Expected 20-100 kb', 'Observed/expected 20-100 kb', 'Observed 0.1-1 Mb',
      'Expected 0.1-1 Mb', 'Observed/expected 0.1-1 Mb', 'Observed 1-10 Mb', 'Expected 1-10 Mb', 'Observed/expected 1-10 Mb',
      'Chromosomes with resolved shoulder (background-corrected)', 'P(s) shoulder, pooled anther, background-corrected (Mb)', 'P(s) shoulder, pooled leaf (Mb)', 'Note']
sheet(wb.create_sheet('b Anther libraries'), 'Supplementary Dataset 4b. Arima Hi-C libraries from anthers at meiotic stages: library statistics and enrichment over random ligation',
      Hb, rows, widths={1: 16, 2: 22, 25: 22, 28: 30},
      notes=['Expected contacts: random-ligation background (interchromosomal contacts per pixel pair) multiplied by the number of intrachromosomal pixel pairs in each distance class. '
             'Pooled shoulder: maximum of the local slope of P(s) between 0.3 Mb and 0.4 x mean chromosome length, all chromosomes pooled.'])
# c: per chromosome
rows = [['R. ' + r.species, f'{r.species} library {r.replicate[-1]}', r.chrom, r.chr_Mb, r.shoulder_Mb if np.isfinite(r.shoulder_Mb) else 'not resolved',
         'yes' if r.library_id in set(L[L.selected].library_id) else 'no'] for r in C.itertuples()]
sheet(wb.create_sheet('c Per chromosome'), 'Supplementary Dataset 4c. P(s) shoulder position of every chromosome of every leaf library',
      ['Species', 'Library', 'Chromosome', 'Chromosome size (Mb)', 'P(s) shoulder (Mb)', 'Used in Fig. 4c'], rows, widths={1: 18, 2: 22, 3: 22})
wb.save(results('Supplementary_Table_HiC_libraries.xlsx')); print('saved')
