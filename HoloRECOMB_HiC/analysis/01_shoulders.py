"""Step 1. P(s) shoulder (contact-domain size) for every chromosome of every Hi-C library, a featureless-null control for
each library, and selection of one library per species (the deepest library that passed QC) for the species-level analyses.

Inputs : config/libraries.tsv; data/contact_summaries/<library>/<chromosome>.npz and <library>.stats
Outputs: results/library_chromosome_shoulder.tsv  shoulder per chromosome, all libraries
         results/library_summary.tsv              depth, Juicer statistics, shoulder summary and null control per library
         results/chromosome_shoulder.tsv          per chromosome, selected library of each species
         results/species_shoulder.tsv             per species, selected library
         results/null_model.txt                   featureless-null results"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hicps import (LIBRARIES, ORDER, DATA, RES, load_library, shoulder, chrom_length, contacts_beyond, null_chroms,
                   juicer_stats, results)

NULL_REPS, NULL_REPS_FLAGGED = 3, 20
rows, crow, nrow = [], [], []
for i, lib in LIBRARIES.iterrows():
    ch = load_library(lib.library_id)
    sh = np.array([shoulder([c], chrom_length(c)) for c in ch], float); ok = np.isfinite(sh)
    crow += [dict(library_id=lib.library_id, species=lib.species, replicate=lib.replicate, chrom=c['name'], chr_Mb=chrom_length(c) / 1e6,
                  shoulder_Mb=v) for c, v in zip(ch, sh)]
    # featureless null: single power law per chromosome, same depth, length and valid bins (independent seed per library)
    rng = np.random.default_rng(1000 + i)
    nv = np.array([shoulder([c], chrom_length(o)) for _ in range(NULL_REPS) for c, o in zip(null_chroms(ch, rng), ch)], float)
    nrow += [dict(library_id=lib.library_id, rep=r, shoulder_Mb=v) for r, v in zip(np.repeat(range(NULL_REPS), len(ch)), nv)]
    rows.append(dict(library_id=lib.library_id, species=lib.species, replicate=lib.replicate, status=lib.status, note=lib.note,
                     n_chr=len(ch), n_resolved=int(ok.sum()), mean_chr_Mb=np.mean([chrom_length(c) for c in ch]) / 1e6,
                     shoulder_median_Mb=np.nanmedian(sh) if ok.any() else np.nan, shoulder_min_Mb=np.nanmin(sh) if ok.any() else np.nan,
                     shoulder_max_Mb=np.nanmax(sh) if ok.any() else np.nan, contacts_gt20kb_M=contacts_beyond(ch) / 1e6,
                     contacts_1_10Mb_per_chr_k=np.median([c['S'][int(1e6 / RES):int(1e7 / RES)].sum() for c in ch]) / 1e3,
                     null_with_shoulder=int(np.isfinite(nv).sum()), null_total=len(nv),
                     **juicer_stats(os.path.join(DATA, lib.library_id + '.stats'))))
    print(f'{lib.library_id:28s} {ok.sum():2d}/{len(ch):2d} resolved, median {rows[-1]["shoulder_median_Mb"]:.2f} Mb, '
          f'{rows[-1]["contacts_gt20kb_M"]:.2f} M contacts > 20 kb, null {np.isfinite(nv).sum()}/{len(nv)}', flush=True)
L = pd.DataFrame(rows); C = pd.DataFrame(crow); N = pd.DataFrame(nrow)
L['low_depth'] = L.null_with_shoulder > 0             # the featureless null produces a shoulder: estimate not distinguishable from depth artefact
# one library per species: deepest library (contacts > 20 kb) among those that passed QC
L['selected'] = False
for sp, g in L[L.status == 'ok'].groupby('species'):
    L.loc[g.contacts_gt20kb_M.idxmax(), 'selected'] = True
L['n_libraries'] = L.species.map(L[L.status == 'ok'].species.value_counts())
L.to_csv(results('library_summary.tsv'), sep='\t', index=False); C.to_csv(results('library_chromosome_shoulder.tsv'), sep='\t', index=False)

sel = L[L.selected].set_index('species').reindex([s for s in ORDER if s in set(L.species)])
cs = C[C.library_id.isin(sel.library_id)].copy(); cs.insert(0, 'library', cs.species); cs.to_csv(results('chromosome_shoulder.tsv'), sep='\t', index=False)
S = sel.reset_index().rename(columns={'species': 'library'})
S['resolved'] = S.n_resolved / S.n_chr >= 0.5         # species with fewer than half of the chromosomes resolved: not resolved
S.to_csv(results('species_shoulder.tsv'), sep='\t', index=False)

# extended null for libraries in which the null produced a shoulder
out = ['Featureless null (single power law; same depth, chromosome lengths and valid bins), 3 replicates per library',
       N.merge(L[['library_id', 'selected']]).groupby(['library_id']).shoulder_Mb.apply(lambda v: f'{v.notna().sum()}/{len(v)}').to_string()]
sn = N[N.library_id.isin(sel.library_id)]
out.append(f'selected libraries: null chromosomes with a shoulder {sn.shoulder_Mb.notna().sum()}/{len(sn)}; observed {cs.shoulder_Mb.notna().sum()}/{len(cs)}')
for lid in L[L.low_depth & L.selected].library_id:
    ch = load_library(lid); rng = np.random.default_rng(99)
    v = np.array([shoulder([c], chrom_length(o)) for _ in range(NULL_REPS_FLAGGED) for c, o in zip(null_chroms(ch, rng), ch)], float)
    out.append(f'{lid}, {NULL_REPS_FLAGGED} null replicates: shoulder in {np.isfinite(v).sum()}/{len(v)} simulated chromosomes'
               + (f'; median {np.nanmedian(v):.2f} Mb, range {np.nanmin(v):.2f}-{np.nanmax(v):.2f} Mb' if np.isfinite(v).any() else ''))
    L.loc[L.library_id == lid, 'null20'] = f'{np.isfinite(v).sum()}/{len(v)}'
open(results('null_model.txt'), 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
L.to_csv(results('library_summary.tsv'), sep='\t', index=False)
print(S[['library', 'library_id', 'n_resolved', 'n_chr', 'resolved', 'low_depth', 'mean_chr_Mb', 'shoulder_median_Mb', 'contacts_gt20kb_M']].round(2).to_string(index=False))
