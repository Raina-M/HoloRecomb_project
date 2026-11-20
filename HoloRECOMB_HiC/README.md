# HoloRECOMB Hi-C: contact-domain size in holocentric *Rhynchospora*

Code for the Hi-C contact-decay analyses in Zhang, Steckenborn, Castellani *et al.*, "A holocentric pangenome links karyotype
evolution to meiotic recombination" (Marques Lab, Max Planck Institute for Plant Breeding Research). It produces Fig. 4 and its
statistics, Supplementary Fig. 12, Supplementary Fig. 13, Supplementary Dataset 4 and the Hi-C figures of
the response to referees.

## What is measured

For every chromosome, intrachromosomal contacts (Juicer `inter_30.hic`, MAPQ >= 30, VC_SQRT, 5-kb bins) are summed by genomic
distance s. The contact probability P(s) is the summed contacts divided by the number of valid pixel pairs at that distance,
so pixels without contacts are included. Its local slope is the regression slope of log10 P(s) on log10 s in windows of 0.3
log10 units. The **contact-domain size** of a chromosome is the position of the P(s) **shoulder**: the maximum of the local
slope between 300 kb and the slope minimum that precedes the increase in contacts between chromosome ends. That minimum is
searched between 3 Mb and 0.6 x chromosome length, and the shoulder is retained only if the slope falls by at least 0.1 from
it. A species value is the median over resolved chromosomes; a species with fewer than half of its chromosomes resolved is
reported as not resolved.

**Featureless null.** Each library is simulated without domains: a single power law is fitted to each chromosome, and
Poisson contacts are drawn with the same depth, chromosome length and valid bins. A library is flagged as low depth if any
simulated chromosome shows a shoulder.

**Library selection.** When a species has two libraries, the species-level analyses use the library with the most
intrachromosomal contacts beyond 20 kb among those that passed QC. The other library is used to assess reproducibility.

## Repository layout

```
config/libraries.tsv        one row per Hi-C library: species, replicate, tissue, reference, .hic path, QC status, Tyba annotation
hicps/                      core functions (P(s), local slope, shoulder, featureless null, thinning) and shared settings
extraction/                 .hic -> per-chromosome contact summaries (juicer_tools dump + aggregation)
analysis/01_shoulders.py    shoulder per chromosome for all libraries, featureless null, library selection
analysis/02_fig4c.py        Fig. 4c statistics (Spearman, log-log OLS, linear mixed model, within-species slope)
analysis/03_robustness.py   analysis settings, leave-one-species-out, depth, Poisson thinning
analysis/04_replicates.py   agreement between the two libraries of a species, HiCRep SCC, Fig. 4c with the other library
analysis/05_tyba_spacing.py Tyba inter-array spacing versus chromosome size and shoulder (Supplementary Fig. 13)
anther/                     Hi-C from anthers at meiotic stages: enrichment over random ligation, background-corrected P(s)
analysis/06_suppl_fig_hic.py Supplementary Fig. 12
analysis/07_rugosa_diagnostic.py R. rugosa: why the shoulder is not resolved
analysis/08_fig4.py         Fig. 4 (a-e), statistics and source data
analysis/09_supplementary_table.py Supplementary Dataset 4 (xlsx)
response_letter/            figures and tests used only in the response to referees
run_all.sh                  runs everything after extraction
```

## Running

```bash
pip install -r requirements.txt

# 1. extraction (once; needs the .hic files, Java and juicer_tools 2.20.00)
JUICER_TOOLS=/path/to/juicer_tools.jar extraction/dump_all.sh /path/to/HiC     # leaf libraries listed in config/libraries.tsv
JUICER_TOOLS=/path/to/juicer_tools.jar extraction/dump_anther.sh anther.hic R_corymbosa

# 2. analyses, figures and tables (about 2 minutes)
./run_all.sh
```

All outputs are written to `results/`. Random seeds are fixed: the featureless null of library i uses `default_rng(1000 + i)`,
where i is the library's row in `config/libraries.tsv`, so a library's null does not depend on the other libraries.

## Data

The `.hic` files and the per-chromosome contact summaries (`data/contact_summaries`, `data/anther`,
`data/somatic_extra`) are deposited with the manuscript data at EDMOND (https://doi.org/doi:10.17617/3.DGOJRQ). Unpack them
into `data/` to run the analyses without the extraction step. Small inputs are included in the repository:

| path | content |
|---|---|
| `data/tyba/tyba_arrays_merged10kb.tsv` | Tyba arrays (monomer runs < 10 kb apart merged) for all assemblies |
| `data/fig4_inputs/` | ZYP1 axis lengths, double-crossover and polymer-simulation source data, and micrographs for Fig. 4a, b, e |

Each contact summary (`<library>/<chromosome>.npz`) contains:

- `S`: summed contacts per distance (index = distance / 5 kb)
- `C`: non-zero pixels per distance
- `pairs`: valid pixel pairs per distance
- `cov`: coverage per bin
- `M`: 100-kb contact matrix
- `N`: number of 5-kb bins

`<library>.stats` is the Juicer statistics block from the `.hic` header.

## Requirements

Python >= 3.9 with numpy, scipy, pandas, statsmodels, matplotlib and openpyxl (`requirements.txt`). Extraction additionally
needs Java and juicer_tools 2.20.00. Figures use Arial when it is installed; otherwise Liberation Sans, which has the same
character widths.

## Licence

MIT (see LICENSE).
