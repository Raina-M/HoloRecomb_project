# HoloRecomb_project

Custom scripts for Zhang, Steckenborn, Castellani *et al.*, "A holocentric pangenome links karyotype evolution to meiotic
recombination" (Marques Lab, Max Planck Institute for Plant Breeding Research, Cologne). The reference will be added once the
DOI is available. Sequencing data, assemblies, annotations and the Hi-C contact summaries are deposited as described in the
Data availability statement of the article.

Each folder contains its scripts and the small input tables they read, and has its own README. Scripts are run from inside
their folder and write their outputs there. Where present, `expected_output/` contains the outputs used in the article.

## Contents

| Folder | Analysis | Article item | Language |
|---|---|---|---|
| `synteny_DEEPSPACE/` | DEEPSPACE synteny and karyotype reconstruction with CO landscapes as tracks | Fig. 1a, Extended Data Fig. 1 | R |
| `deepspace_with_CO_landscape.R` | Earlier DEEPSPACE script of the submitted version | Fig. 1a (submitted version) | R |
| `break_interval_identification/` | Fission and fusion junctions between karyotypes | Supplementary Dataset 2 | Shell, R |
| `karyotype_simulation/` | Random fission and fusion simulation of chromosome-size symmetry (CVCL) | Extended Data Fig. 3 | Python |
| `random_simulation.py` | Earlier copy of `karyotype_simulation/random_simulation.py` | Extended Data Fig. 3 | Python |
| `tyba_array_statistics/` | Tyba array number, length, genome fraction and spacing | Supplementary Table [TybaStats] | Python |
| `telomere_NND/` | Telomere nearest-neighbour distances normalised by nucleus size | Fig. 2d, Supplementary Fig. 8 | Python, R |
| `CO_number/` | CO number per gamete and per meiocyte | Fig. 3a | R |
| `CO_rate_chromosome_size/` | CO rate against chromosome size, linear mixed model | Fig. 3b, Supplementary Fig. [MixedModel] | Python, R |
| `marey_maps/` | Marey maps | Fig. 3c, Supplementary Fig. 9 | R |
| `CO_landscape_classification/` | Distal-to-central CO rate ratio, PCA and clustering of CO landscapes | Extended Data Fig. 5 | Python |
| `Rcephalotes_fission_ends/` | CO rates at rearrangement-derived and conserved chromosome ends of *R. cephalotes* | Extended Data Fig. 6 | Python |
| `axis_length/` | ZYP1 axis length per nucleus and per chromosome (submitted version) | Fig. 4a, b | R |
| `HoloRECOMB_HiC/` | Hi-C contact decay: P(s) shoulder, contact-domain size, featureless null, library reproducibility, Tyba inter-array spacing, anther Hi-C; assembly of Fig. 4 | Fig. 4, Supplementary Fig. [HiC], Extended Data Fig. [TybaSpacing], Supplementary Table [HiCQC] | Python, Shell |
| `Ps_slope_calculation/` | Hi-C P(s) pipeline of the submitted version, replaced by `HoloRECOMB_HiC/` | Fig. 4c (submitted version) | Shell, R |
| `study_overview/` | Overview of species, data types and analyses | Study-design figure | Python |

Scripts written for the revision replace those of the submitted version where both are present. The submitted versions are
kept for transparency and are identified in the table and in the folder READMEs.

## Inputs not included

Some scripts need large files that are not in this repository: the genome assemblies and Tyba annotations
(`tyba_array_statistics/`, `Rcephalotes_fission_ends/ceph_end_sequence_features.py`), the per-species
`linkage_info_for_plot_*.txt` tables (`marey_maps/`), the `.hic` files or contact summaries (`HoloRECOMB_HiC/`), and local
paths to assemblies, annotations and recombination landscapes (`synteny_DEEPSPACE/`). The folder READMEs specify which files
are needed and where they are deposited.

## Requirements

Python >= 3.9 with numpy, scipy, pandas, statsmodels, scikit-learn, matplotlib and openpyxl (pymupdf, lxml and cairosvg for the
figure-assembly scripts). R >= 4.2 with ggplot2, dplyr, tidyverse, RColorBrewer, scales, gghalves, ggstatsplot and
LinkageMapView. The Hi-C extraction additionally needs Java and juicer_tools 2.20.00, and DEEPSPACE needs MCScanX.
