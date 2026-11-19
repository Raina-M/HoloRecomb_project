# CO rates at rearrangement-derived chromosome ends of R. cephalotes (Extended Data Fig. 6)

Inputs: `R_cephalotes_CO_rates.txt`, `R_breviuscula_CO_rates.txt` (Supplementary Dataset 3) and
`ceph_brev_synteny_blocks_from_Fig2a.tsv` (DEEPSPACE synteny blocks >= 0.5 Mb between R. cephalotes and R. breviuscula).

1. `python3 ceph_fission_ends.py`: classifies the 18 chromosome ends as rearrangement-derived or conserved by synteny with
   R. breviuscula and compares terminal CO rates (one-sided Wilcoxon signed-rank test); writes `ceph_end_classes.tsv`, `stats.txt`.
2. `python3 ed_fig.py`: Extended Data Fig. 6.
3. `python3 ceph_end_sequence_features.py <tracks_dir>`: GC, Tyba, TE and telomeric-repeat content of the terminal regions;
   needs the R. cephalotes haplotype 1 assembly (`Rcephalotes.hap1.scaffolded.chr.fasta`, not included) and the feature tracks.
