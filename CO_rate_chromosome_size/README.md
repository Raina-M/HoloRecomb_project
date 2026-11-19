# CO rate and chromosome size (Fig. 3b, Supplementary Fig. [MixedModel])

Input: `<species>_CO_num_per_chr.txt` (chromosome, size in bp, CO count), 39 chromosomes of seven species.

1. `python3 r39_analysis.py`: least squares and linear mixed model (species as random intercept) of log10 CO rate on log10
   chromosome size, pooled within-species slope, per-species Spearman correlations; writes `r39_stats.txt`,
   `CO_rate_per_chromosome.tsv`, `Suppl_Table_MixedModel_per_species.tsv`, `MLH1_genomewide_points.tsv`, `fits.json`.
2. `python3 r39_panels.py`: Fig. 3b and Supplementary Fig. [MixedModel].
3. `python3 build_fig3_newB.py Fig_3.pdf`: places the revised panel b into the Fig. 3 layout.

`CO_rate_vs_chrsize.R` is the script of the submitted version. It divided the CO count of *R. breviuscula* by the chromosome
sizes of *R. barbata* and fitted the line including the two MLH1-based genome-wide values; both are corrected in
`r39_analysis.py` (see the response to referees, point 3.9).
