# Telomere nearest-neighbour distances (Fig. 2d, Supplementary Fig. 8)

Inputs: `telomere_NND.tsv` (species, NND in µm, one row per telomere signal pair) and `cell_areas.tsv` (species, nucleus area in µm²).

- `python3 nnd_stats.py`: Kruskal-Wallis test and Dunn's tests with Benjamini-Hochberg correction on raw NND.
- `python3 panel_2d.py`: Fig. 2d, NND divided by the square root of the mean nucleus area of the species; linear mixed model of
  log normalised NND with CO landscape class as fixed effect and species as random intercept (`panel_2d_stats.txt`).
- `python3 suppl_fig7_raw.py`: raw NND in the layout of Fig. 2d (Supplementary Fig. 8).
- `build_fig2_newD.py`: places the new panel d into the Fig. 2 SVG layout.
- `telomere_NND_stats.R`: script of the submitted version.
