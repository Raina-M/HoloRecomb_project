# Classification of CO landscapes (Extended Data Fig. 5)

Inputs: `<species>_CO_rates.txt` (CO rate in 100-kb steps; Supplementary Dataset 3). The *R. pubera* table is derived from the
F2 linkage map with `python3 pubera_rates_from_marey.py` (reads `merged_linkage_brev_ceph_cili_pube.txt`).

1. `python3 dcr_analysis.py`: distal-to-central ratio (mean CO rate at 5-20% of chromosome length from each end / central
   40%), all bipartitions of the seven species, permutation test, PCA and k-means clustering of chromosome-scaled profiles;
   writes `dcr_stats.txt`, `dcr_per_chromosome.tsv`, `dcr.pkl`.
2. `python3 dcr_figure.py`: Extended Data Fig. 5.
