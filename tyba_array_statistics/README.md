# Tyba array statistics (Supplementary Table [TybaStats])

`python3 tyba_array_stats.py <root> [merge_bp]` reads `<root>/<species>/<species>_Tyba_arrays.gff3` and
`<root>/<species>/*_chrs.fasta.fai` (the Tyba annotations of Fig. 2a; Supplementary Dataset 3) and reports, per species, the number
of arrays, arrays per Mb, mean and maximum array length, genome fraction, median centre-to-centre spacing, fraction of 1-Mb
windows with an array and the longest array-free interval. Annotation records closer than `merge_bp` (default 5,000) are merged.
`expected_output/` has the tables without merging and with 5-kb merging.
