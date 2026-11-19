#!/bin/bash
# Reproduce all Hi-C results, figures and tables from the contact summaries in data/ (about 3 minutes on one core).
set -euo pipefail
cd "$(dirname "$0")"
python3 analysis/01_shoulders.py
python3 analysis/02_fig4c.py
python3 analysis/03_robustness.py
python3 analysis/04_replicates.py
python3 analysis/05_tyba_spacing.py
python3 anther/anther_hic.py
python3 anther/anther_sensitivity.py
python3 anther/anther_reeval_fig.py
python3 analysis/06_suppl_fig_hic.py
python3 analysis/07_rugosa_diagnostic.py
python3 analysis/08_fig4.py
python3 analysis/09_supplementary_table.py
python3 response_letter/replicate_fig.py
python3 response_letter/published_pipeline.py
echo "done: results/"
