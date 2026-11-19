#!/bin/bash
# Contact summaries of all leaf libraries listed in config/libraries.tsv.
# Usage: JUICER_TOOLS=/path/to/juicer_tools.jar extraction/dump_all.sh /path/to/HiC   (folder containing R_<species>/rep<n>/inter_30.hic)
set -euo pipefail
ROOT=$1; D=$(cd "$(dirname "$0")" && pwd)
tail -n +2 "$D/../config/libraries.tsv" | while IFS=$'\t' read -r LIB SP REP TISSUE PROT REF HIC STATUS NOTE TYBA; do
  N=0; [ "$LIB" = "R_pubera_rep1" ] && N=5          # R. pubera library 1: chromosomes 1-5 only (unplaced scaffolds in the header)
  "$D/dump_library.sh" "$ROOT/$HIC" "$LIB" VC_SQRT "$N"
done
