#!/bin/bash
# Per-chromosome contact summaries of one Juicer .hic file (intrachromosomal, 5-kb bins, VC_SQRT by default).
# Usage: extraction/dump_library.sh path/to/inter_30.hic LIBRARY_ID [NORM] [N_CHROMOSOMES]
#   NORM           VC_SQRT (default) or NONE for maps without normalisation vectors
#   N_CHROMOSOMES  use only the first N sequences of the header (for assemblies with unplaced scaffolds)
# Output: data/contact_summaries/LIBRARY_ID/<chromosome>.npz and data/contact_summaries/LIBRARY_ID.stats
# Requires: juicer_tools 2.20.00 (set JUICER_TOOLS=/path/to/juicer_tools.jar), Java >= 11, Python 3 with numpy.
set -euo pipefail
HIC=$1; LIB=$2; NORM=${3:-VC_SQRT}; NMAX=${4:-0}; JT=${JUICER_TOOLS:?set JUICER_TOOLS to juicer_tools.jar}
D=$(cd "$(dirname "$0")" && pwd); OUT=$D/../data/contact_summaries/$LIB; mkdir -p "$OUT"
python3 "$D/hic_header.py" "$HIC" --stats > "$OUT.stats"
python3 "$D/hic_header.py" "$HIC" | { [ "$NMAX" -gt 0 ] && head -n "$NMAX" || cat; } | while read -r CHR LEN; do
  [ -s "$OUT/$CHR.npz" ] && continue
  java -Xmx4g -jar "$JT" dump observed "$NORM" "$HIC" "$CHR" "$CHR" BP 5000 2>/dev/null \
    | grep -E '^[0-9]+\s+[0-9]+\s+[-0-9.eE]+$' | python3 "$D/aggregate_contacts.py" - "$OUT/$CHR.tmp" >/dev/null
  mv "$OUT/$CHR.tmp.npz" "$OUT/$CHR.npz"; echo "$LIB $CHR"
done
