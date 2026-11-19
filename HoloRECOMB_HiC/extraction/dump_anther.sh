#!/bin/bash
# Extract the anther Hi-C data used in anther/anther_hic.py and anther/anther_sensitivity.py from inter_30.hic.
# Usage: dump_anther.sh inter_30.hic SPECIES   (e.g. R_corymbosa)
# Output in ../data/anther/SPECIES/: raw/<chr>.npz, vcsqrt/<chr>.npz, gw/<c1>__<c2>.txt (500 kb, raw), juicer_stats.txt
# Sequences shorter than MIN_LEN (default 5 Mb) are skipped.
set -euo pipefail
HIC=$1; SP=$2; JT=${JUICER_TOOLS:?set JUICER_TOOLS to juicer_tools.jar}
D=$(cd "$(dirname "$0")" && pwd); O=$D/../data/anther/$SP; mkdir -p "$O/raw" "$O/vcsqrt" "$O/gw"
python3 "$D/hic_header.py" "$HIC" --stats > "$O/juicer_stats.txt"
mapfile -t CH < <(python3 "$D/hic_header.py" "$HIC" | awk -v m="${MIN_LEN:-5000000}" '$2>=m')
for line in "${CH[@]}"; do
  read -r CHR LEN <<< "$line"
  for NORM in NONE VC_SQRT; do
    sub=raw; [ "$NORM" = VC_SQRT ] && sub=vcsqrt
    [ -s "$O/$sub/$CHR.npz" ] && continue
    java -Xmx4g -jar "$JT" dump observed "$NORM" "$HIC" "$CHR" "$CHR" BP 5000 2>/dev/null | python3 "$D/aggregate_distance.py" "$O/$sub/$CHR" "$LEN"
  done
done
for ((i=0;i<${#CH[@]};i++)); do for ((j=i;j<${#CH[@]};j++)); do
  A=${CH[i]%% *}; B=${CH[j]%% *}
  java -Xmx2g -jar "$JT" dump observed NONE "$HIC" "$A" "$B" BP 500000 "$O/gw/${A}__${B}.txt" >/dev/null 2>&1
done; done
