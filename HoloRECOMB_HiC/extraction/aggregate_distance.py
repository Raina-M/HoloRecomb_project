"""Sum a juicer_tools dump of one chromosome (bin1 bin2 value at 5 kb, stdin) per distance; used for the anther libraries.
Writes S (summed contacts), C (non-zero pixels) and L (chromosome length). Usage: ... | python aggregate_distance.py out.npz LENGTH"""
import sys, math, numpy as np
out = sys.argv[1]; L = int(sys.argv[2]); res = 5000; n = L // res + 2
S = np.zeros(n); C = np.zeros(n, dtype=np.int64)
for line in sys.stdin:
    p = line.split()
    if len(p) != 3 or not p[0].isdigit(): continue
    v = float(p[2])
    if not math.isfinite(v) or v <= 0: continue
    d = abs(int(p[1]) - int(p[0])) // res; S[d] += v; C[d] += 1
np.savez_compressed(out, S=S, C=C, L=L)
