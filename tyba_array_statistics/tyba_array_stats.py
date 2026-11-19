#!/usr/bin/env python3
"""Tyba array statistics for Supplementary Table [TybaStats] (revision, point R1.3).
Input: Suppl_Dataset_Fig2a/<species>/<species>_Tyba_arrays.gff3 and *_chrs.fasta.fai
(from Suppl_Dataset_3_recombination_tracks.zip).
Records separated by <= MERGE bp on the same chromosome are merged into one array.
Spacing = distance between midpoints of consecutive arrays on the same chromosome.
Occupancy = fraction of non-overlapping 1-Mb windows containing >= 1 array midpoint.
Usage: python3 tyba_array_stats.py <Suppl_Dataset_Fig2a dir> [merge_bp=5000] > out.tsv
"""
import sys, glob, os, re, statistics as st
from collections import defaultdict
root = sys.argv[1]; MERGE = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
WIN = 1_000_000
print("\t".join(["species","haplotype","assembly_Mb","TideCluster_records","arrays_merged",
  "arrays_per_Mb","mean_array_kb","median_array_kb","max_array_kb","Tyba_Mb","pct_assembly",
  "median_spacing_kb","IQR_spacing_kb","windows_1Mb_with_Tyba_pct","largest_Tyba_free_Mb"]))
for gff in sorted(glob.glob(os.path.join(root, "*", "*_Tyba_arrays.gff3"))):
    d = os.path.dirname(gff); sp = os.path.basename(d)
    fai = glob.glob(os.path.join(d, "*_chrs.fasta.fai"))[0]
    hap = re.search(r"_(hap\d)_", fai).group(1)
    L = {l.split()[0]: int(l.split()[1]) for l in open(fai)}
    by = defaultdict(list); nrec = 0
    for l in open(gff):
        r = l.rstrip("\n").split("\t")
        if l.startswith("#") or len(r) < 5: continue
        c = re.sub(r"^HiC_scaffold_(\d+)", r"Chr\1", r[0])  # R. nervosa seqid harmonisation
        if c not in L: sys.exit(f"{sp}: seqid {r[0]} not in {fai}")
        by[c].append([int(r[3]), int(r[4])]); nrec += 1
    arrays, sp_, occ, free = [], [], 0, []
    for c, v in by.items():
        v.sort(); m = [v[0][:]]
        for s, e in v[1:]:
            if s - m[-1][1] - 1 <= MERGE: m[-1][1] = max(m[-1][1], e)
            else: m.append([s, e])
        arrays += m
        mids = [(s + e) / 2 for s, e in m]
        sp_ += [mids[i+1] - mids[i] for i in range(len(mids) - 1)]
        occ += len({int(x // WIN) for x in mids})
        gaps = [m[0][0] - 1] + [m[i+1][0] - m[i][1] - 1 for i in range(len(m) - 1)] + [L[c] - m[-1][1]]
        free.append(max(gaps))
    G = sum(L.values()); nwin = sum(-(-x // WIN) for x in L.values())
    lens = [e - s + 1 for s, e in arrays]; q = st.quantiles(sp_, n=4)
    print("\t".join(map(str, [sp.replace("_", ". ", 1), hap, f"{G/1e6:.1f}", nrec, len(arrays),
      f"{len(arrays)/(G/1e6):.2f}", f"{st.mean(lens)/1e3:.1f}", f"{st.median(lens)/1e3:.1f}",
      f"{max(lens)/1e3:.1f}", f"{sum(lens)/1e6:.2f}", f"{100*sum(lens)/G:.1f}",
      f"{st.median(sp_)/1e3:.0f}", f"{q[0]/1e3:.0f}-{q[2]/1e3:.0f}",
      f"{100*occ/nwin:.0f}", f"{max(free)/1e6:.1f}"])))
