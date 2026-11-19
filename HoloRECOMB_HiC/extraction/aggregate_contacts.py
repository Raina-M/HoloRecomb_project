"""Aggregate a juicer_tools dump of one chromosome (bin1 bin2 value at 5 kb; file or '-' for stdin) into the per-chromosome
summary used by somatic/hicps.py: S (summed contacts per distance), C (non-zero pixels per distance), cov (bin coverage),
pairs (valid pixel pairs per distance: autocorrelation of bins with >= 1 contact), M (100-kb matrix), N (number of bins).
Usage: python aggregate_contacts.py dump.txt|- out.npz"""
import sys, numpy as np
fn, out = sys.argv[1], sys.argv[2]; res = 5000; cr = 20                     # 100 kb = 20 bins
f = sys.stdin if fn == '-' else open(fn)
def grow(a, n):
    if a.shape[0] < n: b = np.zeros((n,) + a.shape[1:], a.dtype); b[:a.shape[0]] = a; return b
    return a
S = np.zeros(1); C = np.zeros(1); cov = np.zeros(1); M = np.zeros((1, 1), np.float32)
while True:
    lines = [l for l in f.readlines(64_000_000)]
    if not lines: break
    rows = [l.split() for l in lines]; rows = [r for r in rows if len(r) == 3 and r[0].isdigit() and r[1].isdigit()]
    if not rows: continue
    a = np.array(rows, float); a = a[np.isfinite(a[:, 2]) & (a[:, 2] > 0)]
    if not len(a): continue
    i = (a[:, 0] // res).astype(np.int64); j = (a[:, 1] // res).astype(np.int64); v = a[:, 2]; d = np.abs(j - i)
    n = int(max(i.max(), j.max())) + 1
    S = grow(S, n); C = grow(C, n); cov = grow(cov, n)
    np.add.at(S, d, v); np.add.at(C, d, 1); np.add.at(cov, i, v); np.add.at(cov, j, np.where(i != j, v, 0))
    nc = n // cr + 1
    if M.shape[0] < nc: m2 = np.zeros((nc, nc), np.float32); m2[:M.shape[0], :M.shape[1]] = M; M = m2
    np.add.at(M, (i // cr, j // cr), v)
N = len(cov); valid = (cov > 0).astype(float)
L = 2 ** int(np.ceil(np.log2(2 * N))); fv = np.fft.rfft(valid, L); pairs = np.round(np.fft.irfft(fv * np.conj(fv), L)[:N])
np.savez_compressed(out, S=S[:N], C=C[:N], pairs=pairs, cov=cov, M=M, N=N)
print(out, N, int(C.sum()), round(S.sum()))
