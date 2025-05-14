#!/usr/bin/env python3
import sys
import csv
import math
import cmath
import argparse
import time
from itertools import product
import numpy as np
from collections import Counter

# ────────────────────────────────────────────────────────────────────────────────
# FASTA parser
# ────────────────────────────────────────────────────────────────────────────────
def parse_fasta(handle):
    """Yield (id, seq) for each record in FASTA."""
    header = None
    seq_chunks = []
    for line in handle:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header:
                yield header, "".join(seq_chunks)
            header = line[1:].split()[0]
            seq_chunks = []
        else:
            seq_chunks.append(line)
    if header:
        yield header, "".join(seq_chunks)

# ────────────────────────────────────────────────────────────────────────────────
# 1) Basic features
# ────────────────────────────────────────────────────────────────────────────────
def basic_features(seq):
    seq = seq.upper()
    length = len(seq)
    counts = {nt: seq.count(nt) for nt in ("A", "C", "G", "T")}
    counts["N"] = seq.count("N")
    gc = counts["G"] + counts["C"]
    at = counts["A"] + counts["T"]
    return {
        "length": length,
        "gc_percent":   (gc / length * 100) if length else 0.0,
        "at_percent":   (at / length * 100) if length else 0.0,
        **{f"{nt}_count": c for nt, c in counts.items()},
    }

# ────────────────────────────────────────────────────────────────────────────────
# 2) k-mer frequencies
# ────────────────────────────────────────────────────────────────────────────────
def kmer_freq(seq, k=2):
    s = seq.upper()
    counts = Counter(s[i:i+k] for i in range(len(s)-k+1))
    total = float(sum(counts.values())) or 1.
    freq = {
        f"kmer_{kmer_str}": counts.get(kmer_str, 0) / total
        for kmer_tuple in product("ACGT", repeat=k)
        for kmer_str in ["".join(kmer_tuple)]
    }
    return freq

# ────────────────────────────────────────────────────────────────────────────────
# 3) Pseudo-k-nucleotide composition (PseKNC)
#    following Chou’s original formulation with a simple numeric mapping
# ────────────────────────────────────────────────────────────────────────────────
_nt2num = {"A":1.0, "C":2.0, "G":3.0, "T":4.0}
def pseknc(seq, k=2, lamda=3, w=0.05):
    seq = seq.upper()
    L = len(seq)
    # 3.1 k-mer freqs
    freqs = kmer_freq(seq, k)
    # 3.2 sequence‐order correlation factors θ_j for j=1..lamda
    thetas = []
    for j in range(1, lamda+1):
        tot = 0.0
        count = 0
        for i in range(L - j):
            a, b = seq[i], seq[i+j]
            if a in _nt2num and b in _nt2num:
                diff = _nt2num[a] - _nt2num[b]
                tot += diff * diff
                count += 1
        thetas.append( (tot/count) if count else 0.0 )
    # 3.3 normalization denominator
    sum_freq = sum(freqs.values())
    D = sum_freq + w * sum(thetas)
    # 3.4 build final feature vector
    out = {}
    for kmer, f in freqs.items():
        out[f"pse_{kmer}"] = f / D
    for j, theta in enumerate(thetas, start=1):
        out[f"pse_theta_{j}"] = (w * theta) / D
    return out

# ────────────────────────────────────────────────────────────────────────────────
# 4) Spectral features via manual DFT
# ────────────────────────────────────────────────────────────────────────────────
_num_map = {"A": -1.5, "C":  0.5, "G": -0.5, "T":  1.5}
def dft_spectral_fft(seq):
    x = np.fromiter((_num_map.get(nt,0.0) for nt in seq.upper()), dtype=float)
    N = x.size
    if N < 2:
        return {"psd_max": 0.0, "psd_mean": 0.0, "psd_freq_max": 0}
    # real FFT: returns N//2+1 bins including DC
    X = np.fft.rfft(x)
    psd = (np.abs(X)**2) / N
    # skip the zero‐frequency bin if you prefer
    psd = psd[1:]
    idx = int(psd.argmax()) + 1
    return {
        "psd_max": float(psd.max()),
        "psd_mean": float(psd.mean()),
        "psd_freq_max": idx
    }

# ────────────────────────────────────────────────────────────────────────────────
# 5) Chaos-Game Representation (CGR) occupancy
# ────────────────────────────────────────────────────────────────────────────────
_corners = {"A":(0,0), "C":(0,1), "G":(1,1), "T":(1,0)}
def cgr_grid(seq, resolution=8):
    """Compute CGR occupancy grid (flattened)."""
    grid = [[0]*resolution for _ in range(resolution)]
    x, y = 0.5, 0.5
    for nt in seq.upper():
        if nt not in _corners: continue
        cx, cy = _corners[nt]
        x, y = (x+cx)/2.0, (y+cy)/2.0
        # map to cell index
        ix = min(int(x * resolution), resolution-1)
        iy = min(int(y * resolution), resolution-1)
        grid[iy][ix] += 1
    total = sum(sum(row) for row in grid) or 1
    out = {}
    for i in range(resolution):
        for j in range(resolution):
            out[f"cgr_{i}_{j}"] = grid[i][j] / total
    return out

# ────────────────────────────────────────────────────────────────────────────────
# 6) Shannon entropy of k-mer distribution
# ────────────────────────────────────────────────────────────────────────────────
def shannon_entropy(seq, k=2):
    freqs = kmer_freq(seq, k)
    ent = 0.0
    for p in freqs.values():
        if p > 0:
            ent -= p * math.log(p, 2)
    return {"entropy_k"+str(k): ent}

# ────────────────────────────────────────────────────────────────────────────────
# 7) Simple complex-network stats over k-mer adjacency
# ────────────────────────────────────────────────────────────────────────────────
def network_stats(seq, k=2):
    seq = seq.upper()
    nodes = set()
    edges = set()
    for i in range(len(seq) - k):
        k1 = seq[i:i+k]
        k2 = seq[i+1:i+1+k]
        if len(k1)==k and len(k2)==k:
            nodes.add(k1); nodes.add(k2)
            e = tuple(sorted((k1,k2)))
            edges.add(e)
    n = len(nodes)
    E = len(edges)
    avg_deg = (2*E / n) if n else 0.0
    density = (2*E / (n*(n-1))) if n>1 else 0.0
    return {"net_nodes": n, "net_edges": E,
            "net_avg_degree": avg_deg, "net_density": density}

# ────────────────────────────────────────────────────────────────────────────────
# Main: read FASTA, compute all, write CSV
# ────────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("fasta", help="input FASTA")
    p.add_argument("-o","--out", default=sys.stdout, type=argparse.FileType("w"),
                   help="output CSV")
    p.add_argument("--kmer_k", type=int,   default=2)
    p.add_argument("--pse_lambda", type=int, default=3)
    p.add_argument("--pse_w",      type=float, default=0.05)
    p.add_argument("--cgr_res",    type=int,   default=8)
    args = p.parse_args()

    writer = None
    with open(args.fasta) as fasta_file:
        for rid, seq in parse_fasta(fasta_file):
            feats = {}
            feats["id"] = rid
        feats.update(basic_features(seq))
        feats.update(kmer_freq(seq, k=args.kmer_k))
        feats.update(pseknc(seq, k=args.kmer_k,
                            lamda=args.pse_lambda, w=args.pse_w))
        feats.update(dft_spectral_fft(seq))
        feats.update(cgr_grid(seq, resolution=args.cgr_res))
        feats.update(shannon_entropy(seq, k=args.kmer_k))
        feats.update(network_stats(seq, k=args.kmer_k))
        if writer is None:
            writer = csv.DictWriter(args.out, fieldnames=list(feats.keys()))
            writer.writeheader()
        writer.writerow(feats)

if __name__ == "__main__":
    tic = time.time()
    
    main()
    toc = time.time()
    print("Elapsed time", toc - tic)