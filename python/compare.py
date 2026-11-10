"""Compares two k-Shape results (labels and centroids), up to a permutation-free match since both start
from the same initial labels.

  python compare.py --a prefixA --b prefixB --k 3 --L 5000 --D 6 [--truth X.truth]
"""
import argparse
import numpy as np


def ari(a, b):
    from math import comb
    n = len(a)
    ka, kb = a.max() + 1, b.max() + 1
    t = np.zeros((ka, kb), dtype=np.int64)
    for x, y in zip(a, b):
        t[x, y] += 1
    s = sum(comb(int(v), 2) for v in t.ravel())
    sa = sum(comb(int(v), 2) for v in t.sum(1))
    sb = sum(comb(int(v), 2) for v in t.sum(0))
    e = sa * sb / comb(n, 2)
    mx = (sa + sb) / 2
    return 1.0 if mx == e else (s - e) / (mx - e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--truth")
    a = ap.parse_args()
    la = np.fromfile(a.a + ".labels", dtype="<i4")
    lb = np.fromfile(a.b + ".labels", dtype="<i4")
    ca = np.fromfile(a.a + ".centroids", dtype="<f4").reshape(a.k, a.L, a.D)
    cb = np.fromfile(a.b + ".centroids", dtype="<f4").reshape(a.k, a.L, a.D)
    same = float(np.mean(la == lb))
    err = float(np.max(np.abs(ca - cb)))
    print(f"labels identical: {same * 100:.2f}%  ARI(a,b)={ari(la, lb):.4f}  max|centroid diff|={err:.3e}")
    if a.truth:
        t = np.fromfile(a.truth, dtype="<i4")
        print(f"ARI vs truth: a={ari(t, la):.4f} b={ari(t, lb):.4f}")


if __name__ == "__main__":
    main()
