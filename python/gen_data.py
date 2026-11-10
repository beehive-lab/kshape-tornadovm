"""Writes k-Shape benchmark datasets and initial labels.

Dataset file: little-endian int32 m, L, D, then float32 (m, L, D). Init file: m int32 labels.
"""
import argparse
import numpy as np


def shapes(m, L, D, k, rng):
    """k shape classes (random smooth prototypes), each series randomly shifted, scaled and noised."""
    t = np.linspace(0, 1, L)
    protos = []
    for _ in range(k):
        p = np.zeros((L, D))
        for d in range(D):
            for _h in range(4):
                p[:, d] += rng.normal() * np.sin(2 * np.pi * (rng.integers(1, 6) * t + rng.random()))
        protos.append(p)
    y = rng.integers(0, k, size=m)
    x = np.empty((m, L, D))
    for i in range(m):
        s = rng.integers(-L // 10, L // 10 + 1)
        x[i] = np.roll(protos[y[i]], s, axis=0) * rng.uniform(0.5, 2.0) + rng.uniform(-1, 1) + 0.3 * rng.normal(size=(L, D))
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["uniform", "shapes"], default="shapes")
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--D", type=int, default=1)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="prefix: writes prefix.bin, prefix.init, prefix.truth")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    if a.kind == "uniform":
        x = rng.random((a.m, a.L, a.D))  # as the upstream CPU-vs-GPU notebook
        y = np.zeros(a.m, dtype=np.int32)
    else:
        x, y = shapes(a.m, a.L, a.D, a.k, rng)
    init = rng.integers(0, a.k, size=a.m).astype(np.int32)
    with open(a.out + ".bin", "wb") as f:
        np.array([a.m, a.L, a.D], dtype="<i4").tofile(f)
        x.astype("<f4").tofile(f)
    init.astype("<i4").tofile(a.out + ".init")
    np.asarray(y, dtype="<i4").tofile(a.out + ".truth")


if __name__ == "__main__":
    main()
