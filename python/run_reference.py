"""Runs the upstream Python k-Shape (TheDatumOrg/kshape-python) on a dataset with fixed initial labels.

  python run_reference.py --kshape-src PATH --backend cpu|gpu --data X.bin --init X.init --k 3 [--repeat 5] [--out prefix]
"""
import argparse
import sys
import time
import types

import numpy as np


def load(path):
    with open(path, "rb") as f:
        m, L, D = np.fromfile(f, dtype="<i4", count=3)
        x = np.fromfile(f, dtype="<f4", count=m * L * D).reshape(m, L, D)
    return x.astype(np.float64)


def stub_sklearn():
    # the estimator classes only use sklearn for their base classes
    try:
        import sklearn.base  # noqa: F401
    except ImportError:
        base = types.ModuleType("sklearn.base")
        base.ClusterMixin = type("ClusterMixin", (), {})
        base.BaseEstimator = type("BaseEstimator", (), {})
        sk = types.ModuleType("sklearn")
        sk.base = base
        sys.modules["sklearn"] = sk
        sys.modules["sklearn.base"] = base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kshape-src", required=True)
    ap.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.path.insert(0, a.kshape_src)
    stub_sklearn()
    x = load(a.data)
    init = np.fromfile(a.init, dtype="<i4")
    m, L, D = x.shape
    iters = [0]

    if a.backend == "cpu":
        import multiprocessing
        # worker processes must inherit the patched modules
        multiprocessing.set_start_method("fork", force=True)
        import kshape.core as core
        core.randint = lambda lo, hi, size: init.astype(np.int64).copy()

        def fit():
            ks = core.KShapeClusteringCPU(n_clusters=a.k, max_iter=a.max_iter, n_jobs=a.n_jobs)
            ks.fit(x)
            return ks.labels_.astype(np.int32), ks.centroids_
    else:
        import torch
        import kshape.core_gpu as core

        def fake_randint(lo, hi, size, dtype=None, **kw):
            return torch.tensor(init, dtype=dtype or torch.int64)
        core.torch.randint = fake_randint

        def fit():
            ks = core.KShapeClusteringGPU(n_clusters=a.k, max_iter=a.max_iter)
            ks.fit(x)
            torch.cuda.synchronize()
            return ks.labels_.astype(np.int32), ks.centroids_.cpu().numpy()

    # empty clusters: reseed from the same fixed sequence as the Java version (EmptyPick)
    class EmptyPick:
        def __init__(self):
            self.state = 12345

        def __call__(self):
            self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
            return self.state % m
    pick = [EmptyPick()]
    if a.backend == "cpu":
        np.random.choice = lambda n, size: np.array([pick[0]()])
    else:
        core.torch.randperm = lambda n, **kw: torch.tensor([pick[0]()])

    # count iterations: shape extraction runs k*D times per iteration
    calls = [0]
    orig_extract = core._extract_shape

    def counting(*args, **kw):
        calls[0] += 1
        return orig_extract(*args, **kw)
    core._extract_shape = counting

    times = []
    for r in range(a.repeat):
        calls[0] = 0
        pick[0] = EmptyPick()
        t0 = time.time()
        labels, cents = fit()
        times.append(time.time() - t0)
        it = calls[0] // (a.k * D)
        print(f"[ref-{a.backend}] run {r}: {times[-1]:.4f}s iters={it}", flush=True)
    warm = float(np.mean(times[1:])) if len(times) > 1 else times[0]
    print(f"[ref-{a.backend}] first={times[0]:.4f}s warm_mean={warm:.4f}s iters={it} per_iter={warm / max(1, it):.5f}s")
    if a.out:
        labels.astype("<i4").tofile(a.out + ".labels")
        np.asarray(cents, dtype="<f4").tofile(a.out + ".centroids")


if __name__ == "__main__":
    main()
