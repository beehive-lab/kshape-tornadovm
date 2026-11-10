# k-Shape on GPUs with TornadoVM

A Java implementation of the [k-Shape](https://www.paparrizos.org/papers/PaparrizosSIGMOD15.pdf) time-series clustering algorithm, accelerated on GPUs with [TornadoVM](https://github.com/beehive-lab/TornadoVM). It follows the NumPy version in [TheDatumOrg/kshape-python](https://github.com/TheDatumOrg/kshape-python) and gives the same clusters from the same initial labels.

k-Shape groups time series by shape. Two series are compared after sliding one against the other to the shift where they line up best (shape-based distance, computed for all shifts at once with an FFT). Each iteration then:

1. aligns every series to the centroid of its cluster;
2. computes each cluster's new centroid, the series that best matches all its aligned members (the top eigenvector of a matrix built from them);
3. moves every series to the cluster with the closest centroid;

and stops when no series moves.

## How it works

Each step runs for all series and clusters at once, as a few large GPU kernels, instead of one series or one (series, centroid) pair at a time:

| Step | Where | What runs |
|---|---|---|
| Data FFT | GPU, once per run | Radix-2 Stockham FFT of every series and dimension (length `N`, the power of two ≥ `2L-1`). The spectra stay on the device. |
| Align | GPU | FFT of the centroids, spectrum product of each series with its own centroid, inverse FFT, best shift per series (one workgroup per series), then shift with zero padding and z-normalise. |
| Shape matrices | GPU | For each (cluster, dimension), the smaller of the `n_j x n_j` Gram matrix `Y Yᵀ` and the `L x L` matrix `Yᵀ Y` of the z-normalised members, with 16 x 16 tiled kernels. |
| Top eigenvector | Host, all clusters in parallel | Lanczos with full reorthogonalisation; falls back to a dense tridiagonal solver (Householder reduction, QL eigenvalues, inverse iteration) when the Lanczos residual is not small enough. |
| Centroids | GPU, then host | Back-projection `Yᵀ u` in Gram mode and the dot products for the sign choice on the GPU; sign and z-normalisation on the host. |
| Assign | GPU | FFT of the new centroids, spectrum product of every series with every centroid summed over dimensions, inverse FFT, best correlation per pair. |

The shape-matrix step uses two identities that keep the result unchanged but shrink the eigenproblem: the z-normalised rows have zero mean, so `P S P = S = Yᵀ Y`, and the top eigenvector of `Yᵀ Y` is `Yᵀ u` for the top eigenvector `u` of `Y Yᵀ`. The eigenproblem therefore has size `min(n_j, L)` instead of `L`.

All kernels use the TornadoVM `KernelContext` API. The five task graphs (data, centroids, align, direction, assign) share their device buffers in one execution plan, so between steps only labels, centroids and a few small vectors move between host and device.

Centroids start at zero, as in the Python version. An empty cluster takes a series picked by a fixed pseudo-random sequence, one per dimension, so runs are reproducible.

## Requirements

* JDK 21 or newer
* TornadoVM 8.1.0, with `TORNADOVM_HOME` set (tested with the CUDA and Metal backends)
* For the benchmarks: Python 3 with NumPy (and PyTorch with CUDA for the PyTorch baselines), and a checkout of [kshape-python](https://github.com/TheDatumOrg/kshape-python)

## Build and run

```bash
./mvnw -q compile
```

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp target/classes org.beehive.kshape.KShapeBench --data data.bin --k 3 --init data.init --max-iter 100 --repeat 5 --out result
```

* `--data`: little-endian `int32` `m`, `L`, `D`, then `m*L*D` `float32` values in NumPy `(m, L, D)` order.
* `--init`: `m` `int32` initial labels (or `--random-init`).
* `--out`: writes `result.labels` (`m` `int32`) and `result.centroids` (`k*L*D` `float32`).

`python/gen_data.py` writes datasets in this format (`--kind shapes` for clustered series, `--kind uniform` for `np.random.rand` data).

From Java:

```java
try (KShapeTornado ks = new KShapeTornado(data, m, L, D, k)) {   // data in (m, L, D) order
    KShapeTornado.Result r = ks.fit(initLabels, false, 100, 42L);
    // r.labels, r.centroids (k, L, D), r.iterations
}
```

## Results

NVIDIA A100 80GB PCIe, 128-core Neoverse-N1 host. TornadoVM 8.1.0 (CUDA backend) on JDK 25; Python 3.11, NumPy 2.4.6, PyTorch 2.14.0+cu130.

Wall time of a full clustering run from the same initial labels:

* **NumPy**: `KShapeClusteringCPU` with `n_jobs=-1`.
* **PyTorch**: `KShapeClusteringGPU` from kshape-python.
* **Batched PyTorch**: `KShapeClusteringGPU` rewritten to process all series and clusters at once ([TheDatumOrg/kshape-python#21](https://github.com/TheDatumOrg/kshape-python/pull/21)), with the same smaller eigenproblem.
* **TornadoVM**: this project.

TornadoVM and batched PyTorch: mean of the runs after the first. NumPy and PyTorch: mean of the runs after the first, or a single run for the 4096-length datasets.

| Dataset | m x L x D, k | Iterations | NumPy | PyTorch | Batched PyTorch | TornadoVM | Same labels as NumPy |
|---|---|---:|---:|---:|---:|---:|---:|
| ucr-small (shapes) | 1000 x 256 x 1, 8 | 13 | 9.45 s | 61.35 s | 0.090 s | **0.071 s** | 100% |
| ucr-large (shapes) | 5000 x 512 x 1, 10 | 47 | 355.5 s | 1476.7 s | 1.376 s | **0.709 s** | 99.72% |
| multi (shapes) | 2000 x 1000 x 3, 5 | 8 | 36.84 s | 80.76 s | 0.313 s | **0.192 s** | 100% |
| notebook (uniform) | 500 x 5000 x 6, 3 | 5 (max) | 192.5 s | 31.47 s | 0.117 s | 0.173 s | 100% |
| readme-n512 (uniform) | 512 x 4096 x 1, 3 | 100 (max) | 489.5 s | 139.8 s | **0.871 s** | 0.991 s | 100% |
| readme-n4096 (uniform) | 4096 x 4096 x 1, 3 | 100 (max) | 1301.0 s | 986.3 s | **14.01 s** | 50.98 s | 100% |

"shapes" datasets have `k` random smooth prototypes, each series shifted, scaled, offset and noised. "uniform" is `np.random.rand` data with the sizes of the kshape-python benchmark notebook and of two points of its README benchmark chart.

The last column compares TornadoVM's labels with the NumPy version (float64). On ucr-large the float32 runs take a different number of iterations (NumPy 55, TornadoVM 47), because near-ties move a few series in late iterations. The ARI between the two labelings there is 0.996.

The first run in a process is slower: for TornadoVM it includes the JIT compilation of the kernels (ucr-small: 1.30 s instead of 0.071 s), and for batched PyTorch the CUDA initialisation (about 0.5 s).

Notes:

* The PyTorch version in kshape-python processes one series or one (series, centroid) pair at a time and synchronises with the host for each, so it is slower than NumPy for series shorter than a few thousand points. Most of the gap between it and TornadoVM comes from that; batched PyTorch closes most of it.
* Batched PyTorch is faster on the three uniform datasets. TornadoVM computes the eigenvectors on the host, while batched PyTorch runs `eigh` on the GPU. On `readme-n4096` each cluster has about 1365 members, and the host eigenvector step takes 41 of the 51 s.

## Reproducing the benchmarks

```bash
PY=python3 KSHAPE_SRC=/path/to/kshape-python BATCHED_SRC=/path/to/kshape-python-batched scripts/bench.sh results
```

The script generates each dataset, runs every version from the same initial labels, and compares the labels of each with TornadoVM's. `BATCHED_SRC` is optional and points at a checkout of the batched PyTorch branch. `SKIP_CPU=1` and `SKIP_GPU=1` skip the NumPy and PyTorch runs. `REF_REPEAT` (default 2) sets how many times they run. A configuration can also be given on the command line as `name:kind:m:L:D:k:max_iter:repeat`.

`python/run_reference.py` runs the Python versions with the given initial labels and the same empty-cluster re-seeding as the Java version; `python/compare.py` compares two runs.

## Apache IoTDB UDF

[Apache IoTDB](https://iotdb.apache.org) (2.0.10 and later) has a `cluster` UDF in its UDF library that splits a series into non-overlapping windows and clusters them, with `method=kshape` among others. `org.beehive.kshape.iotdb.UDTFClusterGPU` is a GPU version of that UDF: it takes the same parameters (`l`, `k`, `norm`, `maxiter`, `output`) and gives the same output, and runs k-Shape on a TornadoVM device.

IoTDB's k-Shape is a variant of the one above: no alignment step, the first `k` windows as initial centroids, centroids from the first right singular vector of the member windows, z-normalisation with the population standard deviation, and an empty cluster takes the previous centroid of the largest cluster. `WindowKShapeTornado` implements that variant with the kernels of this project. It computes in float32 on the device, while IoTDB computes in float64.

Setup (IoTDB 2.0.11, TornadoVM 8.1.0, JDK 25):

```bash
./mvnw -q package
cp target/kshape-tornadovm-0.1.0-SNAPSHOT.jar $IOTDB_HOME/ext/udf/
$IOTDB_HOME/sbin/start-confignode.sh
IOTDB_JVM_OPTS="@$TORNADOVM_HOME/tornado-argfile" $IOTDB_HOME/sbin/start-datanode.sh
```

```sql
CREATE FUNCTION cluster_gpu AS 'org.beehive.kshape.iotdb.UDTFClusterGPU'
SELECT cluster_gpu(s, 'l'='256', 'k'='8') FROM root.demo.d1
SELECT cluster_gpu(s, 'l'='256', 'k'='8', 'output'='centroid') FROM root.demo.d1
```

The first query compiles the kernels. The compiled kernels are kept for later queries with the same number of windows, window length and `k`.

`ClusterBench` compares `WindowKShapeTornado` with IoTDB's k-Shape (`library-udf.jar` from the IoTDB `library-udf` release on the class path) on synthetic sensor windows:

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp target/classes:library-udf.jar org.beehive.kshape.iotdb.ClusterBench --windows 5000 --l 256 --k 8
```

Results on the machine above, IoTDB 2.0.11 (`library-udf` release), synthetic windows of `k` random smooth shapes, each shifted, scaled, offset and noised, `norm=true`, `maxiter=200`. IoTDB's k-Shape runs on one CPU core. TornadoVM: mean of the runs after the first.

| Windows x l, k | Iterations | IoTDB k-Shape | TornadoVM | Speedup | Same labels as IoTDB |
|---|---:|---:|---:|---:|---:|
| 1000 x 128, 4 | 3 | 1.69 s | 0.036 s | 47x | 100% |
| 5000 x 256, 8 | 200 (max) | 1449 s | 1.11 s | 1300x | 100% |
| 2000 x 1024, 6 | 4 | 102.6 s | 0.174 s | 590x | 100% |
| 10000 x 512, 8 | 8 | 674.3 s | 0.405 s | 1660x | 100% |
| 20000 x 256, 10 | 9 | 409.8 s | 0.407 s | 1000x | 100% |

The centroids differ from IoTDB's by at most 1.8e-6. The first fit in a process also compiles the kernels (1.7 to 3.7 s).

Through SQL, on a 5.12 million point series, `cluster_gpu` takes 1.35 s (10000 windows of 512) and 1.6 to 1.8 s (20000 windows of 256) after the first query, including reading the series. `cluster` with `method=kshape` stops at IoTDB's default query timeout of 60 s on both. On a 400 x 64 series, both UDFs return the same labels through SQL (0.07 s and 0.88 s).

Labels can differ from IoTDB when two centroids have exactly the same shape, for example two constant windows of the same sign among the first `k` windows. Every window is then equally close to both, and either choice is correct.

## References

* J. Paparrizos and L. Gravano. k-Shape: Efficient and Accurate Clustering of Time Series. SIGMOD 2015.
* J. Paparrizos and L. Gravano. Fast and Accurate Time-Series Clustering. ACM TODS 42(2), 2017.

## License

MIT, see [LICENSE](LICENSE).
