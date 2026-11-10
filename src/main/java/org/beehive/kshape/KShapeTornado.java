/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.kshape;

import java.util.Random;
import java.util.stream.IntStream;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * k-Shape clustering (Paparrizos and Gravano, SIGMOD 2015) on a TornadoVM device.
 *
 * <p>Follows the reference Python implementation step for step: random initial labels, shape extraction
 * per (cluster, dimension) with SBD alignment, z-normalisation and the top eigenvector of the k-Shape
 * matrix, then assignment by the multivariate normalised cross-correlation, until the labels stop changing.
 *
 * <p>On the device: all cross-correlations (FFT based), alignment, z-normalisation, shape matrices and
 * reductions. On the host: the small symmetric eigenproblems and O(k D L) bookkeeping. The k-Shape matrix
 * P^T S P equals S = Y^T Y because every row of Y has zero mean, and its top eigenvector is Y^T u for the
 * top eigenvector u of the Gram matrix Y Y^T, so each eigenproblem has size min(members, L).
 */
public final class KShapeTornado implements AutoCloseable {

    private static final int LOCAL = 256;

    private final int m;
    private final int L;
    private final int D;
    private final int k;
    private final int N;
    private final int logN;

    // host copies
    private final float[] xHost; // [i][d][t]
    private final double[] xNormD; // per (i,d)
    private final double[] xNormF; // per i

    // device buffers
    private final FloatArray x;
    private final FloatArray tw;
    private final FloatArray fx;
    private final FloatArray fxTmp;
    private final FloatArray c;
    private final FloatArray fc;
    private final FloatArray fcTmp;
    private final FloatArray alignA;
    private final FloatArray alignB;
    private final FloatArray assignA;
    private final FloatArray assignB;
    private final IntArray labels;
    private final FloatArray xNormDev;
    private final FloatArray cNormDev;
    private final IntArray shift;
    private final FloatArray aligned;
    private final FloatArray y;
    private final FloatArray aNorm2;
    private final IntArray members;
    private final IntArray tiles;
    private final IntArray blocks;
    private final IntArray meta;
    private final FloatArray shapeOut;
    private final FloatArray u;
    private final FloatArray v;
    private final FloatArray dots;
    private final FloatArray maxCC;

    private final int maxTiles;
    private final TornadoExecutionPlan plan;

    /** Timings of the last fit, in nanoseconds. */
    public long tUpload, tAlign, tEig, tCentroid, tAssign, tHost;
    public int iterations;

    /**
     * @param data series in NumPy order: data[(i * L + t) * D + d]
     */
    public KShapeTornado(float[] data, int m, int L, int D, int k) {
        this.m = m;
        this.L = L;
        this.D = D;
        this.k = k;
        int n = 1;
        int lg = 0;
        while (n < 2 * L - 1) {
            n <<= 1;
            lg++;
        }
        this.N = n;
        this.logN = lg;
        long complexFloats = 2L * Math.max(m * D, m * k) * N;
        if (complexFloats >= Integer.MAX_VALUE) {
            throw new IllegalArgumentException("problem too large for 32-bit indexing: " + complexFloats);
        }

        xHost = new float[m * D * L];
        xNormD = new double[m * D];
        xNormF = new double[m];
        for (int i = 0; i < m; i++) {
            double f = 0;
            for (int d = 0; d < D; d++) {
                double s = 0;
                for (int t = 0; t < L; t++) {
                    float val = data[(i * L + t) * D + d];
                    xHost[(i * D + d) * L + t] = val;
                    s += (double) val * val;
                }
                xNormD[i * D + d] = Math.sqrt(s);
                f += s;
            }
            xNormF[i] = Math.sqrt(f);
        }

        x = FloatArray.fromArray(xHost);
        tw = new FloatArray(N);
        for (int t = 0; t < N / 2; t++) {
            double ang = -2.0 * Math.PI * t / N;
            tw.set(2 * t, (float) Math.cos(ang));
            tw.set(2 * t + 1, (float) Math.sin(ang));
        }
        fx = new FloatArray(2 * m * D * N);
        fxTmp = new FloatArray(2 * m * D * N);
        c = new FloatArray(k * D * L);
        fc = new FloatArray(2 * k * D * N);
        fcTmp = new FloatArray(2 * k * D * N);
        alignA = new FloatArray(2 * m * D * N);
        alignB = new FloatArray(2 * m * D * N);
        assignA = new FloatArray(2 * m * k * N);
        assignB = new FloatArray(2 * m * k * N);
        labels = new IntArray(m);
        xNormDev = new FloatArray(m * D);
        for (int r = 0; r < m * D; r++) {
            xNormDev.set(r, (float) xNormD[r]);
        }
        cNormDev = new FloatArray(k * D);
        shift = new IntArray(m * D);
        aligned = new FloatArray(m * D * L);
        y = new FloatArray(m * D * L);
        aNorm2 = new FloatArray(m * D);
        members = new IntArray(m);
        int maxR = Math.min(m, L);
        long sumR = Math.min((long) m, (long) k * L);
        long tilesBound = (long) D * (sumR / KShapeKernels.TILE + k) * (maxR / KShapeKernels.TILE + 1);
        maxTiles = (int) tilesBound;
        tiles = new IntArray(3 * maxTiles);
        blocks = new IntArray(7 * k * D);
        meta = new IntArray(1);
        long shapeBound = (long) D * sumR * maxR;
        if (shapeBound >= Integer.MAX_VALUE) {
            throw new IllegalArgumentException("shape matrices too large: " + shapeBound);
        }
        shapeOut = new FloatArray((int) Math.max(1, shapeBound));
        u = new FloatArray((int) Math.max(1L, D * sumR));
        v = new FloatArray(k * D * L);
        dots = new FloatArray(m * D);
        maxCC = new FloatArray(m * k);

        plan = buildPlan();
    }

    private static WorkerGrid grid(long global, int local) {
        WorkerGrid g = new WorkerGrid1D(Math.toIntExact(global));
        g.setLocalWork(local, 1, 1);
        return g;
    }

    private static long roundUp(long n, int local) {
        return ((n + local - 1) / local) * local;
    }

    /** Appends log2(N) Stockham passes; returns the buffer that holds the result. */
    private FloatArray addFft(TaskGraph g, GridScheduler gs, String gName, String prefix, FloatArray a, FloatArray b, int rows, float conj) {
        FloatArray src = a;
        FloatArray dst = b;
        int pass = 0;
        for (int ns = 1; ns < N; ns <<= 1) {
            String name = prefix + pass;
            g.task(name, KShapeKernels::fftPass, new KernelContext(), src, dst, tw, rows, N, ns, N / (2 * ns), conj);
            gs.addWorkerGrid(gName + "." + name, grid(roundUp((long) rows * N / 2, LOCAL), LOCAL));
            FloatArray t = src;
            src = dst;
            dst = t;
            pass++;
        }
        return src;
    }

    private TornadoExecutionPlan buildPlan() {
        GridScheduler gs = new GridScheduler();
        int rowsX = m * D;
        int rowsC = k * D;
        int rowsA = m * k;

        // graph 0: upload the data and transform it
        TaskGraph g0 = new TaskGraph("data").transferToDevice(DataTransferMode.EVERY_EXECUTION, x) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, tw);
        // the transform ping-pongs log2(N) times and must end in fx
        FloatArray fxIn = logN % 2 == 0 ? fx : fxTmp;
        g0.task("load", KShapeKernels::loadComplex, new KernelContext(), x, fxIn, rowsX, L, N);
        gs.addWorkerGrid("data.load", grid(roundUp((long) rowsX * N, LOCAL), LOCAL));
        addFft(g0, gs, "data", "fft", fxIn, fxIn == fx ? fxTmp : fx, rowsX, 1.0f);
        g0.persistOnDevice(x, tw, fx);

        // graph 1: centroid spectra
        TaskGraph g1 = new TaskGraph("centroids").consumeFromDevice("data", tw) //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, c);
        FloatArray fcIn = logN % 2 == 0 ? fc : fcTmp;
        g1.task("load", KShapeKernels::loadComplex, new KernelContext(), c, fcIn, rowsC, L, N);
        gs.addWorkerGrid("centroids.load", grid(roundUp((long) rowsC * N, LOCAL), LOCAL));
        addFft(g1, gs, "centroids", "fft", fcIn, fcIn == fc ? fcTmp : fc, rowsC, 1.0f);
        g1.persistOnDevice(fc);

        // graph 2: alignment, z-normalisation and shape matrices
        TaskGraph g2 = new TaskGraph("align").consumeFromDevice("data", x, tw, fx) //
                .consumeFromDevice("centroids", fc) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, xNormDev) //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, labels, cNormDev, members, tiles, blocks, meta);
        g2.task("product", KShapeKernels::alignProduct, new KernelContext(), fx, fc, labels, alignA, rowsX, D, N);
        gs.addWorkerGrid("align.product", grid(roundUp((long) rowsX * N, LOCAL), LOCAL));
        FloatArray ccAlign = addFft(g2, gs, "align", "ifft", alignA, alignB, rowsX, -1.0f);
        g2.task("argmax", KShapeKernels::lagArgmax, new KernelContext(), ccAlign, shift, xNormDev, cNormDev, labels, rowsX, D, L, N);
        gs.addWorkerGrid("align.argmax", grid((long) rowsX * KShapeKernels.RED_LOCAL, KShapeKernels.RED_LOCAL));
        g2.task("znorm", KShapeKernels::alignZ, new KernelContext(), x, shift, aligned, y, aNorm2, rowsX, L);
        gs.addWorkerGrid("align.znorm", grid((long) rowsX * KShapeKernels.RED_LOCAL, KShapeKernels.RED_LOCAL));
        g2.task("shape", KShapeKernels::shapeTiles, new KernelContext(), y, members, tiles, blocks, meta, shapeOut, D, L);
        gs.addWorkerGrid("align.shape", grid((long) maxTiles * KShapeKernels.TILE * KShapeKernels.TILE, KShapeKernels.TILE * KShapeKernels.TILE));
        g2.transferToHost(DataTransferMode.EVERY_EXECUTION, shapeOut, aNorm2);
        g2.persistOnDevice(aligned, y, members, blocks, labels);

        // graph 3: centroid directions and sign test
        TaskGraph g3 = new TaskGraph("direction").consumeFromDevice("align", aligned, y, members, blocks, labels) //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, u);
        g3.task("project", KShapeKernels::backProject, new KernelContext(), y, members, blocks, u, v, rowsC, D, L);
        gs.addWorkerGrid("direction.project", grid(roundUp((long) rowsC * L, LOCAL), LOCAL));
        g3.task("dots", KShapeKernels::signDots, new KernelContext(), aligned, v, labels, dots, rowsX, D, L);
        gs.addWorkerGrid("direction.dots", grid((long) rowsX * KShapeKernels.RED_LOCAL, KShapeKernels.RED_LOCAL));
        g3.transferToHost(DataTransferMode.EVERY_EXECUTION, v, dots);

        // graph 4: assignment
        TaskGraph g4 = new TaskGraph("assign").consumeFromDevice("data", tw, fx) //
                .consumeFromDevice("centroids", fc);
        g4.task("product", KShapeKernels::assignProduct, new KernelContext(), fx, fc, assignA, m, k, D, N);
        gs.addWorkerGrid("assign.product", grid(roundUp((long) rowsA * N, LOCAL), LOCAL));
        FloatArray ccAssign = addFft(g4, gs, "assign", "ifft", assignA, assignB, rowsA, -1.0f);
        g4.task("max", KShapeKernels::lagMax, new KernelContext(), ccAssign, maxCC, rowsA, L, N);
        gs.addWorkerGrid("assign.max", grid((long) rowsA * KShapeKernels.RED_LOCAL, KShapeKernels.RED_LOCAL));
        g4.transferToHost(DataTransferMode.EVERY_EXECUTION, maxCC);

        TornadoExecutionPlan p = new TornadoExecutionPlan(g0.snapshot(), g1.snapshot(), g2.snapshot(), g3.snapshot(), g4.snapshot());
        p.withGridScheduler(gs);
        return p;
    }

    /** Result of a fit. */
    public static final class Result {
        public final int[] labels;
        /** centroids[(j * L + t) * D + d], NumPy order. */
        public final float[] centroids;
        public final int iterations;

        Result(int[] labels, float[] centroids, int iterations) {
            this.labels = labels;
            this.centroids = centroids;
            this.iterations = iterations;
        }
    }

    /**
     * @param initLabels initial labels (as the reference draws with randint), or null for random ones
     * @param randomInit start from k random series instead of zero centroids
     */
    public Result fit(int[] initLabels, boolean randomInit, int maxIter, long seed) {
        Random rnd = new Random(seed);
        int[] idx = new int[m];
        for (int i = 0; i < m; i++) {
            idx[i] = initLabels != null ? initLabels[i] : rnd.nextInt(k);
        }
        double[] cent = new double[k * D * L]; // [j][d][t]
        if (randomInit) {
            for (int j = 0; j < k; j++) {
                int s = rnd.nextInt(m);
                for (int q = 0; q < D * L; q++) {
                    cent[j * D * L + q] = xHost[s * D * L + q];
                }
            }
        }
        EmptyPick emptyPick = new EmptyPick(m);
        tUpload = tAlign = tEig = tCentroid = tAssign = tHost = 0;

        long t0 = System.nanoTime();
        plan.withGraph(0).execute();
        tUpload += System.nanoTime() - t0;
        uploadCentroids(cent);

        int it = 0;
        for (; it < maxIter; it++) {
            t0 = System.nanoTime();
            // cluster membership, sorted by cluster
            int[] count = new int[k];
            for (int i = 0; i < m; i++) {
                count[idx[i]]++;
            }
            int[] start = new int[k];
            for (int j = 1; j < k; j++) {
                start[j] = start[j - 1] + count[j - 1];
            }
            int[] fill = start.clone();
            for (int i = 0; i < m; i++) {
                labels.set(i, idx[i]);
                members.set(fill[idx[i]]++, i);
            }
            int[] r = new int[k];
            int[] outOff = new int[k * D];
            int[] uOff = new int[k * D];
            int o = 0;
            int uo = 0;
            int nt = 0;
            for (int j = 0; j < k; j++) {
                r[j] = count[j] <= L ? count[j] : L;
                for (int d = 0; d < D; d++) {
                    int b = j * D + d;
                    blocks.set(7 * b, start[j]);
                    blocks.set(7 * b + 1, count[j]);
                    blocks.set(7 * b + 2, r[j]);
                    blocks.set(7 * b + 3, count[j] <= L ? 1 : 0);
                    blocks.set(7 * b + 4, d);
                    blocks.set(7 * b + 5, o);
                    blocks.set(7 * b + 6, uo);
                    outOff[b] = o;
                    uOff[b] = uo;
                    o += r[j] * r[j];
                    uo += r[j];
                    int nTile = (r[j] + KShapeKernels.TILE - 1) / KShapeKernels.TILE;
                    for (int ti = 0; ti < nTile; ti++) {
                        for (int tj = 0; tj < nTile; tj++) {
                            tiles.set(3 * nt, b);
                            tiles.set(3 * nt + 1, ti * KShapeKernels.TILE);
                            tiles.set(3 * nt + 2, tj * KShapeKernels.TILE);
                            nt++;
                        }
                    }
                }
            }
            meta.set(0, nt);
            tHost += System.nanoTime() - t0;

            t0 = System.nanoTime();
            plan.withGraph(2).execute();
            tAlign += System.nanoTime() - t0;

            // top eigenvectors on the host, in parallel over blocks
            t0 = System.nanoTime();
            float[] shapeHost = shapeOut.toHeapArray();
            int[] rr = r;
            IntStream.range(0, k * D).parallel().forEach(b -> {
                int j = b / D;
                int n = rr[j];
                if (n == 0) {
                    return;
                }
                double[] top = SymmetricEigen.topEigenvector(shapeHost, outOff[b], n);
                for (int p = 0; p < n; p++) {
                    u.set(uOff[b] + p, (float) top[p]);
                }
            });
            tEig += System.nanoTime() - t0;

            t0 = System.nanoTime();
            plan.withGraph(3).execute();
            tCentroid += System.nanoTime() - t0;

            t0 = System.nanoTime();
            float[] vHost = v.toHeapArray();
            float[] dHost = dots.toHeapArray();
            float[] a2 = aNorm2.toHeapArray();
            for (int j = 0; j < k; j++) {
                for (int d = 0; d < D; d++) {
                    int b = j * D + d;
                    int cb = b * L;
                    if (count[j] == 0) {
                        int s = emptyPick.next();
                        for (int t = 0; t < L; t++) {
                            cent[cb + t] = xHost[(s * D + d) * L + t];
                        }
                        continue;
                    }
                    double nv = 0;
                    for (int t = 0; t < L; t++) {
                        nv += (double) vHost[cb + t] * vHost[cb + t];
                    }
                    nv = Math.sqrt(nv);
                    double inv = nv > 0 ? 1.0 / nv : 0.0;
                    double d1 = 0;
                    double d2 = 0;
                    for (int q = start[j]; q < start[j] + count[j]; q++) {
                        int row = members.get(q) * D + d;
                        double an = a2[row];
                        double dot = dHost[row] * inv;
                        double cn = nv > 0 ? 1.0 : 0.0;
                        d1 += Math.sqrt(Math.max(0.0, an - 2 * dot + cn));
                        d2 += Math.sqrt(Math.max(0.0, an + 2 * dot + cn));
                    }
                    double sign = d1 >= d2 ? -inv : inv;
                    double mean = 0;
                    for (int t = 0; t < L; t++) {
                        mean += vHost[cb + t] * sign;
                    }
                    mean /= L;
                    double var = 0;
                    for (int t = 0; t < L; t++) {
                        double e = vHost[cb + t] * sign - mean;
                        var += e * e;
                    }
                    double std = Math.sqrt(var / (L - 1));
                    for (int t = 0; t < L; t++) {
                        cent[cb + t] = std > 0 ? (vHost[cb + t] * sign - mean) / std : 0.0;
                    }
                }
            }
            tHost += System.nanoTime() - t0;

            t0 = System.nanoTime();
            double[] cNormF = uploadCentroids(cent);
            plan.withGraph(4).execute();
            tAssign += System.nanoTime() - t0;

            t0 = System.nanoTime();
            boolean changed = false;
            for (int i = 0; i < m; i++) {
                int bestJ = 0;
                double bestD = Double.POSITIVE_INFINITY;
                for (int j = 0; j < k; j++) {
                    double den = xNormF[i] * cNormF[j];
                    double ncc = den < 1e-9 ? 0.0 : maxCC.get(i * k + j) / N / den;
                    double dist = 1.0 - ncc;
                    if (dist < bestD) {
                        bestD = dist;
                        bestJ = j;
                    }
                }
                if (bestJ != idx[i]) {
                    changed = true;
                }
                idx[i] = bestJ;
            }
            tHost += System.nanoTime() - t0;
            if (!changed) {
                it++;
                break;
            }
        }
        iterations = it;
        float[] outC = new float[k * L * D];
        for (int j = 0; j < k; j++) {
            for (int d = 0; d < D; d++) {
                for (int t = 0; t < L; t++) {
                    outC[(j * L + t) * D + d] = (float) cent[(j * D + d) * L + t];
                }
            }
        }
        return new Result(idx.clone(), outC, it);
    }

    /**
     * Series that reseeds an empty cluster. The reference draws it at random; a fixed LCG sequence (also
     * installed by python/run_reference.py) keeps runs reproducible and comparable.
     */
    static final class EmptyPick {
        private final int m;
        private long state = 12345;

        EmptyPick(int m) {
            this.m = m;
        }

        int next() {
            state = (state * 1103515245L + 12345L) & 0x7fffffffL;
            return (int) (state % m);
        }
    }

    /** Uploads centroids, computes their spectra; returns their Frobenius norms. */
    private double[] uploadCentroids(double[] cent) {
        double[] normF = new double[k];
        for (int j = 0; j < k; j++) {
            for (int d = 0; d < D; d++) {
                double s = 0;
                for (int t = 0; t < L; t++) {
                    double val = cent[(j * D + d) * L + t];
                    c.set((j * D + d) * L + t, (float) val);
                    s += val * val;
                }
                cNormDev.set(j * D + d, (float) Math.sqrt(s));
                normF[j] += s;
            }
            normF[j] = Math.sqrt(normF[j]);
        }
        plan.withGraph(1).execute();
        return normF;
    }

    public int fftSize() {
        return N;
    }

    @Override
    public void close() {
        try {
            plan.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
