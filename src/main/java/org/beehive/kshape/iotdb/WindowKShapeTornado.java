/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.kshape.iotdb;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.beehive.kshape.KShapeKernels;
import org.beehive.kshape.SymmetricEigen;

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
 * The k-Shape variant of the Apache IoTDB {@code cluster} UDF ({@code method=kshape}) on a TornadoVM device.
 *
 * <p>Same steps and results as {@code org.apache.iotdb.library.dlearn.util.cluster.KShape}: the windows are
 * z-normalised (population standard deviation) when {@code normalize} is set, the first {@code k} windows are
 * the initial centroids, each iteration assigns every window to the centroid with the smallest shape-based
 * distance, then replaces each centroid by the first right singular vector of its members (sign chosen so that
 * the members' dot products sum to a non-negative value), z-normalised or scaled to unit length. An empty
 * cluster takes the previous centroid of the largest cluster. It stops when no label changes.
 *
 * <p>On the device: FFTs of the windows and centroids, all window-centroid cross-correlations and their
 * maxima, the per-cluster Gram (or covariance) matrices, the back-projection and the sign dot products. On the
 * host: the small symmetric eigenproblems and O(m k + k L) bookkeeping. The first right singular vector of the
 * member matrix Y is the top eigenvector of Y^T Y, which is Y^T u for the top eigenvector u of Y Y^T, so each
 * eigenproblem has size min(members, L).
 *
 * <p>The device computes in float32; the host bookkeeping uses float64. An instance can be reused for any
 * number of fits with the same (m, L, k), which avoids compiling the kernels again.
 */
public final class WindowKShapeTornado implements AutoCloseable {

    private static final int LOCAL = 256;
    private static final double EPS = 1e-9;

    private final int m;
    private final int L;
    private final int k;
    private final int N;
    private final int logN;

    private final double[] zHost; // [i][t], float64 copy of what the device sees
    private final double[] rowNorm;

    private final FloatArray x;
    private final FloatArray tw;
    private final FloatArray fx;
    private final FloatArray fxTmp;
    private final FloatArray c;
    private final FloatArray fc;
    private final FloatArray fcTmp;
    private final FloatArray assignA;
    private final FloatArray assignB;
    private final FloatArray maxCC;
    private final IntArray labelsDev;
    private final IntArray members;
    private final IntArray tiles;
    private final IntArray blocks;
    private final IntArray meta;
    private final FloatArray shapeOut;
    private final FloatArray u;
    private final FloatArray v;
    private final FloatArray dots;

    private final int maxTiles;
    private final TornadoExecutionPlan plan;

    private int[] labels;
    private double[][] centroids;
    private int iterations;

    /** Time spent in the last fit, in nanoseconds: device graphs, host eigensolver, other host work. */
    public long tDevice, tEig, tHost;

    public WindowKShapeTornado(int m, int L, int k) {
        if (k < 2 || k > m) {
            throw new IllegalArgumentException("k must satisfy 2 <= k <= number of windows.");
        }
        if (L < 1) {
            throw new IllegalArgumentException("window length must be positive.");
        }
        this.m = m;
        this.L = L;
        this.k = k;
        int n = 1;
        int lg = 0;
        while (n < 2 * L - 1) {
            n <<= 1;
            lg++;
        }
        this.N = n;
        this.logN = lg;
        long complexFloats = 2L * Math.max(m, (long) m * k) * N;
        if (complexFloats >= Integer.MAX_VALUE) {
            throw new IllegalArgumentException("problem too large for 32-bit indexing: " + complexFloats);
        }

        zHost = new double[m * L];
        rowNorm = new double[m];
        x = new FloatArray(m * L);
        tw = new FloatArray(Math.max(2, N));
        for (int t = 0; t < N / 2; t++) {
            double ang = -2.0 * Math.PI * t / N;
            tw.set(2 * t, (float) Math.cos(ang));
            tw.set(2 * t + 1, (float) Math.sin(ang));
        }
        fx = new FloatArray(2 * m * N);
        fxTmp = new FloatArray(2 * m * N);
        c = new FloatArray(k * L);
        fc = new FloatArray(2 * k * N);
        fcTmp = new FloatArray(2 * k * N);
        assignA = new FloatArray(2 * m * k * N);
        assignB = new FloatArray(2 * m * k * N);
        maxCC = new FloatArray(m * k);
        labelsDev = new IntArray(m);
        members = new IntArray(m);
        int maxR = Math.min(m, L);
        long sumR = Math.min((long) m, (long) k * L);
        maxTiles = (int) ((sumR / KShapeKernels.TILE + k) * (maxR / KShapeKernels.TILE + 1));
        tiles = new IntArray(3 * maxTiles);
        blocks = new IntArray(7 * k);
        meta = new IntArray(1);
        long shapeBound = sumR * maxR;
        if (shapeBound >= Integer.MAX_VALUE) {
            throw new IllegalArgumentException("shape matrices too large: " + shapeBound);
        }
        shapeOut = new FloatArray((int) Math.max(1, shapeBound));
        u = new FloatArray((int) Math.max(1L, sumR));
        v = new FloatArray(k * L);
        dots = new FloatArray(m);

        plan = buildPlan();
    }

    public int windows() {
        return m;
    }

    public int length() {
        return L;
    }

    public int clusters() {
        return k;
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

        // graph 0: upload the windows and transform them
        TaskGraph g0 = new TaskGraph("data").transferToDevice(DataTransferMode.EVERY_EXECUTION, x) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, tw);
        FloatArray fxIn = logN % 2 == 0 ? fx : fxTmp;
        g0.task("load", KShapeKernels::loadComplex, new KernelContext(), x, fxIn, m, L, N);
        gs.addWorkerGrid("data.load", grid(roundUp((long) m * N, LOCAL), LOCAL));
        addFft(g0, gs, "data", "fft", fxIn, fxIn == fx ? fxTmp : fx, m, 1.0f);
        g0.persistOnDevice(x, tw, fx);

        // graph 1: centroid spectra
        TaskGraph g1 = new TaskGraph("centroids").consumeFromDevice("data", tw) //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, c);
        FloatArray fcIn = logN % 2 == 0 ? fc : fcTmp;
        g1.task("load", KShapeKernels::loadComplex, new KernelContext(), c, fcIn, k, L, N);
        gs.addWorkerGrid("centroids.load", grid(roundUp((long) k * N, LOCAL), LOCAL));
        addFft(g1, gs, "centroids", "fft", fcIn, fcIn == fc ? fcTmp : fc, k, 1.0f);
        g1.persistOnDevice(fc);

        // graph 2: cross-correlation of every window with every centroid, maximum over lags
        TaskGraph g2 = new TaskGraph("assign").consumeFromDevice("data", tw, fx) //
                .consumeFromDevice("centroids", fc);
        g2.task("product", KShapeKernels::assignProduct, new KernelContext(), fx, fc, assignA, m, k, 1, N);
        gs.addWorkerGrid("assign.product", grid(roundUp((long) m * k * N, LOCAL), LOCAL));
        FloatArray cc = addFft(g2, gs, "assign", "ifft", assignA, assignB, m * k, -1.0f);
        g2.task("max", KShapeKernels::lagMax, new KernelContext(), cc, maxCC, m * k, L, N);
        gs.addWorkerGrid("assign.max", grid((long) m * k * KShapeKernels.RED_LOCAL, KShapeKernels.RED_LOCAL));
        g2.transferToHost(DataTransferMode.EVERY_EXECUTION, maxCC);

        // graph 3: Gram or covariance matrix of every cluster
        TaskGraph g3 = new TaskGraph("shape").consumeFromDevice("data", x) //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, labelsDev, members, tiles, blocks, meta);
        g3.task("shape", KShapeKernels::shapeTiles, new KernelContext(), x, members, tiles, blocks, meta, shapeOut, 1, L);
        gs.addWorkerGrid("shape.shape", grid((long) maxTiles * KShapeKernels.TILE * KShapeKernels.TILE, KShapeKernels.TILE * KShapeKernels.TILE));
        g3.transferToHost(DataTransferMode.EVERY_EXECUTION, shapeOut);
        g3.persistOnDevice(labelsDev, members, blocks);

        // graph 4: singular vectors from the eigenvectors, and the dot products for the sign
        TaskGraph g4 = new TaskGraph("direction").consumeFromDevice("data", x) //
                .consumeFromDevice("shape", labelsDev, members, blocks) //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, u);
        g4.task("project", KShapeKernels::backProject, new KernelContext(), x, members, blocks, u, v, k, 1, L);
        gs.addWorkerGrid("direction.project", grid(roundUp((long) k * L, LOCAL), LOCAL));
        g4.task("dots", KShapeKernels::signDots, new KernelContext(), x, v, labelsDev, dots, m, 1, L);
        gs.addWorkerGrid("direction.dots", grid((long) m * KShapeKernels.RED_LOCAL, KShapeKernels.RED_LOCAL));
        g4.transferToHost(DataTransferMode.EVERY_EXECUTION, v, dots);

        TornadoExecutionPlan p = new TornadoExecutionPlan(g0.snapshot(), g1.snapshot(), g2.snapshot(), g3.snapshot(), g4.snapshot());
        p.withGridScheduler(gs);
        return p;
    }

    /**
     * Clusters {@code samples} (m windows of length L).
     *
     * @param normalize z-normalise the windows and the centroids (otherwise centroids are scaled to unit length)
     */
    public void fit(double[][] samples, boolean normalize, int maxIterations) {
        if (samples.length != m) {
            throw new IllegalArgumentException("expected " + m + " windows, got " + samples.length);
        }
        if (maxIterations < 1) {
            throw new IllegalArgumentException("maxIterations must be at least 1.");
        }
        tDevice = tEig = tHost = 0;
        long t0 = System.nanoTime();
        for (int i = 0; i < m; i++) {
            if (samples[i].length != L) {
                throw new IllegalArgumentException("All samples must have the same length " + L + ".");
            }
            double[] z = normalize ? zNormalize(samples[i]) : samples[i];
            double s = 0;
            for (int t = 0; t < L; t++) {
                zHost[i * L + t] = z[t];
                x.set(i * L + t, (float) z[t]);
                s += z[t] * z[t];
            }
            rowNorm[i] = Math.sqrt(s);
        }
        double[][] cent = new double[k][];
        for (int j = 0; j < k; j++) {
            cent[j] = Arrays.copyOfRange(zHost, j * L, (j + 1) * L);
        }
        int[] lab = new int[m];
        Arrays.fill(lab, -1);
        tHost += System.nanoTime() - t0;

        t0 = System.nanoTime();
        plan.withGraph(0).execute();
        tDevice += System.nanoTime() - t0;

        int it = 0;
        while (it < maxIterations) {
            it++;
            // assignment
            t0 = System.nanoTime();
            double[] cNorm = new double[k];
            for (int j = 0; j < k; j++) {
                double s = 0;
                for (int t = 0; t < L; t++) {
                    c.set(j * L + t, (float) cent[j][t]);
                    s += cent[j][t] * cent[j][t];
                }
                cNorm[j] = Math.sqrt(s);
            }
            plan.withGraph(1).execute();
            plan.withGraph(2).execute();
            tDevice += System.nanoTime() - t0;

            t0 = System.nanoTime();
            boolean changed = false;
            for (int i = 0; i < m; i++) {
                int best = 0;
                double bestDist = Double.POSITIVE_INFINITY;
                for (int j = 0; j < k; j++) {
                    double den = rowNorm[i] * cNorm[j];
                    double ncc = den < EPS ? 0.0 : maxCC.get(i * k + j) / N / den;
                    double d = 1.0 - ncc;
                    if (d < bestDist) {
                        bestDist = d;
                        best = j;
                    }
                }
                if (lab[i] != best) {
                    lab[i] = best;
                    changed = true;
                }
            }

            // members sorted by cluster, Gram tiles
            int[] count = new int[k];
            for (int i = 0; i < m; i++) {
                count[lab[i]]++;
            }
            int[] start = new int[k];
            for (int j = 1; j < k; j++) {
                start[j] = start[j - 1] + count[j - 1];
            }
            int[] fill = start.clone();
            for (int i = 0; i < m; i++) {
                labelsDev.set(i, lab[i]);
                members.set(fill[lab[i]]++, i);
            }
            int[] r = new int[k];
            int[] outOff = new int[k];
            int[] uOff = new int[k];
            int o = 0;
            int uo = 0;
            int nt = 0;
            for (int j = 0; j < k; j++) {
                r[j] = Math.min(count[j], L);
                blocks.set(7 * j, start[j]);
                blocks.set(7 * j + 1, count[j]);
                blocks.set(7 * j + 2, r[j]);
                blocks.set(7 * j + 3, count[j] <= L ? 1 : 0);
                blocks.set(7 * j + 4, 0);
                blocks.set(7 * j + 5, o);
                blocks.set(7 * j + 6, uo);
                outOff[j] = o;
                uOff[j] = uo;
                o += r[j] * r[j];
                uo += r[j];
                int nTile = (r[j] + KShapeKernels.TILE - 1) / KShapeKernels.TILE;
                for (int ti = 0; ti < nTile; ti++) {
                    for (int tj = 0; tj < nTile; tj++) {
                        tiles.set(3 * nt, j);
                        tiles.set(3 * nt + 1, ti * KShapeKernels.TILE);
                        tiles.set(3 * nt + 2, tj * KShapeKernels.TILE);
                        nt++;
                    }
                }
            }
            meta.set(0, nt);
            tHost += System.nanoTime() - t0;

            t0 = System.nanoTime();
            plan.withGraph(3).execute();
            tDevice += System.nanoTime() - t0;

            t0 = System.nanoTime();
            float[] shapeHost = shapeOut.toHeapArray();
            IntStream.range(0, k).parallel().forEach(j -> {
                int n = r[j];
                if (n == 0) {
                    return;
                }
                double[] top = SymmetricEigen.topEigenvector(shapeHost, outOff[j], n);
                for (int p = 0; p < n; p++) {
                    u.set(uOff[j] + p, (float) top[p]);
                }
            });
            tEig += System.nanoTime() - t0;

            t0 = System.nanoTime();
            plan.withGraph(4).execute();
            tDevice += System.nanoTime() - t0;

            // sign, normalisation, empty clusters
            t0 = System.nanoTime();
            float[] vHost = v.toHeapArray();
            float[] dHost = dots.toHeapArray();
            double[][] prev = cent;
            int largest = 0;
            for (int j = 1; j < k; j++) {
                if (count[j] > count[largest]) {
                    largest = j;
                }
            }
            double[][] next = new double[k][];
            for (int j = 0; j < k; j++) {
                if (count[j] == 0) {
                    next[j] = prev[largest].clone();
                    continue;
                }
                double sumDot = 0;
                boolean allZero = true;
                for (int q = start[j]; q < start[j] + count[j]; q++) {
                    int i = members.get(q);
                    sumDot += dHost[i];
                    allZero &= rowNorm[i] == 0.0;
                }
                double sign = sumDot < 0 ? -1.0 : 1.0;
                double[] dir = new double[L];
                if (allZero && count[j] > 1) {
                    // every member is a constant window; the SVD used by IoTDB (Apache Commons Math) then
                    // returns the first unit vector, with a negative sign when the matrix is not wide
                    dir[0] = count[j] < L ? 1.0 : -1.0;
                } else {
                    for (int t = 0; t < L; t++) {
                        dir[t] = sign * vHost[j * L + t];
                    }
                }
                next[j] = normalize ? zNormalize(dir) : unit(dir);
            }
            cent = next;
            tHost += System.nanoTime() - t0;
            if (!changed) {
                break;
            }
        }
        labels = lab;
        centroids = cent;
        iterations = it;
    }

    public int[] getLabels() {
        return labels;
    }

    public double[][] getCentroids() {
        return centroids;
    }

    public int getIterations() {
        return iterations;
    }

    /** z-normalisation with the population standard deviation; a constant window becomes all zeros. */
    static double[] zNormalize(double[] a) {
        int n = a.length;
        double sum = 0;
        for (double val : a) {
            sum += val;
        }
        double mean = sum / n;
        double var = 0;
        for (double val : a) {
            double d = val - mean;
            var += d * d;
        }
        double std = Math.sqrt(Math.max(var / n, 0.0));
        double[] z = new double[n];
        if (std < EPS) {
            return z;
        }
        for (int i = 0; i < n; i++) {
            z[i] = (a[i] - mean) / std;
        }
        return z;
    }

    private static double[] unit(double[] a) {
        double s = 0;
        for (double val : a) {
            s += val * val;
        }
        s = Math.sqrt(s);
        double[] o = new double[a.length];
        if (s < EPS) {
            return o;
        }
        for (int i = 0; i < a.length; i++) {
            o[i] = a[i] / s;
        }
        return o;
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
