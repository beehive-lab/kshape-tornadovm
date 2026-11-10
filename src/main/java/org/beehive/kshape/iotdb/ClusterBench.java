/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.kshape.iotdb;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * Compares {@link WindowKShapeTornado} with the k-Shape of the IoTDB {@code cluster} UDF on synthetic sensor
 * windows.
 *
 * <pre>
 * ClusterBench [--windows 2000] [--l 256] [--k 6] [--max-iter 200] [--repeat 5] [--seed 1] [--flat 0.0]
 *              [--no-norm] [--no-cpu]
 * </pre>
 *
 * The windows are k random smooth shapes, each window circularly shifted by up to l/10, scaled, offset and
 * with Gaussian noise; {@code --flat} sets the fraction of constant windows. The IoTDB version runs when
 * {@code library-udf.jar} from the IoTDB library-udf release is on the class path.
 */
public final class ClusterBench {

    private static final String IOTDB_KSHAPE = "org.apache.iotdb.library.dlearn.util.cluster.KShape";

    public static void main(String[] args) throws Exception {
        int m = 2000;
        int L = 256;
        int k = 6;
        int maxIter = 200;
        int repeat = 5;
        long seed = 1;
        double flat = 0.0;
        boolean norm = true;
        boolean cpu = true;
        for (int a = 0; a < args.length; a++) {
            switch (args[a]) {
                case "--windows" -> m = Integer.parseInt(args[++a]);
                case "--l" -> L = Integer.parseInt(args[++a]);
                case "--k" -> k = Integer.parseInt(args[++a]);
                case "--max-iter" -> maxIter = Integer.parseInt(args[++a]);
                case "--repeat" -> repeat = Integer.parseInt(args[++a]);
                case "--seed" -> seed = Long.parseLong(args[++a]);
                case "--flat" -> flat = Double.parseDouble(args[++a]);
                case "--no-norm" -> norm = false;
                case "--no-cpu" -> cpu = false;
                default -> throw new IllegalArgumentException("unknown argument " + args[a]);
            }
        }
        double[][] w = windows(m, L, k, seed, flat);
        System.out.printf(Locale.ROOT, "windows=%d l=%d k=%d norm=%b flat=%.2f%n", m, L, k, norm, flat);

        int[] gpuLabels;
        double[][] gpuCent;
        int gpuIters;
        try (WindowKShapeTornado ks = new WindowKShapeTornado(m, L, k)) {
            double sum = 0;
            for (int r = 0; r < repeat; r++) {
                long t0 = System.nanoTime();
                ks.fit(w, norm, maxIter);
                double s = (System.nanoTime() - t0) * 1e-9;
                if (r > 0) {
                    sum += s;
                }
                System.out.printf(Locale.ROOT, "[gpu] run %d: %.4f s iters=%d (device %.3f, eig %.3f, host %.3f)%n", r, s, ks.getIterations(), ks.tDevice * 1e-9,
                        ks.tEig * 1e-9, ks.tHost * 1e-9);
            }
            if (repeat > 1) {
                System.out.printf(Locale.ROOT, "[gpu] warm mean %.4f s%n", sum / (repeat - 1));
            }
            gpuLabels = ks.getLabels().clone();
            gpuCent = ks.getCentroids();
            gpuIters = ks.getIterations();
        }

        if (!cpu) {
            return;
        }
        Class<?> cls;
        try {
            cls = Class.forName(IOTDB_KSHAPE);
        } catch (ClassNotFoundException e) {
            System.out.println("[cpu] " + IOTDB_KSHAPE + " not on the class path, skipped");
            return;
        }
        Object ref = cls.getDeclaredConstructor().newInstance();
        Method fit = cls.getMethod("fit", double[][].class, int.class, boolean.class, int.class);
        long t0 = System.nanoTime();
        fit.invoke(ref, w, k, norm, maxIter);
        double s = (System.nanoTime() - t0) * 1e-9;
        int[] cpuLabels = (int[]) cls.getMethod("getLabels").invoke(ref);
        double[][] cpuCent = (double[][]) cls.getMethod("getCentroids").invoke(ref);
        int same = 0;
        for (int i = 0; i < m; i++) {
            if (cpuLabels[i] == gpuLabels[i]) {
                same++;
            }
        }
        double maxDiff = 0;
        for (int j = 0; j < k; j++) {
            for (int t = 0; t < L; t++) {
                maxDiff = Math.max(maxDiff, Math.abs(cpuCent[j][t] - gpuCent[j][t]));
            }
        }
        System.out.printf(Locale.ROOT, "[cpu] %.3f s%n", s);
        System.out.printf(Locale.ROOT, "same labels as IoTDB: %.2f%%  ARI=%.4f  max|centroid diff|=%.3e  gpu iterations=%d%n", 100.0 * same / m,
                ari(cpuLabels, gpuLabels), maxDiff, gpuIters);
    }

    static double[][] windows(int m, int L, int k, long seed, double flat) {
        Random rnd = new Random(seed);
        double[][] proto = new double[k][L];
        for (int j = 0; j < k; j++) {
            int terms = 2 + rnd.nextInt(3);
            for (int q = 0; q < terms; q++) {
                double f = 1 + rnd.nextInt(6);
                double ph = rnd.nextDouble() * 2 * Math.PI;
                double amp = 0.5 + rnd.nextDouble();
                for (int t = 0; t < L; t++) {
                    proto[j][t] += amp * Math.sin(2 * Math.PI * f * t / L + ph);
                }
            }
        }
        double[][] w = new double[m][L];
        int maxShift = Math.max(1, L / 10);
        for (int i = 0; i < m; i++) {
            if (rnd.nextDouble() < flat) {
                double level = rnd.nextGaussian() * 3;
                for (int t = 0; t < L; t++) {
                    w[i][t] = level;
                }
                continue;
            }
            int j = rnd.nextInt(k);
            int sh = rnd.nextInt(2 * maxShift + 1) - maxShift;
            double scale = 0.5 + 2 * rnd.nextDouble();
            double off = rnd.nextGaussian() * 5;
            for (int t = 0; t < L; t++) {
                w[i][t] = off + scale * proto[j][Math.floorMod(t - sh, L)] + 0.3 * rnd.nextGaussian();
            }
        }
        return w;
    }

    static double ari(int[] a, int[] b) {
        Map<Long, Integer> cells = new HashMap<>();
        Map<Integer, Integer> ra = new HashMap<>();
        Map<Integer, Integer> rb = new HashMap<>();
        for (int i = 0; i < a.length; i++) {
            cells.merge(((long) a[i] << 32) | (b[i] & 0xffffffffL), 1, Integer::sum);
            ra.merge(a[i], 1, Integer::sum);
            rb.merge(b[i], 1, Integer::sum);
        }
        double sumCells = 0;
        for (int c : cells.values()) {
            sumCells += c * (c - 1) / 2.0;
        }
        double sumA = 0;
        for (int c : ra.values()) {
            sumA += c * (c - 1) / 2.0;
        }
        double sumB = 0;
        for (int c : rb.values()) {
            sumB += c * (c - 1) / 2.0;
        }
        double n2 = a.length * (a.length - 1) / 2.0;
        double expected = sumA * sumB / n2;
        double max = (sumA + sumB) / 2;
        return max == expected ? 1.0 : (sumCells - expected) / (max - expected);
    }
}
