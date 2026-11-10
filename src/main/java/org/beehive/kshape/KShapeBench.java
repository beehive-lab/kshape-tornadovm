/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.kshape;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;

/**
 * Runs k-Shape on a dataset file and reports timings.
 *
 * <pre>
 * KShapeBench --data X.bin --k 3 [--init init.bin] [--max-iter 100] [--repeat 5] [--out prefix] [--random-init]
 * </pre>
 *
 * Dataset file: little-endian int32 m, L, D, then m*L*D float32 in NumPy (m, L, D) order. Init file: m int32
 * labels. Outputs: prefix.labels (m int32) and prefix.centroids (k*L*D float32).
 */
public final class KShapeBench {

    public static void main(String[] args) throws IOException {
        String data = null;
        String init = null;
        String out = null;
        int k = 3;
        int maxIter = 100;
        int repeat = 5;
        boolean randomInit = false;
        for (int a = 0; a < args.length; a++) {
            switch (args[a]) {
                case "--data" -> data = args[++a];
                case "--init" -> init = args[++a];
                case "--out" -> out = args[++a];
                case "--k" -> k = Integer.parseInt(args[++a]);
                case "--max-iter" -> maxIter = Integer.parseInt(args[++a]);
                case "--repeat" -> repeat = Integer.parseInt(args[++a]);
                case "--random-init" -> randomInit = true;
                default -> throw new IllegalArgumentException("unknown argument " + args[a]);
            }
        }
        if (data == null) {
            throw new IllegalArgumentException("--data is required");
        }
        ByteBuffer bb = ByteBuffer.wrap(Files.readAllBytes(Path.of(data))).order(ByteOrder.LITTLE_ENDIAN);
        int m = bb.getInt();
        int L = bb.getInt();
        int D = bb.getInt();
        float[] x = new float[m * L * D];
        bb.asFloatBuffer().get(x);
        int[] initLabels = null;
        if (init != null) {
            ByteBuffer ib = ByteBuffer.wrap(Files.readAllBytes(Path.of(init))).order(ByteOrder.LITTLE_ENDIAN);
            initLabels = new int[m];
            ib.asIntBuffer().get(initLabels);
        }

        long b0 = System.nanoTime();
        KShapeTornado.Result res = null;
        double[] times = new double[repeat];
        try (KShapeTornado ks = new KShapeTornado(x, m, L, D, k)) {
            double build = (System.nanoTime() - b0) / 1e9;
            System.out.printf(Locale.ROOT, "[kshape] m=%d L=%d D=%d k=%d fft=%d plan=%.3fs%n", m, L, D, k, ks.fftSize(), build);
            for (int r = 0; r < repeat; r++) {
                long t0 = System.nanoTime();
                res = ks.fit(initLabels, randomInit, maxIter, 42L + r);
                times[r] = (System.nanoTime() - t0) / 1e9;
                System.out.printf(Locale.ROOT,
                        "[kshape] run %d: %.4fs iters=%d (upload %.4f align %.4f eig %.4f direction %.4f assign %.4f host %.4f)%n", r, times[r],
                        res.iterations, ks.tUpload / 1e9, ks.tAlign / 1e9, ks.tEig / 1e9, ks.tCentroid / 1e9, ks.tAssign / 1e9, ks.tHost / 1e9);
            }
        }
        double warm = repeat > 1 ? Arrays.stream(times, 1, repeat).average().orElse(0) : times[0];
        System.out.printf(Locale.ROOT, "[kshape] first=%.4fs warm_mean=%.4fs iters=%d per_iter=%.5fs%n", times[0], warm, res.iterations,
                warm / Math.max(1, res.iterations));
        if (out != null) {
            ByteBuffer lb = ByteBuffer.allocate(4 * m).order(ByteOrder.LITTLE_ENDIAN);
            for (int v : res.labels) {
                lb.putInt(v);
            }
            Files.write(Path.of(out + ".labels"), lb.array());
            ByteBuffer cb = ByteBuffer.allocate(4 * res.centroids.length).order(ByteOrder.LITTLE_ENDIAN);
            for (float v : res.centroids) {
                cb.putFloat(v);
            }
            Files.write(Path.of(out + ".centroids"), cb.array());
        }
    }
}
