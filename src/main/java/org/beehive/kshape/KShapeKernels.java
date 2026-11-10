/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.kshape;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * GPU kernels for k-Shape.
 *
 * <p>Complex buffers are interleaved (re, im). A "row" is one series of one dimension, laid out as
 * {@code row = series * D + dim}. FFT rows have length {@code N} (a power of two, {@code N >= 2L - 1}).
 */
public final class KShapeKernels {

    /** Threads per workgroup of the row reductions. Must match {@link #RED_LOCAL} local arrays. */
    public static final int RED_LOCAL = 256;
    /** Tile edge of the shape-matrix kernel. */
    public static final int TILE = 16;

    private KShapeKernels() {
    }

    /** Zero-pads real rows of length L into complex rows of length N. */
    public static void loadComplex(KernelContext ctx, FloatArray src, FloatArray dst, int rows, int L, int N) {
        int id = ctx.globalIdx;
        if (id < rows * N) {
            int r = id / N;
            int t = id - r * N;
            float v = 0.0f;
            if (t < L) {
                v = src.get(r * L + t);
            }
            dst.set(2 * id, v);
            dst.set(2 * id + 1, 0.0f);
        }
    }

    /**
     * One radix-2 Stockham pass over a batch of complex rows. {@code tw} holds exp(-2 pi i t / N) for
     * t &lt; N/2; {@code conj} is 1 for the forward and -1 for the inverse (unscaled) transform.
     */
    public static void fftPass(KernelContext ctx, FloatArray src, FloatArray dst, FloatArray tw, int rows, int N, int ns, int twStep, float conj) {
        int id = ctx.globalIdx;
        int half = N / 2;
        if (id < rows * half) {
            int r = id / half;
            int j = id - r * half;
            int base = r * N;
            int k = j & (ns - 1);
            int t = k * twStep;
            float wr = tw.get(2 * t);
            float wi = conj * tw.get(2 * t + 1);
            int i0 = 2 * (base + j);
            int i1 = 2 * (base + j + half);
            float ar = src.get(i0);
            float ai = src.get(i0 + 1);
            float br = src.get(i1);
            float bi = src.get(i1 + 1);
            float vr = br * wr - bi * wi;
            float vi = br * wi + bi * wr;
            int d = (j / ns) * ns * 2 + k;
            int o0 = 2 * (base + d);
            int o1 = 2 * (base + d + ns);
            dst.set(o0, ar + vr);
            dst.set(o0 + 1, ai + vi);
            dst.set(o1, ar - vr);
            dst.set(o1 + 1, ai - vi);
        }
    }

    /** Alignment spectra: out[i,d] = FC[label(i), d] * conj(FX[i, d]). */
    public static void alignProduct(KernelContext ctx, FloatArray fx, FloatArray fc, IntArray labels, FloatArray out, int rows, int D, int N) {
        int id = ctx.globalIdx;
        if (id < rows * N) {
            int row = id / N;
            int f = id - row * N;
            int i = row / D;
            int d = row - i * D;
            int c = (labels.get(i) * D + d) * N + f;
            float a = fc.get(2 * c);
            float b = fc.get(2 * c + 1);
            float x = fx.get(2 * id);
            float y = fx.get(2 * id + 1);
            out.set(2 * id, a * x + b * y);
            out.set(2 * id + 1, b * x - a * y);
        }
    }

    /** Assignment spectra: out[i,j] = sum_d FX[i, d] * conj(FC[j, d]). */
    public static void assignProduct(KernelContext ctx, FloatArray fx, FloatArray fc, FloatArray out, int m, int k, int D, int N) {
        int id = ctx.globalIdx;
        if (id < m * k * N) {
            int row = id / N;
            int f = id - row * N;
            int i = row / k;
            int j = row - i * k;
            float re = 0.0f;
            float im = 0.0f;
            for (int d = 0; d < D; d++) {
                int xi = 2 * ((i * D + d) * N + f);
                int ci = 2 * ((j * D + d) * N + f);
                float a = fx.get(xi);
                float b = fx.get(xi + 1);
                float c = fc.get(ci);
                float e = fc.get(ci + 1);
                re += a * c + b * e;
                im += b * c - a * e;
            }
            out.set(2 * id, re);
            out.set(2 * id + 1, im);
        }
    }

    /**
     * Per alignment row: first lag of the maximum cross-correlation, turned into the zero-padded shift
     * that aligns the series with its centroid. One workgroup per row.
     */
    public static void lagArgmax(KernelContext ctx, FloatArray cc, IntArray shift, FloatArray xNorm, FloatArray cNorm, IntArray labels, int rows, int D, int L, int N) {
        float[] vals = ctx.allocateFloatLocalArray(RED_LOCAL);
        int[] idxs = ctx.allocateIntLocalArray(RED_LOCAL);
        int row = ctx.groupIdx;
        int lid = ctx.localIdx;
        int lags = 2 * L - 1;
        float best = Float.NEGATIVE_INFINITY;
        int bi = lags;
        if (row < rows) {
            for (int t = lid; t < lags; t += RED_LOCAL) {
                int p = t - (L - 1);
                if (p < 0) {
                    p += N;
                }
                float v = cc.get(2 * (row * N + p));
                if (v > best) {
                    best = v;
                    bi = t;
                }
            }
        }
        vals[lid] = best;
        idxs[lid] = bi;
        ctx.localBarrier();
        for (int s = RED_LOCAL / 2; s > 0; s /= 2) {
            if (lid < s) {
                float ov = vals[lid + s];
                int oi = idxs[lid + s];
                float mv = vals[lid];
                int mi = idxs[lid];
                if (ov > mv || (ov == mv && oi < mi)) {
                    vals[lid] = ov;
                    idxs[lid] = oi;
                }
            }
            ctx.localBarrier();
        }
        if (lid == 0 && row < rows) {
            int i = row / D;
            int d = row - i * D;
            float cn = cNorm.get(labels.get(i) * D + d);
            int s = 0;
            if (cn > 0.0f) {
                int arg = idxs[0];
                if (cn * xNorm.get(row) < 1e-9f) {
                    arg = 0;
                }
                s = arg + 1 - L;
            }
            shift.set(row, s);
        }
    }

    /** Per assignment row: maximum of the (unscaled) cross-correlation over the 2L-1 valid lags. */
    public static void lagMax(KernelContext ctx, FloatArray cc, FloatArray out, int rows, int L, int N) {
        float[] vals = ctx.allocateFloatLocalArray(RED_LOCAL);
        int row = ctx.groupIdx;
        int lid = ctx.localIdx;
        int lags = 2 * L - 1;
        float best = Float.NEGATIVE_INFINITY;
        if (row < rows) {
            for (int t = lid; t < lags; t += RED_LOCAL) {
                int p = t - (L - 1);
                if (p < 0) {
                    p += N;
                }
                best = TornadoMath.max(best, cc.get(2 * (row * N + p)));
            }
        }
        vals[lid] = best;
        ctx.localBarrier();
        for (int s = RED_LOCAL / 2; s > 0; s /= 2) {
            if (lid < s) {
                vals[lid] = TornadoMath.max(vals[lid], vals[lid + s]);
            }
            ctx.localBarrier();
        }
        if (lid == 0 && row < rows) {
            out.set(row, vals[0]);
        }
    }

    /**
     * Shifts every row by its alignment shift with zero padding (A), and z-normalises it with ddof=1 (Y).
     * Also writes the squared norm of A. One workgroup per row.
     */
    public static void alignZ(KernelContext ctx, FloatArray x, IntArray shift, FloatArray a, FloatArray y, FloatArray aNorm2, int rows, int L) {
        float[] acc = ctx.allocateFloatLocalArray(RED_LOCAL);
        float[] acc2 = ctx.allocateFloatLocalArray(RED_LOCAL);
        int row = ctx.groupIdx;
        int lid = ctx.localIdx;
        int s = 0;
        if (row < rows) {
            s = shift.get(row);
        }
        float sum = 0.0f;
        float sq = 0.0f;
        if (row < rows) {
            for (int t = lid; t < L; t += RED_LOCAL) {
                int src = t - s;
                float v = 0.0f;
                if (src >= 0 && src < L) {
                    v = x.get(row * L + src);
                }
                a.set(row * L + t, v);
                sum += v;
                sq += v * v;
            }
        }
        acc[lid] = sum;
        acc2[lid] = sq;
        ctx.localBarrier();
        for (int st = RED_LOCAL / 2; st > 0; st /= 2) {
            if (lid < st) {
                acc[lid] = acc[lid] + acc[lid + st];
                acc2[lid] = acc2[lid] + acc2[lid + st];
            }
            ctx.localBarrier();
        }
        float mean = acc[0] / L;
        float norm2 = acc2[0];
        ctx.localBarrier();
        float dev = 0.0f;
        if (row < rows) {
            for (int t = lid; t < L; t += RED_LOCAL) {
                float v = a.get(row * L + t) - mean;
                dev += v * v;
            }
        }
        acc[lid] = dev;
        ctx.localBarrier();
        for (int st = RED_LOCAL / 2; st > 0; st /= 2) {
            if (lid < st) {
                acc[lid] = acc[lid] + acc[lid + st];
            }
            ctx.localBarrier();
        }
        float std = TornadoMath.sqrt(acc[0] / (L - 1));
        if (row < rows) {
            for (int t = lid; t < L; t += RED_LOCAL) {
                float v = 0.0f;
                if (std > 0.0f) {
                    v = (a.get(row * L + t) - mean) / std;
                }
                y.set(row * L + t, v);
            }
            if (lid == 0) {
                aNorm2.set(row, norm2);
            }
        }
    }

    /**
     * Tiled shape matrices, one 16x16 output tile per workgroup. Per (cluster, dim) block, either the Gram
     * matrix Y_j Y_j^T (members x members, mode 1) or Y_j^T Y_j (L x L, mode 0), whichever is smaller; both
     * share the top eigenvector direction of the k-Shape matrix.
     *
     * <p>{@code tiles}: per tile (block, row0, col0); {@code blocks}: per block (memStart, n, r, mode, dim,
     * outOffset, uOffset); {@code meta[0]}: number of tiles.
     */
    public static void shapeTiles(KernelContext ctx, FloatArray y, IntArray members, IntArray tiles, IntArray blocks, IntArray meta, FloatArray out, int D, int L) {
        float[] as = ctx.allocateFloatLocalArray(TILE * TILE);
        float[] bs = ctx.allocateFloatLocalArray(TILE * TILE);
        int w = ctx.groupIdx;
        int lid = ctx.localIdx;
        int tx = lid % TILE;
        int ty = lid / TILE;
        int numTiles = meta.get(0);
        int b = 0;
        int row0 = 0;
        int col0 = 0;
        int active = 0;
        if (w < numTiles) {
            active = 1;
            b = tiles.get(3 * w);
            row0 = tiles.get(3 * w + 1);
            col0 = tiles.get(3 * w + 2);
        }
        int ms = blocks.get(7 * b);
        int n = blocks.get(7 * b + 1);
        int r = blocks.get(7 * b + 2);
        int gram = blocks.get(7 * b + 3);
        int d = blocks.get(7 * b + 4);
        int outOff = blocks.get(7 * b + 5);
        int kLen = n;
        if (gram == 1) {
            kLen = L;
        }
        if (active == 0) {
            kLen = 0;
        }
        float sum = 0.0f;
        for (int q0 = 0; q0 < kLen; q0 += TILE) {
            float va = 0.0f;
            float vb = 0.0f;
            if (gram == 1) {
                int t = q0 + tx;
                if (t < L) {
                    if (row0 + ty < r) {
                        va = y.get((members.get(ms + row0 + ty) * D + d) * L + t);
                    }
                    if (col0 + ty < r) {
                        vb = y.get((members.get(ms + col0 + ty) * D + d) * L + t);
                    }
                }
            } else {
                int p = q0 + ty;
                if (p < n) {
                    int base = (members.get(ms + p) * D + d) * L;
                    if (row0 + tx < L) {
                        va = y.get(base + row0 + tx);
                    }
                    if (col0 + tx < L) {
                        vb = y.get(base + col0 + tx);
                    }
                }
            }
            as[ty * TILE + tx] = va;
            bs[ty * TILE + tx] = vb;
            ctx.localBarrier();
            if (gram == 1) {
                for (int kk = 0; kk < TILE; kk++) {
                    sum += as[ty * TILE + kk] * bs[tx * TILE + kk];
                }
            } else {
                for (int kk = 0; kk < TILE; kk++) {
                    sum += as[kk * TILE + ty] * bs[kk * TILE + tx];
                }
            }
            ctx.localBarrier();
        }
        if (active == 1 && row0 + ty < r && col0 + tx < r) {
            out.set(outOff + (row0 + ty) * r + col0 + tx, sum);
        }
    }

    /**
     * Centroid direction per (cluster, dim) block: Y_j^T u for Gram blocks, u itself for L x L blocks.
     * Unnormalised.
     */
    public static void backProject(KernelContext ctx, FloatArray y, IntArray members, IntArray blocks, FloatArray u, FloatArray v, int numBlocks, int D, int L) {
        int id = ctx.globalIdx;
        if (id < numBlocks * L) {
            int b = id / L;
            int t = id - b * L;
            int ms = blocks.get(7 * b);
            int n = blocks.get(7 * b + 1);
            int gram = blocks.get(7 * b + 3);
            int d = blocks.get(7 * b + 4);
            int uOff = blocks.get(7 * b + 6);
            float s = 0.0f;
            if (n > 0) {
                if (gram == 1) {
                    for (int p = 0; p < n; p++) {
                        s += u.get(uOff + p) * y.get((members.get(ms + p) * D + d) * L + t);
                    }
                } else {
                    s = u.get(uOff + t);
                }
            }
            v.set(id, s);
        }
    }

    /** Per row: dot product of the aligned series with its cluster's (unnormalised) centroid direction. */
    public static void signDots(KernelContext ctx, FloatArray a, FloatArray v, IntArray labels, FloatArray dots, int rows, int D, int L) {
        float[] acc = ctx.allocateFloatLocalArray(RED_LOCAL);
        int row = ctx.groupIdx;
        int lid = ctx.localIdx;
        float s = 0.0f;
        if (row < rows) {
            int i = row / D;
            int d = row - i * D;
            int vb = (labels.get(i) * D + d) * L;
            for (int t = lid; t < L; t += RED_LOCAL) {
                s += a.get(row * L + t) * v.get(vb + t);
            }
        }
        acc[lid] = s;
        ctx.localBarrier();
        for (int st = RED_LOCAL / 2; st > 0; st /= 2) {
            if (lid < st) {
                acc[lid] = acc[lid] + acc[lid + st];
            }
            ctx.localBarrier();
        }
        if (lid == 0 && row < rows) {
            dots.set(row, acc[0]);
        }
    }
}
