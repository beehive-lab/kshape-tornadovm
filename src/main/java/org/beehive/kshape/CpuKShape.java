/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.kshape;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * k-Shape on the CPU in double precision, a port of kshape-python's NumPy version ({@code kshape.core._kshape}
 * with zero initial centroids) for univariate series: the CPU reference, and a fallback when the device
 * fails. It starts from given initial labels and re-seeds an empty cluster with the series picked by
 * {@link KShapeTornado.EmptyPick}, as {@link KShapeTornado} does, so both give the same clustering.
 *
 * <p>Cross-correlations use a radix-2 FFT of size 2^ceil(log2(2L - 1)). The shape of a cluster is the top
 * eigenvector of Y^T Y for its z-normalised aligned members Y (their rows have zero mean, so the centring
 * projection of the reference leaves the matrix unchanged), taken as Y^T u for the top eigenvector u of Y Y^T
 * when the cluster has fewer members than the window length.
 */
public final class CpuKShape {

    private final double[][] x;
    private final int m;
    private final int L;
    private final int k;
    private final int nfft;
    private final double[][] xRe;
    private final double[][] xIm;
    private final double[] xNorm;

    public CpuKShape(double[][] series, int k) {
        this.x = series;
        this.m = series.length;
        this.L = series[0].length;
        this.k = k;
        int n = 1;
        while (n < 2 * L - 1) {
            n <<= 1;
        }
        nfft = n;
        xRe = new double[m][];
        xIm = new double[m][];
        xNorm = new double[m];
        IntStream.range(0, m).parallel().forEach(i -> {
            double[] re = Arrays.copyOf(x[i], nfft);
            double[] im = new double[nfft];
            fft(re, im, false);
            xRe[i] = re;
            xIm[i] = im;
            double s = 0;
            for (double v : x[i]) {
                s += v * v;
            }
            xNorm[i] = Math.sqrt(s);
        });
    }

    /** Labels after at most {@code maxIter} iterations from {@code init}. */
    public int[] fit(int[] init, int maxIter) {
        int[] idx = init.clone();
        double[][] cent = new double[k][L];
        KShapeTornado.EmptyPick pick = new KShapeTornado.EmptyPick(m);
        for (int it = 0; it < maxIter; it++) {
            for (int j = 0; j < k; j++) {
                cent[j] = extractShape(idx, j, cent[j], pick);
            }
            double[][] cRe = new double[k][];
            double[][] cIm = new double[k][];
            double[] cNorm = new double[k];
            for (int j = 0; j < k; j++) {
                cRe[j] = Arrays.copyOf(cent[j], nfft);
                cIm[j] = new double[nfft];
                fft(cRe[j], cIm[j], false);
                double s = 0;
                for (double v : cent[j]) {
                    s += v * v;
                }
                cNorm[j] = Math.sqrt(s);
            }
            int[] next = new int[m];
            IntStream.range(0, m).parallel().forEach(i -> {
                int arg = 0;
                double best = Double.POSITIVE_INFINITY;
                for (int j = 0; j < k; j++) {
                    double d = 1 - maxNcc(xRe[i], xIm[i], xNorm[i], cRe[j], cIm[j], cNorm[j]);
                    if (d < best) {
                        best = d;
                        arg = j;
                    }
                }
                next[i] = arg;
            });
            boolean same = Arrays.equals(idx, next);
            idx = next;
            if (same) {
                break;
            }
        }
        return idx;
    }

    /** The cross-correlation of x and y, normalised, at every shift (kshape-python's _ncc_c_3dim, D = 1). */
    private double[] ncc(double[] aRe, double[] aIm, double aNorm, double[] bRe, double[] bIm, double bNorm) {
        double den = aNorm * bNorm;
        if (den < 1e-9) {
            den = Double.POSITIVE_INFINITY;
        }
        double[] re = new double[nfft];
        double[] im = new double[nfft];
        for (int f = 0; f < nfft; f++) {
            // a * conj(b)
            re[f] = aRe[f] * bRe[f] + aIm[f] * bIm[f];
            im[f] = aIm[f] * bRe[f] - aRe[f] * bIm[f];
        }
        fft(re, im, true);
        // concatenate cc[-(L - 1):] and cc[:L]
        double[] cc = new double[2 * L - 1];
        for (int s = 0; s < L - 1; s++) {
            cc[s] = re[nfft - (L - 1) + s] / den;
        }
        for (int s = 0; s < L; s++) {
            cc[L - 1 + s] = re[s] / den;
        }
        return cc;
    }

    private double maxNcc(double[] aRe, double[] aIm, double aNorm, double[] bRe, double[] bIm, double bNorm) {
        double best = Double.NEGATIVE_INFINITY;
        for (double v : ncc(aRe, aIm, aNorm, bRe, bIm, bNorm)) {
            best = Math.max(best, v);
        }
        return best;
    }

    private double[] extractShape(int[] idx, int j, double[] cur, KShapeTornado.EmptyPick pick) {
        boolean zero = true;
        for (double v : cur) {
            zero &= v == 0;
        }
        int[] members = IntStream.range(0, m).filter(i -> idx[i] == j).toArray();
        if (members.length == 0) {
            return x[pick.next()].clone();
        }
        double[][] a = new double[members.length][];
        if (zero) {
            for (int p = 0; p < members.length; p++) {
                a[p] = x[members[p]];
            }
        } else {
            double[] cRe = Arrays.copyOf(cur, nfft);
            double[] cIm = new double[nfft];
            fft(cRe, cIm, false);
            double cn = 0;
            for (double v : cur) {
                cn += v * v;
            }
            double cNorm = Math.sqrt(cn);
            IntStream.range(0, members.length).parallel().forEach(p -> {
                int i = members[p];
                // _sbd(cur, x_i): shift x_i to where it best matches the centroid, zero-padded
                double[] cc = ncc(cRe, cIm, cNorm, xRe[i], xIm[i], xNorm[i]);
                int arg = 0;
                for (int s = 1; s < cc.length; s++) {
                    if (cc[s] > cc[arg]) {
                        arg = s;
                    }
                }
                int shift = (arg + 1) - L;
                a[p] = rollZeropad(x[i], shift);
            });
        }
        // y = zscore(a, axis=1, ddof=1)
        int n = members.length;
        double[][] y = new double[n][L];
        for (int p = 0; p < n; p++) {
            double mu = 0;
            for (double v : a[p]) {
                mu += v;
            }
            mu /= L;
            double q = 0;
            for (double v : a[p]) {
                q += (v - mu) * (v - mu);
            }
            double sd = Math.sqrt(q / (L - 1));
            for (int t = 0; t < L; t++) {
                y[p][t] = sd > 0 ? (a[p][t] - mu) / sd : 0;
            }
        }
        double[] centroid = topShape(y);
        double d1 = 0;
        double d2 = 0;
        for (int p = 0; p < n; p++) {
            double s1 = 0;
            double s2 = 0;
            for (int t = 0; t < L; t++) {
                s1 += (a[p][t] - centroid[t]) * (a[p][t] - centroid[t]);
                s2 += (a[p][t] + centroid[t]) * (a[p][t] + centroid[t]);
            }
            d1 += Math.sqrt(s1);
            d2 += Math.sqrt(s2);
        }
        if (d1 >= d2) {
            for (int t = 0; t < L; t++) {
                centroid[t] = -centroid[t];
            }
        }
        // zscore(centroid, ddof=1)
        double mu = 0;
        for (double v : centroid) {
            mu += v;
        }
        mu /= L;
        double q = 0;
        for (double v : centroid) {
            q += (v - mu) * (v - mu);
        }
        double sd = Math.sqrt(q / (L - 1));
        for (int t = 0; t < L; t++) {
            centroid[t] = sd > 0 ? (centroid[t] - mu) / sd : 0;
        }
        return centroid;
    }

    /** Unit top eigenvector of Y^T Y (L x L), through the smaller of Y^T Y and Y Y^T. */
    private double[] topShape(double[][] y) {
        int n = y.length;
        if (n >= L) {
            float[] s = new float[L * L];
            IntStream.range(0, L).parallel().forEach(a -> {
                for (int b = a; b < L; b++) {
                    double v = 0;
                    for (double[] row : y) {
                        v += row[a] * row[b];
                    }
                    s[a * L + b] = (float) v;
                    s[b * L + a] = (float) v;
                }
            });
            return SymmetricEigen.topEigenvector(s, 0, L);
        }
        float[] g = new float[n * n];
        IntStream.range(0, n).parallel().forEach(p -> {
            for (int q = p; q < n; q++) {
                double v = 0;
                for (int t = 0; t < L; t++) {
                    v += y[p][t] * y[q][t];
                }
                g[p * n + q] = (float) v;
                g[q * n + p] = (float) v;
            }
        });
        double[] u = SymmetricEigen.topEigenvector(g, 0, n);
        double[] c = new double[L];
        double nrm = 0;
        for (int t = 0; t < L; t++) {
            double v = 0;
            for (int p = 0; p < n; p++) {
                v += y[p][t] * u[p];
            }
            c[t] = v;
            nrm += v * v;
        }
        nrm = Math.sqrt(nrm);
        for (int t = 0; t < L; t++) {
            c[t] = nrm > 0 ? c[t] / nrm : 0;
        }
        return c;
    }

    /** NumPy's roll with zero fill (kshape-python's roll_zeropad), for 1-d arrays. */
    static double[] rollZeropad(double[] a, int shift) {
        int n = a.length;
        double[] r = new double[n];
        if (shift == 0) {
            return a.clone();
        }
        if (Math.abs(shift) > n) {
            return r;
        }
        if (shift < 0) {
            int s = -shift;
            System.arraycopy(a, s, r, 0, n - s);
        } else {
            System.arraycopy(a, 0, r, shift, n - shift);
        }
        return r;
    }

    /** In-place radix-2 FFT (inverse scaled by 1/n). */
    static void fft(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) {
                double t = re[i];
                re[i] = re[j];
                re[j] = t;
                t = im[i];
                im[i] = im[j];
                im[j] = t;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = 2 * Math.PI / len * (inverse ? 1 : -1);
            double wr = Math.cos(ang);
            double wi = Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                double cr = 1;
                double ci = 0;
                for (int j = 0; j < len / 2; j++) {
                    int u = i + j;
                    int v = i + j + len / 2;
                    double vr = re[v] * cr - im[v] * ci;
                    double vi = re[v] * ci + im[v] * cr;
                    re[v] = re[u] - vr;
                    im[v] = im[u] - vi;
                    re[u] += vr;
                    im[u] += vi;
                    double nr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = nr;
                }
            }
        }
        if (inverse) {
            for (int i = 0; i < n; i++) {
                re[i] /= n;
                im[i] /= n;
            }
        }
    }
}
