/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.kshape;

/**
 * Eigenvector of the largest eigenvalue of a small dense symmetric matrix: Householder
 * tridiagonalisation and implicit QL for the eigenvalues (the public-domain JAMA algorithm, after EISPACK
 * tred2/tql2, with the orthogonal matrix stored transposed so that the inner loops run along rows), then
 * inverse iteration on the tridiagonal matrix for the top eigenvector. Larger matrices first try Lanczos,
 * which needs only matrix-vector products, and keep its answer only when its residual is at rounding level.
 */
final class SymmetricEigen {

    private SymmetricEigen() {
    }

    /**
     * @param a row-major n x n symmetric matrix (only read)
     * @return unit eigenvector of the largest eigenvalue (the last one in ascending order)
     */
    static double[] topEigenvector(float[] a, int off, int n) {
        if (n > LANCZOS_MIN) {
            double[] x = lanczosTop(a, off, n);
            if (x != null) {
                return x;
            }
        }
        return topEigenvectorDense(a, off, n);
    }

    private static final int LANCZOS_MIN = 32;
    private static final int LANCZOS_MAX_STEPS = 200;
    private static final double LANCZOS_TOL = 1e-13;

    /**
     * Top eigenpair by Lanczos with full reorthogonalisation. Returns null unless the Ritz residual
     * ||A x - theta x|| falls below 1e-13 ||A||, in which case the caller uses the dense solver.
     */
    static double[] lanczosTop(float[] a, int off, int n) {
        double[] m = new double[n * n];
        double anorm = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double val = 0.5 * ((double) a[off + i * n + j] + a[off + j * n + i]);
                m[i * n + j] = val;
                anorm = Math.max(anorm, Math.abs(val));
            }
        }
        if (anorm == 0) {
            return null;
        }
        anorm *= n; // bound on the 2-norm
        int maxSteps = Math.min(n, LANCZOS_MAX_STEPS);
        double[][] q = new double[maxSteps + 1][];
        double[] alpha = new double[maxSteps];
        double[] beta = new double[maxSteps];
        // deterministic start that is not orthogonal to the top eigenvector in practice
        double[] q0 = new double[n];
        long st = 88172645463325252L;
        double nq = 0;
        for (int i = 0; i < n; i++) {
            st ^= st << 13;
            st ^= st >>> 7;
            st ^= st << 17;
            q0[i] = ((st >>> 11) * 0x1.0p-53) - 0.5;
            nq += q0[i] * q0[i];
        }
        nq = Math.sqrt(nq);
        for (int i = 0; i < n; i++) {
            q0[i] /= nq;
        }
        q[0] = q0;
        double[] w = new double[n];
        for (int j = 0; j < maxSteps; j++) {
            double[] qj = q[j];
            for (int r = 0; r < n; r++) {
                double s = 0;
                int base = r * n;
                for (int c = 0; c < n; c++) {
                    s += m[base + c] * qj[c];
                }
                w[r] = s;
            }
            double al = 0;
            for (int i = 0; i < n; i++) {
                al += w[i] * qj[i];
            }
            alpha[j] = al;
            // full reorthogonalisation, twice
            for (int pass = 0; pass < 2; pass++) {
                for (int p = 0; p <= j; p++) {
                    double[] qp = q[p];
                    double h = 0;
                    for (int i = 0; i < n; i++) {
                        h += w[i] * qp[i];
                    }
                    for (int i = 0; i < n; i++) {
                        w[i] -= h * qp[i];
                    }
                }
            }
            double b = 0;
            for (int i = 0; i < n; i++) {
                b += w[i] * w[i];
            }
            b = Math.sqrt(b);
            beta[j] = b;
            int steps = j + 1;
            boolean exhausted = b <= 1e-14 * anorm || steps == maxSteps;
            if (exhausted || steps % 8 == 0) {
                // top Ritz pair of the tridiagonal T_steps
                double[][] v = new double[steps][steps];
                for (int i = 0; i < steps; i++) {
                    v[i][i] = 1.0;
                }
                double[] d = new double[steps];
                double[] e = new double[steps];
                for (int i = 0; i < steps; i++) {
                    d[i] = alpha[i];
                    if (i > 0) {
                        e[i] = beta[i - 1];
                    }
                }
                tql2(v, d, e, steps, true);
                int best = 0;
                for (int i = 1; i < steps; i++) {
                    if (d[i] >= d[best]) {
                        best = i;
                    }
                }
                double resid = Math.abs(b * v[best][steps - 1]);
                if (resid <= LANCZOS_TOL * anorm || exhausted) {
                    if (resid > LANCZOS_TOL * anorm && b > 1e-14 * anorm) {
                        return null;
                    }
                    double[] x = new double[n];
                    for (int p = 0; p < steps; p++) {
                        double sp = v[best][p];
                        double[] qp = q[p];
                        for (int i = 0; i < n; i++) {
                            x[i] += sp * qp[i];
                        }
                    }
                    double nx = 0;
                    for (int i = 0; i < n; i++) {
                        nx += x[i] * x[i];
                    }
                    nx = Math.sqrt(nx);
                    for (int i = 0; i < n; i++) {
                        x[i] /= nx;
                    }
                    return x;
                }
            }
            if (b <= 1e-14 * anorm) {
                return null;
            }
            double[] qn = new double[n];
            for (int i = 0; i < n; i++) {
                qn[i] = w[i] / b;
            }
            q[j + 1] = qn;
        }
        return null;
    }

    /** Dense path: tridiagonalisation, QL eigenvalues and inverse iteration. */
    static double[] topEigenvectorDense(float[] a, int off, int n) {
        double[][] v = load(a, off, n);
        double[] d = new double[n];
        double[] e = new double[n];
        tred2(v, d, e, n);
        // eigenvalues of the tridiagonal T only, then its top eigenvector by inverse iteration
        double[] dt = d.clone();
        double[] et = e.clone();
        tql2(v, d, e, n, false);
        double lambda = d[0];
        double scale = 0;
        for (int i = 0; i < n; i++) {
            lambda = Math.max(lambda, d[i]);
            scale = Math.max(scale, Math.abs(d[i]));
        }
        double[] z = inverseIteration(dt, et, n, lambda, scale);
        // back to the original basis: x = Q z, Q stored transposed in v
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double zi = z[i];
            double[] row = v[i];
            for (int k = 0; k < n; k++) {
                out[k] += zi * row[k];
            }
        }
        double nrm = 0;
        for (int k = 0; k < n; k++) {
            nrm += out[k] * out[k];
        }
        nrm = Math.sqrt(nrm);
        for (int k = 0; k < n; k++) {
            out[k] /= nrm;
        }
        return out;
    }

    /** Reference: all eigenvectors by QL, then the one of the largest eigenvalue. */
    static double[] topEigenvectorFull(float[] a, int off, int n) {
        double[][] v = load(a, off, n);
        double[] d = new double[n];
        double[] e = new double[n];
        tred2(v, d, e, n);
        tql2(v, d, e, n, true);
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (d[i] >= d[best]) {
                best = i;
            }
        }
        return v[best].clone();
    }

    private static double[][] load(float[] a, int off, int n) {
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // symmetrise: float32 sums are not exactly symmetric
                v[i][j] = 0.5 * ((double) a[off + i * n + j] + a[off + j * n + i]);
            }
        }
        return v;
    }

    /**
     * Eigenvector of the tridiagonal matrix (diagonal d, off-diagonal e[1..n-1] between i-1 and i) for the
     * eigenvalue lambda: three steps of inverse iteration, Gaussian elimination with partial pivoting.
     */
    private static double[] inverseIteration(double[] d, double[] e, int n, double lambda, double scale) {
        double tiny = Math.max(scale, Double.MIN_NORMAL) * 1e-15;
        double[] u0 = new double[n];
        double[] u1 = new double[n];
        double[] u2 = new double[n];
        double[] l = new double[n];
        boolean[] piv = new boolean[n];
        double c0 = d[0] - lambda;
        double c1 = n > 1 ? e[1] : 0;
        double c2 = 0;
        for (int i = 0; i < n - 1; i++) {
            double sub = e[i + 1];
            double nd = d[i + 1] - lambda;
            double ns = i + 2 < n ? e[i + 2] : 0;
            if (Math.abs(c0) >= Math.abs(sub)) {
                if (c0 == 0) {
                    c0 = tiny;
                }
                double mlt = sub / c0;
                u0[i] = c0;
                u1[i] = c1;
                u2[i] = c2;
                l[i] = mlt;
                c0 = nd - mlt * c1;
                c1 = ns - mlt * c2;
            } else {
                double mlt = c0 / sub;
                u0[i] = sub;
                u1[i] = nd;
                u2[i] = ns;
                l[i] = mlt;
                piv[i] = true;
                c0 = c1 - mlt * nd;
                c1 = c2 - mlt * ns;
            }
            c2 = 0;
        }
        u0[n - 1] = c0 == 0 ? tiny : c0;
        double[] z = new double[n];
        java.util.Arrays.fill(z, 1.0 / Math.sqrt(n));
        for (int it = 0; it < 3; it++) {
            for (int i = 0; i < n - 1; i++) {
                if (piv[i]) {
                    double t = z[i];
                    z[i] = z[i + 1];
                    z[i + 1] = t;
                }
                z[i + 1] -= l[i] * z[i];
            }
            for (int i = n - 1; i >= 0; i--) {
                double s = z[i];
                if (i + 1 < n) {
                    s -= u1[i] * z[i + 1];
                }
                if (i + 2 < n) {
                    s -= u2[i] * z[i + 2];
                }
                z[i] = s / u0[i];
            }
            double nrm = 0;
            for (int i = 0; i < n; i++) {
                nrm += z[i] * z[i];
            }
            nrm = Math.sqrt(nrm);
            for (int i = 0; i < n; i++) {
                z[i] /= nrm;
            }
        }
        return z;
    }

    private static void tred2(double[][] V, double[] d, double[] e, int n) {
        for (int j = 0; j < n; j++) {
            d[j] = V[j][n - 1];
        }
        for (int i = n - 1; i > 0; i--) {
            double scale = 0.0;
            double h = 0.0;
            for (int k = 0; k < i; k++) {
                scale += Math.abs(d[k]);
            }
            if (scale == 0.0) {
                e[i] = d[i - 1];
                for (int j = 0; j < i; j++) {
                    d[j] = V[j][i - 1];
                    V[j][i] = 0.0;
                    V[i][j] = 0.0;
                }
            } else {
                for (int k = 0; k < i; k++) {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }
                double f = d[i - 1];
                double g = Math.sqrt(h);
                if (f > 0) {
                    g = -g;
                }
                e[i] = scale * g;
                h = h - f * g;
                d[i - 1] = f - g;
                for (int j = 0; j < i; j++) {
                    e[j] = 0.0;
                }
                for (int j = 0; j < i; j++) {
                    f = d[j];
                    V[i][j] = f;
                    g = e[j] + V[j][j] * f;
                    for (int k = j + 1; k <= i - 1; k++) {
                        g += V[j][k] * d[k];
                        e[k] += V[j][k] * f;
                    }
                    e[j] = g;
                }
                f = 0.0;
                for (int j = 0; j < i; j++) {
                    e[j] /= h;
                    f += e[j] * d[j];
                }
                double hh = f / (h + h);
                for (int j = 0; j < i; j++) {
                    e[j] -= hh * d[j];
                }
                for (int j = 0; j < i; j++) {
                    f = d[j];
                    g = e[j];
                    for (int k = j; k <= i - 1; k++) {
                        V[j][k] -= (f * e[k] + g * d[k]);
                    }
                    d[j] = V[j][i - 1];
                    V[j][i] = 0.0;
                }
            }
            d[i] = h;
        }
        for (int i = 0; i < n - 1; i++) {
            V[i][n - 1] = V[i][i];
            V[i][i] = 1.0;
            double h = d[i + 1];
            if (h != 0.0) {
                for (int k = 0; k <= i; k++) {
                    d[k] = V[i + 1][k] / h;
                }
                for (int j = 0; j <= i; j++) {
                    double g = 0.0;
                    for (int k = 0; k <= i; k++) {
                        g += V[i + 1][k] * V[j][k];
                    }
                    for (int k = 0; k <= i; k++) {
                        V[j][k] -= g * d[k];
                    }
                }
            }
            for (int k = 0; k <= i; k++) {
                V[i + 1][k] = 0.0;
            }
        }
        for (int j = 0; j < n; j++) {
            d[j] = V[j][n - 1];
            V[j][n - 1] = 0.0;
        }
        V[n - 1][n - 1] = 1.0;
        e[0] = 0.0;
    }

    private static void tql2(double[][] V, double[] d, double[] e, int n, boolean vectors) {
        for (int i = 1; i < n; i++) {
            e[i - 1] = e[i];
        }
        e[n - 1] = 0.0;
        double f = 0.0;
        double tst1 = 0.0;
        double eps = Math.pow(2.0, -52.0);
        for (int l = 0; l < n; l++) {
            tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
            int m = l;
            while (m < n) {
                if (Math.abs(e[m]) <= eps * tst1) {
                    break;
                }
                m++;
            }
            if (m == n) {
                m = n - 1;
            }
            if (m > l) {
                int iter = 0;
                do {
                    iter++;
                    double g = d[l];
                    double p = (d[l + 1] - g) / (2.0 * e[l]);
                    double r = Math.hypot(p, 1.0);
                    if (p < 0) {
                        r = -r;
                    }
                    d[l] = e[l] / (p + r);
                    d[l + 1] = e[l] * (p + r);
                    double dl1 = d[l + 1];
                    double h = g - d[l];
                    for (int i = l + 2; i < n; i++) {
                        d[i] -= h;
                    }
                    f = f + h;
                    p = d[m];
                    double c = 1.0;
                    double c2 = c;
                    double c3 = c;
                    double el1 = e[l + 1];
                    double s = 0.0;
                    double s2 = 0.0;
                    for (int i = m - 1; i >= l; i--) {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * e[i];
                        h = c * p;
                        r = Math.hypot(p, e[i]);
                        e[i + 1] = s * r;
                        s = e[i] / r;
                        c = p / r;
                        p = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);
                        for (int k = 0; vectors && k < n; k++) {
                            h = V[i + 1][k];
                            V[i + 1][k] = s * V[i][k] + c * h;
                            V[i][k] = c * V[i][k] - s * h;
                        }
                    }
                    p = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;
                } while (Math.abs(e[l]) > eps * tst1 && iter < 300);
            }
            d[l] = d[l] + f;
            e[l] = 0.0;
        }
    }
}
