/*
 * Copyright (c) 2026 Beehive lab
 * SPDX-License-Identifier: MIT
 */
package org.beehive.kshape.iotdb;

import java.util.ArrayList;
import java.util.List;

import org.apache.iotdb.udf.api.UDTF;
import org.apache.iotdb.udf.api.access.Row;
import org.apache.iotdb.udf.api.collector.PointCollector;
import org.apache.iotdb.udf.api.customizer.config.UDTFConfigurations;
import org.apache.iotdb.udf.api.customizer.parameter.UDFParameterValidator;
import org.apache.iotdb.udf.api.customizer.parameter.UDFParameters;
import org.apache.iotdb.udf.api.customizer.strategy.RowByRowAccessStrategy;
import org.apache.iotdb.udf.api.exception.UDFException;
import org.apache.iotdb.udf.api.type.Type;

/**
 * GPU version of the IoTDB {@code cluster} UDF with {@code method=kshape}.
 *
 * <p>Splits the input series into non-overlapping windows of length {@code l} and clusters them with k-Shape
 * on a TornadoVM device. Takes the same parameters and gives the same output as {@code cluster}: {@code l},
 * {@code k}, {@code norm} (default true), {@code maxiter} (default 200) and {@code output} ({@code label}: one
 * label per window at the window's first timestamp; {@code centroid}: the k centroids concatenated, at
 * timestamps 0, 1, ...). {@code method} is optional and must be {@code kshape}.
 *
 * <p>The DataNode JVM must run with the TornadoVM flags. The compiled kernels are kept for the next query with
 * the same number of windows, window length and k.
 */
public class UDTFClusterGPU implements UDTF {

    private static final String OUTPUT_LABEL = "label";
    private static final String OUTPUT_CENTROID = "centroid";
    private static final int DEFAULT_MAX_ITER = 200;

    private static final Object LOCK = new Object();
    private static WindowKShapeTornado cached;

    private int l;
    private int k;
    private boolean norm;
    private int maxIter;
    private String output;

    private final List<Long> timestamps = new ArrayList<>();
    private final List<Double> values = new ArrayList<>();

    @Override
    public void validate(UDFParameterValidator validator) throws Exception {
        validator.validateInputSeriesNumber(1) //
                .validateInputSeriesDataType(0, Type.INT32, Type.INT64, Type.FLOAT, Type.DOUBLE) //
                .validate(x -> (int) x > 0, "Parameter l must be a positive integer.", validator.getParameters().getInt("l")) //
                .validate(x -> (int) x >= 2, "Parameter k must be at least 2.", validator.getParameters().getInt("k")) //
                .validate(x -> "kshape".equals(((String) x).toLowerCase()), "Parameter method must be kshape.",
                        validator.getParameters().getStringOrDefault("method", "kshape")) //
                .validate(x -> (int) x >= 1, "Parameter maxiter must be a positive integer.",
                        validator.getParameters().getIntOrDefault("maxiter", DEFAULT_MAX_ITER)) //
                .validate(x -> {
                    String o = ((String) x).toLowerCase();
                    return OUTPUT_LABEL.equals(o) || OUTPUT_CENTROID.equals(o);
                }, "Parameter output must be label or centroid.", validator.getParameters().getStringOrDefault("output", OUTPUT_LABEL));
    }

    @Override
    public void beforeStart(UDFParameters parameters, UDTFConfigurations configurations) throws Exception {
        output = parameters.getStringOrDefault("output", OUTPUT_LABEL).toLowerCase();
        configurations.setAccessStrategy(new RowByRowAccessStrategy()).setOutputDataType(OUTPUT_CENTROID.equals(output) ? Type.DOUBLE : Type.INT32);
        l = parameters.getInt("l");
        k = parameters.getInt("k");
        norm = parameters.getBooleanOrDefault("norm", true);
        maxIter = parameters.getIntOrDefault("maxiter", DEFAULT_MAX_ITER);
        timestamps.clear();
        values.clear();
    }

    @Override
    public void transform(Row row, PointCollector collector) throws Exception {
        if (row.isNull(0)) {
            return;
        }
        timestamps.add(row.getTime());
        switch (row.getDataType(0)) {
            case INT32 -> values.add((double) row.getInt(0));
            case INT64 -> values.add((double) row.getLong(0));
            case FLOAT -> values.add((double) row.getFloat(0));
            case DOUBLE -> values.add(row.getDouble(0));
            default -> throw new UDFException("Unsupported data type: " + row.getDataType(0));
        }
    }

    @Override
    public void terminate(PointCollector collector) throws Exception {
        int n = values.size();
        if (n < l) {
            throw new UDFException("Time series length must be at least l; got " + n + " points, l=" + l + ".");
        }
        int numWindows = n / l;
        if (numWindows < k) {
            throw new UDFException("Not enough non-overlapping windows: got " + numWindows + " windows, need at least k=" + k + ".");
        }
        double[][] windows = new double[numWindows][l];
        long[] windowStart = new long[numWindows];
        for (int w = 0; w < numWindows; w++) {
            windowStart[w] = timestamps.get(w * l);
            for (int j = 0; j < l; j++) {
                windows[w][j] = values.get(w * l + j);
            }
        }

        int[] labels;
        double[][] centroids;
        synchronized (LOCK) {
            WindowKShapeTornado ks = engine(numWindows, l, k);
            ks.fit(windows, norm, maxIter);
            labels = ks.getLabels().clone();
            centroids = ks.getCentroids();
        }

        if (OUTPUT_LABEL.equals(output)) {
            for (int w = 0; w < numWindows; w++) {
                collector.putInt(windowStart[w], labels[w]);
            }
        } else {
            long t = 0;
            for (double[] row : centroids) {
                for (double val : row) {
                    collector.putDouble(t++, val);
                }
            }
        }
    }

    private static WindowKShapeTornado engine(int m, int len, int clusters) {
        if (cached != null && cached.windows() == m && cached.length() == len && cached.clusters() == clusters) {
            return cached;
        }
        if (cached != null) {
            cached.close();
            cached = null;
        }
        cached = new WindowKShapeTornado(m, len, clusters);
        return cached;
    }
}
