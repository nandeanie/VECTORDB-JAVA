package com.vectordb.core;

/** Cosine, Euclidean, and Manhattan distance, plus a lookup by name. */
public final class DistanceMetrics {

    private DistanceMetrics() {
    }

    public static float euclidean(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return (float) Math.sqrt(sum);
    }

    public static float cosine(float[] a, float[] b) {
        float dot = 0f, normA = 0f, normB = 0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA < 1e-9f || normB < 1e-9f) {
            return 1.0f;
        }
        return 1.0f - dot / (float) (Math.sqrt(normA) * Math.sqrt(normB));
    }

    public static float manhattan(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            sum += Math.abs(a[i] - b[i]);
        }
        return sum;
    }

    public static DistanceFn get(String name) {
        if (name == null) {
            return DistanceMetrics::cosine;
        }
        switch (name) {
            case "cosine":
                return DistanceMetrics::cosine;
            case "manhattan":
                return DistanceMetrics::manhattan;
            case "euclidean":
                return DistanceMetrics::euclidean;
            default:
                return DistanceMetrics::cosine;
        }
    }
}
