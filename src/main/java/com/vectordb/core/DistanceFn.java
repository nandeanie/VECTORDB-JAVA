package com.vectordb.core;

/** A distance/dissimilarity function between two equal-length embeddings. */
@FunctionalInterface
public interface DistanceFn {
    float distance(float[] a, float[] b);
}
