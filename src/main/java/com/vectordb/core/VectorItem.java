package com.vectordb.core;

/**
 * A single stored vector: an id, free-form metadata/category labels,
 * and the embedding itself.
 */
public final class VectorItem {

    public final int id;
    public final String metadata;
    public final String category;
    public final float[] embedding;

    public VectorItem(int id, String metadata, String category, float[] embedding) {
        this.id = id;
        this.metadata = metadata;
        this.category = category;
        this.embedding = embedding;
    }
}
