package com.vectordb.core;

/**
 * A candidate neighbor: its distance from the query and its item id.
 * Ordered by distance ascending (nearest first).
 */
public final class ScoredId implements Comparable<ScoredId> {

    public final float distance;
    public final int id;

    public ScoredId(float distance, int id) {
        this.distance = distance;
        this.id = id;
    }

    @Override
    public int compareTo(ScoredId other) {
        return Float.compare(this.distance, other.distance);
    }

    @Override
    public String toString() {
        return "(" + id + ", " + distance + ")";
    }
}
