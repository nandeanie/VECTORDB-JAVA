package com.vectordb.index;

import com.vectordb.core.DistanceFn;
import com.vectordb.core.ScoredId;
import com.vectordb.core.VectorItem;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Baseline O(N*d) exact search: scan every stored item and score it.
 * Slow, but always correct — useful as a correctness/speed reference
 * for the approximate HNSW index.
 */
public final class BruteForceIndex {

    private final List<VectorItem> items = new ArrayList<>();

    public synchronized void insert(VectorItem item) {
        items.add(item);
    }

    public synchronized void remove(int id) {
        items.removeIf(v -> v.id == id);
    }

    public synchronized List<ScoredId> knn(float[] query, int k, DistanceFn dist) {
        List<ScoredId> results = new ArrayList<>(items.size());
        for (VectorItem v : items) {
            results.add(new ScoredId(dist.distance(query, v.embedding), v.id));
        }
        Collections.sort(results);
        if (results.size() > k) {
            return new ArrayList<>(results.subList(0, k));
        }
        return results;
    }

    public synchronized int size() {
        return items.size();
    }
}
