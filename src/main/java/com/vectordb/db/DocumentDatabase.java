package com.vectordb.db;

import com.vectordb.core.DistanceMetrics;
import com.vectordb.core.ScoredId;
import com.vectordb.core.VectorItem;
import com.vectordb.index.BruteForceIndex;
import com.vectordb.index.HNSWIndex;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Stores document chunks and their real (Ollama-generated) embeddings.
 * Dimensionality is discovered from the first insert rather than fixed
 * up front, since it depends on whichever embedding model is loaded.
 */
public final class DocumentDatabase {

    public static final class DocItem {
        public final int id;
        public final String title;
        public final String text;
        public final float[] embedding;

        DocItem(int id, String title, String text, float[] embedding) {
            this.id = id;
            this.title = title;
            this.text = text;
            this.embedding = embedding;
        }
    }

    public static final class ScoredDoc {
        public final float distance;
        public final DocItem doc;

        ScoredDoc(float distance, DocItem doc) {
            this.distance = distance;
            this.doc = doc;
        }
    }

    private final Map<Integer, DocItem> store = new LinkedHashMap<>();
    private final HNSWIndex hnsw = new HNSWIndex(16, 200);
    private final BruteForceIndex bruteForce = new BruteForceIndex(); // fallback while the set is small
    private final Object lock = new Object();
    private int nextId = 1;
    private int dims = 0;

    public int insert(String title, String text, float[] embedding) {
        synchronized (lock) {
            if (dims == 0) {
                dims = embedding.length;
            }
            DocItem item = new DocItem(nextId++, title, text, embedding);
            store.put(item.id, item);

            VectorItem vi = new VectorItem(item.id, title, "doc", embedding);
            hnsw.insert(vi, DistanceMetrics::cosine);
            bruteForce.insert(vi);
            return item.id;
        }
    }

    /** Top-k most similar chunks, filtered to those within {@code maxDistance}. */
    public List<ScoredDoc> search(float[] query, int k, float maxDistance) {
        synchronized (lock) {
            if (store.isEmpty()) {
                return Collections.emptyList();
            }
            List<ScoredId> raw = (store.size() < 10)
                    ? bruteForce.knn(query, k, DistanceMetrics::cosine)
                    : hnsw.knn(query, k, 50, DistanceMetrics::cosine);

            List<ScoredDoc> out = new ArrayList<>();
            for (ScoredId s : raw) {
                DocItem doc = store.get(s.id);
                if (doc != null && s.distance <= maxDistance) {
                    out.add(new ScoredDoc(s.distance, doc));
                }
            }
            return out;
        }
    }

    public boolean remove(int id) {
        synchronized (lock) {
            if (!store.containsKey(id)) {
                return false;
            }
            store.remove(id);
            hnsw.remove(id);
            bruteForce.remove(id);
            return true;
        }
    }

    public List<DocItem> all() {
        synchronized (lock) {
            return new ArrayList<>(store.values());
        }
    }

    public int size() {
        synchronized (lock) {
            return store.size();
        }
    }

    public int getDims() {
        return dims;
    }
}
