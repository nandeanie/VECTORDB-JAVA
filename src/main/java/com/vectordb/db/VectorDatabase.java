package com.vectordb.db;

import com.vectordb.core.DistanceFn;
import com.vectordb.core.DistanceMetrics;
import com.vectordb.core.ScoredId;
import com.vectordb.core.VectorItem;
import com.vectordb.index.BruteForceIndex;
import com.vectordb.index.HNSWIndex;
import com.vectordb.index.KDTreeIndex;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Owns the demo-vector collection and mirrors every insert/delete into
 * all three indexes at once, so they can be searched and benchmarked
 * side by side.
 */
public final class VectorDatabase {

    public final int dims;

    private final Map<Integer, VectorItem> store = new LinkedHashMap<>();
    private final BruteForceIndex bruteForce = new BruteForceIndex();
    private final KDTreeIndex kdTree;
    private final HNSWIndex hnsw = new HNSWIndex(16, 200);
    private final Object lock = new Object();
    private int nextId = 1;

    public VectorDatabase(int dims) {
        this.dims = dims;
        this.kdTree = new KDTreeIndex(dims);
    }

    public int insert(String metadata, String category, float[] embedding, DistanceFn dist) {
        synchronized (lock) {
            VectorItem item = new VectorItem(nextId++, metadata, category, embedding);
            store.put(item.id, item);
            bruteForce.insert(item);
            kdTree.insert(item);
            hnsw.insert(item, dist);
            return item.id;
        }
    }

    public boolean remove(int id) {
        synchronized (lock) {
            if (!store.containsKey(id)) {
                return false;
            }
            store.remove(id);
            bruteForce.remove(id);
            hnsw.remove(id);
            kdTree.rebuild(store.values());
            return true;
        }
    }

    public static final class Hit {
        public final int id;
        public final String metadata;
        public final String category;
        public final float[] embedding;
        public final float distance;

        Hit(int id, String metadata, String category, float[] embedding, float distance) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.embedding = embedding;
            this.distance = distance;
        }
    }

    public static final class SearchResult {
        public List<Hit> hits;
        public long latencyMicros;
        public String algo;
        public String metric;
    }

    public SearchResult search(float[] query, int k, String metric, String algo) {
        synchronized (lock) {
            DistanceFn dfn = DistanceMetrics.get(metric);
            long start = System.nanoTime();

            List<ScoredId> raw;
            switch (algo) {
                case "bruteforce":
                    raw = bruteForce.knn(query, k, dfn);
                    break;
                case "kdtree":
                    raw = kdTree.knn(query, k, dfn);
                    break;
                default:
                    raw = hnsw.knn(query, k, 50, dfn);
            }

            long micros = (System.nanoTime() - start) / 1_000;

            SearchResult result = new SearchResult();
            result.latencyMicros = micros;
            result.algo = algo;
            result.metric = metric;
            result.hits = new ArrayList<>(raw.size());
            for (ScoredId s : raw) {
                VectorItem item = store.get(s.id);
                if (item != null) {
                    result.hits.add(new Hit(item.id, item.metadata, item.category, item.embedding, s.distance));
                }
            }
            return result;
        }
    }

    public static final class BenchmarkResult {
        public long bruteforceUs;
        public long kdtreeUs;
        public long hnswUs;
        public int itemCount;
    }

    public BenchmarkResult benchmark(float[] query, int k, String metric) {
        synchronized (lock) {
            DistanceFn dfn = DistanceMetrics.get(metric);
            BenchmarkResult result = new BenchmarkResult();

            long t0 = System.nanoTime();
            bruteForce.knn(query, k, dfn);
            result.bruteforceUs = (System.nanoTime() - t0) / 1_000;

            long t1 = System.nanoTime();
            kdTree.knn(query, k, dfn);
            result.kdtreeUs = (System.nanoTime() - t1) / 1_000;

            long t2 = System.nanoTime();
            hnsw.knn(query, k, 50, dfn);
            result.hnswUs = (System.nanoTime() - t2) / 1_000;

            result.itemCount = store.size();
            return result;
        }
    }

    public List<VectorItem> all() {
        synchronized (lock) {
            return new ArrayList<>(store.values());
        }
    }

    public HNSWIndex.GraphInfo hnswInfo() {
        synchronized (lock) {
            return hnsw.getInfo();
        }
    }

    public int size() {
        synchronized (lock) {
            return store.size();
        }
    }
}
