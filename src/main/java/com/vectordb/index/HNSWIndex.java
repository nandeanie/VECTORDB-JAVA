package com.vectordb.index;

import com.vectordb.core.DistanceFn;
import com.vectordb.core.ScoredId;
import com.vectordb.core.VectorItem;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

/**
 * Hierarchical Navigable Small World graph — the same family of
 * algorithm used by production vector databases such as Pinecone,
 * Weaviate, Chroma, and Milvus.
 *
 * <p>Every node is inserted into a randomly chosen "top layer" (layer
 * 0 always included). Layer 0 holds every node with many short-range
 * connections; higher layers hold exponentially fewer nodes with
 * longer-range connections, acting like an express lane down to the
 * right neighbourhood before a fine-grained search at layer 0.
 *
 * <ul>
 *   <li><b>Insert:</b> greedily descend from the top layer down to the
 *       node's assigned level, then at each layer from there down to 0
 *       run a beam search and connect to the M nearest neighbours
 *       (bidirectionally, pruning over-full neighbour lists).</li>
 *   <li><b>Search:</b> the same greedy descent, then a beam search at
 *       layer 0 expanded to {@code ef} candidates.</li>
 * </ul>
 */
public final class HNSWIndex {

    private static final class Node {
        final VectorItem item;
        final int maxLayer;
        final List<List<Integer>> neighbors; // neighbors.get(layer) -> ids

        Node(VectorItem item, int maxLayer) {
            this.item = item;
            this.maxLayer = maxLayer;
            this.neighbors = new ArrayList<>(maxLayer + 1);
            for (int i = 0; i <= maxLayer; i++) {
                neighbors.add(new ArrayList<>());
            }
        }
    }

    private final Map<Integer, Node> graph = new HashMap<>();
    private final int m;
    private final int m0;
    private final int efConstruction;
    private final double levelMultiplier;
    private final Random rng;

    private int topLayer = -1;
    private int entryPoint = -1;

    public HNSWIndex() {
        this(16, 200);
    }

    public HNSWIndex(int m, int efConstruction) {
        this.m = m;
        this.m0 = 2 * m;
        this.efConstruction = efConstruction;
        this.levelMultiplier = 1.0 / Math.log(m);
        this.rng = new Random(42); // fixed seed: deterministic, reproducible demo runs
    }

    private int randomLevel() {
        double u = rng.nextDouble();
        if (u <= 0.0) {
            u = Double.MIN_VALUE;
        }
        return (int) Math.floor(-Math.log(u) * levelMultiplier);
    }

    public synchronized void insert(VectorItem item, DistanceFn dist) {
        int id = item.id;
        int level = randomLevel();
        Node node = new Node(item, level);
        graph.put(id, node);

        if (entryPoint == -1) {
            entryPoint = id;
            topLayer = level;
            return;
        }

        int ep = entryPoint;

        // Descend greedily from the top layer to just above this node's level.
        for (int lc = topLayer; lc > level; lc--) {
            Node epNode = graph.get(ep);
            if (lc < epNode.neighbors.size()) {
                List<ScoredId> found = searchLayer(item.embedding, ep, 1, lc, dist);
                if (!found.isEmpty()) {
                    ep = found.get(0).id;
                }
            }
        }

        // From min(topLayer, level) down to 0: beam search + connect.
        List<ScoredId> lastFound = Collections.emptyList();
        for (int lc = Math.min(topLayer, level); lc >= 0; lc--) {
            List<ScoredId> found = searchLayer(item.embedding, ep, efConstruction, lc, dist);
            lastFound = found;

            int maxM = (lc == 0) ? m0 : m;
            List<Integer> selected = selectNeighbors(found, maxM);
            node.neighbors.set(lc, selected);

            for (int neighborId : selected) {
                Node neighbor = graph.get(neighborId);
                if (neighbor == null) {
                    continue;
                }
                while (neighbor.neighbors.size() <= lc) {
                    neighbor.neighbors.add(new ArrayList<>());
                }
                List<Integer> conn = neighbor.neighbors.get(lc);
                conn.add(id);

                if (conn.size() > maxM) {
                    // Too many edges: keep only the maxM closest to this neighbor.
                    List<ScoredId> ranked = new ArrayList<>(conn.size());
                    for (int candidateId : conn) {
                        Node candidate = graph.get(candidateId);
                        if (candidate != null) {
                            ranked.add(new ScoredId(
                                    dist.distance(neighbor.item.embedding, candidate.item.embedding),
                                    candidateId));
                        }
                    }
                    Collections.sort(ranked);
                    conn.clear();
                    for (int i = 0; i < maxM && i < ranked.size(); i++) {
                        conn.add(ranked.get(i).id);
                    }
                }
            }

            if (!found.isEmpty()) {
                ep = found.get(0).id;
            }
        }

        if (level > topLayer) {
            topLayer = level;
            entryPoint = id;
        }
    }

    /** Beam search within a single layer, starting from {@code entry}. */
    private List<ScoredId> searchLayer(float[] query, int entry, int ef, int layer, DistanceFn dist) {
        Set<Integer> visited = new HashSet<>();
        PriorityQueue<ScoredId> candidates = new PriorityQueue<>(); // min-heap: closest first
        PriorityQueue<ScoredId> found = new PriorityQueue<>(Collections.reverseOrder()); // max-heap

        Node entryNode = graph.get(entry);
        float d0 = dist.distance(query, entryNode.item.embedding);
        visited.add(entry);
        candidates.offer(new ScoredId(d0, entry));
        found.offer(new ScoredId(d0, entry));

        while (!candidates.isEmpty()) {
            ScoredId current = candidates.poll();
            if (found.size() >= ef && current.distance > found.peek().distance) {
                break;
            }

            Node currentNode = graph.get(current.id);
            if (currentNode == null || layer >= currentNode.neighbors.size()) {
                continue;
            }

            for (int neighborId : currentNode.neighbors.get(layer)) {
                if (visited.contains(neighborId) || !graph.containsKey(neighborId)) {
                    continue;
                }
                visited.add(neighborId);
                Node neighborNode = graph.get(neighborId);
                float d = dist.distance(query, neighborNode.item.embedding);
                if (found.size() < ef || d < found.peek().distance) {
                    candidates.offer(new ScoredId(d, neighborId));
                    found.offer(new ScoredId(d, neighborId));
                    if (found.size() > ef) {
                        found.poll();
                    }
                }
            }
        }

        List<ScoredId> results = new ArrayList<>(found);
        Collections.sort(results);
        return results;
    }

    private List<Integer> selectNeighbors(List<ScoredId> candidates, int maxM) {
        List<Integer> selected = new ArrayList<>();
        for (int i = 0; i < Math.min(candidates.size(), maxM); i++) {
            selected.add(candidates.get(i).id);
        }
        return selected;
    }

    public synchronized List<ScoredId> knn(float[] query, int k, int ef, DistanceFn dist) {
        if (entryPoint == -1) {
            return Collections.emptyList();
        }
        int ep = entryPoint;
        for (int lc = topLayer; lc > 0; lc--) {
            Node epNode = graph.get(ep);
            if (lc < epNode.neighbors.size()) {
                List<ScoredId> found = searchLayer(query, ep, 1, lc, dist);
                if (!found.isEmpty()) {
                    ep = found.get(0).id;
                }
            }
        }
        List<ScoredId> found = searchLayer(query, ep, Math.max(ef, k), 0, dist);
        if (found.size() > k) {
            found = new ArrayList<>(found.subList(0, k));
        }
        return found;
    }

    public synchronized void remove(int id) {
        if (!graph.containsKey(id)) {
            return;
        }
        for (Node node : graph.values()) {
            for (List<Integer> layer : node.neighbors) {
                layer.remove(Integer.valueOf(id));
            }
        }
        if (entryPoint == id) {
            entryPoint = -1;
            for (int otherId : graph.keySet()) {
                if (otherId != id) {
                    entryPoint = otherId;
                    break;
                }
            }
        }
        graph.remove(id);
    }

    public synchronized int size() {
        return graph.size();
    }

    // --- introspection, used by the /hnsw-info endpoint for the graph viewer ---

    public static final class NodeView {
        public final int id;
        public final String metadata;
        public final String category;
        public final int maxLayer;

        NodeView(int id, String metadata, String category, int maxLayer) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.maxLayer = maxLayer;
        }
    }

    public static final class EdgeView {
        public final int src;
        public final int dst;
        public final int layer;

        EdgeView(int src, int dst, int layer) {
            this.src = src;
            this.dst = dst;
            this.layer = layer;
        }
    }

    public static final class GraphInfo {
        public int topLayer;
        public int nodeCount;
        public int[] nodesPerLayer;
        public int[] edgesPerLayer;
        public final List<NodeView> nodes = new ArrayList<>();
        public final List<EdgeView> edges = new ArrayList<>();
    }

    public synchronized GraphInfo getInfo() {
        GraphInfo info = new GraphInfo();
        info.topLayer = topLayer;
        info.nodeCount = graph.size();

        int layerCount = Math.max(topLayer + 1, 1);
        info.nodesPerLayer = new int[layerCount];
        info.edgesPerLayer = new int[layerCount];

        for (Map.Entry<Integer, Node> entry : graph.entrySet()) {
            int id = entry.getKey();
            Node node = entry.getValue();
            info.nodes.add(new NodeView(id, node.item.metadata, node.item.category, node.maxLayer));

            for (int lc = 0; lc <= node.maxLayer && lc < layerCount; lc++) {
                info.nodesPerLayer[lc]++;
                if (lc < node.neighbors.size()) {
                    for (int neighborId : node.neighbors.get(lc)) {
                        // Count each undirected edge once.
                        if (id < neighborId) {
                            info.edgesPerLayer[lc]++;
                            info.edges.add(new EdgeView(id, neighborId, lc));
                        }
                    }
                }
            }
        }
        return info;
    }
}
