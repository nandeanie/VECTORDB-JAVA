package com.vectordb.index;

import com.vectordb.core.DistanceFn;
import com.vectordb.core.ScoredId;
import com.vectordb.core.VectorItem;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Classic binary space-partitioning tree. Splits alternate across
 * dimensions as you descend, and search prunes any subtree whose
 * splitting hyperplane is already farther away than the current
 * worst candidate in the result set.
 *
 * <p>Works well up to a few dozen dimensions; degrades toward
 * brute-force behaviour as dimensionality grows (the "curse of
 * dimensionality" — see {@link HNSWIndex} for the alternative that
 * doesn't share this weakness).
 */
public final class KDTreeIndex {

    private static final class Node {
        final VectorItem item;
        Node left;
        Node right;

        Node(VectorItem item) {
            this.item = item;
        }
    }

    private Node root;
    private final int dims;

    public KDTreeIndex(int dims) {
        this.dims = dims;
    }

    public synchronized void insert(VectorItem item) {
        root = insert(root, item, 0);
    }

    private Node insert(Node node, VectorItem item, int depth) {
        if (node == null) {
            return new Node(item);
        }
        int axis = depth % dims;
        if (item.embedding[axis] < node.item.embedding[axis]) {
            node.left = insert(node.left, item, depth + 1);
        } else {
            node.right = insert(node.right, item, depth + 1);
        }
        return node;
    }

    public synchronized List<ScoredId> knn(float[] query, int k, DistanceFn dist) {
        // Max-heap keyed on distance: worst-of-the-best sits on top so it
        // can be evicted the moment something closer is found.
        PriorityQueue<ScoredId> heap = new PriorityQueue<>(Collections.reverseOrder());
        search(root, query, k, 0, dist, heap);
        List<ScoredId> results = new ArrayList<>(heap);
        Collections.sort(results);
        return results;
    }

    private void search(Node node, float[] query, int k, int depth, DistanceFn dist,
                         PriorityQueue<ScoredId> heap) {
        if (node == null) {
            return;
        }
        float d = dist.distance(query, node.item.embedding);
        if (heap.size() < k || d < heap.peek().distance) {
            heap.offer(new ScoredId(d, node.item.id));
            if (heap.size() > k) {
                heap.poll();
            }
        }

        int axis = depth % dims;
        float diff = query[axis] - node.item.embedding[axis];
        Node nearSide = diff < 0 ? node.left : node.right;
        Node farSide = diff < 0 ? node.right : node.left;

        search(nearSide, query, k, depth + 1, dist, heap);

        // Only descend into the far side if the splitting plane itself
        // could still contain something closer than our current worst.
        if (heap.size() < k || Math.abs(diff) < heap.peek().distance) {
            search(farSide, query, k, depth + 1, dist, heap);
        }
    }

    public synchronized void rebuild(Collection<VectorItem> items) {
        root = null;
        for (VectorItem item : items) {
            insert(item);
        }
    }
}
