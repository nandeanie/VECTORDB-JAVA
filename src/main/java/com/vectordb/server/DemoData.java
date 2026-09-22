package com.vectordb.server;

import com.vectordb.core.DistanceFn;
import com.vectordb.core.DistanceMetrics;
import com.vectordb.db.VectorDatabase;

import java.util.Random;

/**
 * Generates a large deterministic benchmark dataset.
 *
 * 50,000 vectors
 * 16 dimensions
 * 4 semantic categories:
 *   CS, Math, Food, Sports
 *
 * Each category has a dominant region in the vector space,
 * while random noise prevents all vectors from being identical.
 */
final class DemoData {

    private DemoData() {
    }

    static void load(VectorDatabase db) {

        DistanceFn dist = DistanceMetrics.get("cosine");

        final int TOTAL_VECTORS = 50_000;
        final int DIMENSIONS = 16;
        final int PER_CATEGORY = TOTAL_VECTORS / 4;

        Random random = new Random(42);

        String[] categories = {
                "cs",
                "math",
                "food",
                "sports"
        };

        String[] topics = {
                "Algorithms and Data Structures",
                "Mathematics and Statistics",
                "Food and Cooking",
                "Sports and Games"
        };

        /*
         * Each category gets a different dominant region:
         *
         * CS      -> dimensions 0-3
         * Math    -> dimensions 4-7
         * Food    -> dimensions 8-11
         * Sports  -> dimensions 12-15
         */
        for (int category = 0; category < 4; category++) {

            for (int i = 0; i < PER_CATEGORY; i++) {

                float[] vector = new float[DIMENSIONS];

                int start = category * 4;

                for (int d = 0; d < DIMENSIONS; d++) {

                    /*
                     * Small background noise.
                     */
                    float value = 0.05f + random.nextFloat() * 0.10f;

                    /*
                     * Strong signal for this category.
                     */
                    if (d >= start && d < start + 4) {
                        value += 0.70f + random.nextFloat() * 0.25f;
                    }

                    /*
                     * A little cross-category variation.
                     */
                    value += (float) random.nextGaussian() * 0.025f;

                    vector[d] = Math.max(0.01f, value);
                }

                String id = topics[category]
                        + " document " + i;

                db.insert(
                        id,
                        categories[category],
                        vector,
                        dist
                );
            }
        }

        System.out.println(
                "Loaded " + TOTAL_VECTORS +
                " vectors (" + DIMENSIONS + "D)"
        );
    }
}
