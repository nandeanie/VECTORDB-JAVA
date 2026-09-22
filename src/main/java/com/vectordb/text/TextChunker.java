package com.vectordb.text;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Splits long text into overlapping word-count windows for embedding. */
public final class TextChunker {

    private TextChunker() {
    }

    public static List<String> chunk(String text, int chunkWords, int overlapWords) {
        String trimmed = text.trim();
        if (trimmed.isEmpty()) {
            return Collections.emptyList();
        }
        String[] words = trimmed.split("\\s+");
        if (words.length <= chunkWords) {
            return List.of(text);
        }

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) sb.append(' ');
                sb.append(words[j]);
            }
            chunks.add(sb.toString());
            if (end == words.length) break;
        }
        return chunks;
    }
}
