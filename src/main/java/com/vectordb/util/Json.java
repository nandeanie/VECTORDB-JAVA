package com.vectordb.util;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Small, dependency-free JSON helper. Deliberately hand-rolled rather
 * than pulling in a JSON library: the API surface here is tiny (a
 * handful of endpoints exchanging flat objects and arrays), so a
 * ~150-line utility keeps the project self-contained with nothing to
 * install beyond a JDK.
 */
public final class Json {

    private Json() {
    }

    // ---------------------------------------------------------------
    // Writing
    // ---------------------------------------------------------------

    public static String escape(String s) {
        StringBuilder sb = new StringBuilder(s.length() + 8).append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default: sb.append(c);
            }
        }
        return sb.append('"').toString();
    }

    /** Builds a LinkedHashMap (preserves key order) from alternating key/value pairs. */
    public static Map<String, Object> object(Object... kv) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i + 1 < kv.length; i += 2) {
            m.put(String.valueOf(kv[i]), kv[i + 1]);
        }
        return m;
    }

    public static String toJson(Object value) {
        StringBuilder sb = new StringBuilder();
        write(value, sb);
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private static void write(Object value, StringBuilder sb) {
        if (value == null) {
            sb.append("null");
        } else if (value instanceof String) {
            sb.append(escape((String) value));
        } else if (value instanceof Boolean || value instanceof Integer || value instanceof Long) {
            sb.append(value);
        } else if (value instanceof Float || value instanceof Double) {
            sb.append(String.format(Locale.US, "%.6f", ((Number) value).doubleValue()));
        } else if (value instanceof float[]) {
            float[] arr = (float[]) value;
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(String.format(Locale.US, "%.4f", arr[i]));
            }
            sb.append(']');
        } else if (value instanceof int[]) {
            int[] arr = (int[]) value;
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(arr[i]);
            }
            sb.append(']');
        } else if (value instanceof Map) {
            Map<String, Object> map = (Map<String, Object>) value;
            sb.append('{');
            boolean first = true;
            for (Map.Entry<String, Object> e : map.entrySet()) {
                if (!first) sb.append(',');
                first = false;
                sb.append(escape(e.getKey())).append(':');
                write(e.getValue(), sb);
            }
            sb.append('}');
        } else if (value instanceof List) {
            List<?> list = (List<?>) value;
            sb.append('[');
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) sb.append(',');
                write(list.get(i), sb);
            }
            sb.append(']');
        } else {
            sb.append(escape(String.valueOf(value)));
        }
    }

    // ---------------------------------------------------------------
    // Reading (just enough to pull flat fields out of request bodies)
    // ---------------------------------------------------------------

    public static String extractString(String body, String key) {
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) return "";
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                char escaped = body.charAt(p);
                switch (escaped) {
                    case '"': result.append('"'); break;
                    case '\\': result.append('\\'); break;
                    case 'n': result.append('\n'); break;
                    case 'r': result.append('\r'); break;
                    case 't': result.append('\t'); break;
                    default: result.append(escaped);
                }
            } else {
                result.append(c);
            }
            p++;
        }
        return result.toString();
    }

    public static int extractInt(String body, String key, int defaultValue) {
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) return defaultValue;
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        int start = p;
        while (p < body.length() && (Character.isDigit(body.charAt(p)) || body.charAt(p) == '-')) p++;
        if (p == start) return defaultValue;
        try {
            return Integer.parseInt(body.substring(start, p));
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    /** Pulls a JSON number array field out of a body, e.g. {"embedding":[0.1,0.2,...]}. */
    public static float[] extractFloatArray(String body, String key) {
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) return new float[0];
        p = body.indexOf('[', p);
        if (p < 0) return new float[0];
        int depth = 1, e = p + 1;
        while (e < body.length() && depth > 0) {
            char c = body.charAt(e);
            if (c == '[') depth++;
            else if (c == ']') depth--;
            e++;
        }
        return parseCsvFloats(body.substring(p + 1, e - 1));
    }

    /** Parses a comma-separated list of floats, e.g. from a "v" query parameter. */
    public static float[] parseCsvFloats(String s) {
        if (s == null || s.isBlank()) return new float[0];
        String[] parts = s.split(",");
        List<Float> values = new ArrayList<>(parts.length);
        for (String part : parts) {
            try {
                values.add(Float.parseFloat(part.trim()));
            } catch (NumberFormatException ignored) {
                // skip malformed entries, matching the original's lenient parsing
            }
        }
        float[] arr = new float[values.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = values.get(i);
        return arr;
    }
}
