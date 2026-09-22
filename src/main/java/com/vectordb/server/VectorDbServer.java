package com.vectordb.server;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import com.vectordb.client.OllamaClient;
import com.vectordb.core.DistanceMetrics;
import com.vectordb.core.VectorItem;
import com.vectordb.db.DocumentDatabase;
import com.vectordb.db.VectorDatabase;
import com.vectordb.index.HNSWIndex;
import com.vectordb.text.TextChunker;
import com.vectordb.util.Json;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;

/**
 * Entry point and REST layer. Routes are registered by hand rather
 * than pulling in a web framework — the whole surface is about a
 * dozen small endpoints, so {@code com.sun.net.httpserver.HttpServer}
 * (built into the JDK) keeps the project dependency-free, same spirit
 * as the original's single-header HTTP library.
 */
public final class VectorDbServer {

    private static final int DIMS = 16; // demo vector dimensionality

    public static void main(String[] args) throws IOException {
        VectorDatabase db = new VectorDatabase(DIMS);
        DocumentDatabase docDb = new DocumentDatabase();
        OllamaClient ollama = new OllamaClient();

        DemoData.load(db);

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) {
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);
        }

        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", 8080), 0);
        server.setExecutor(Executors.newCachedThreadPool());

        registerDemoVectorRoutes(server, db);
        registerDocumentAndRagRoutes(server, docDb, ollama);
        registerStatusAndStaticRoutes(server, ollama);

        server.start();
    }

    // -----------------------------------------------------------------
    // Demo vector endpoints
    // -----------------------------------------------------------------

    private static void registerDemoVectorRoutes(HttpServer server, VectorDatabase db) {

        server.createContext("/search", exchange -> {
            withCors(exchange);
            if (!requireMethod(exchange, "GET")) return;

            Map<String, String> params = queryParams(exchange.getRequestURI());
            float[] q = Json.parseCsvFloats(params.get("v"));
            if (q.length != DIMS) {
                sendJson(exchange, 400, Json.object("error", "need " + DIMS + "D vector"));
                return;
            }
            int k = parseIntOr(params.get("k"), 5);
            String metric = params.getOrDefault("metric", "cosine");
            String algo = params.getOrDefault("algo", "hnsw");

            VectorDatabase.SearchResult out = db.search(q, k, metric, algo);

            List<Object> hits = new ArrayList<>();
            for (VectorDatabase.Hit h : out.hits) {
                hits.add(Json.object(
                        "id", h.id,
                        "metadata", h.metadata,
                        "category", h.category,
                        "distance", h.distance,
                        "embedding", h.embedding));
            }
            sendJson(exchange, 200, Json.object(
                    "results", hits,
                    "latencyUs", out.latencyMicros,
                    "algo", out.algo,
                    "metric", out.metric));
        });

        server.createContext("/insert", exchange -> {
            withCors(exchange);
            if (!requireMethod(exchange, "POST")) return;

            String body = readBody(exchange);
            String meta = Json.extractString(body, "metadata");
            String cat = Json.extractString(body, "category");
            float[] emb = Json.extractFloatArray(body, "embedding");

            if (meta.isEmpty() || emb.length != DIMS) {
                sendJson(exchange, 400, Json.object("error", "invalid body"));
                return;
            }
            int id = db.insert(meta, cat, emb, DistanceMetrics.get("cosine"));
            sendJson(exchange, 200, Json.object("id", id));
        });

        server.createContext("/delete/", exchange -> {
            withCors(exchange);
            if (!requireMethod(exchange, "DELETE")) return;

            Integer id = trailingIntSegment(exchange.getRequestURI(), "/delete/");
            if (id == null) {
                sendJson(exchange, 400, Json.object("error", "invalid id"));
                return;
            }
            boolean ok = db.remove(id);
            sendJson(exchange, 200, Json.object("ok", ok));
        });

        server.createContext("/items", exchange -> {
            withCors(exchange);
            if (!requireMethod(exchange, "GET")) return;

            List<Object> items = new ArrayList<>();
            for (VectorItem v : db.all()) {
                items.add(Json.object(
                        "id", v.id,
                        "metadata", v.metadata,
                        "category", v.category,
                        "embedding", v.embedding));
            }
            sendJson(exchange, 200, items);
        });

        server.createContext("/benchmark", exchange -> {
            withCors(exchange);
            if (!requireMethod(exchange, "GET")) return;

            Map<String, String> params = queryParams(exchange.getRequestURI());
            float[] q = Json.parseCsvFloats(params.get("v"));
            if (q.length != DIMS) {
                sendJson(exchange, 400, Json.object("error", "need " + DIMS + "D vector"));
                return;
            }
            int k = parseIntOr(params.get("k"), 5);
            String metric = params.getOrDefault("metric", "cosine");

            VectorDatabase.BenchmarkResult b = db.benchmark(q, k, metric);
            sendJson(exchange, 200, Json.object(
                    "bruteforceUs", b.bruteforceUs,
                    "kdtreeUs", b.kdtreeUs,
                    "hnswUs", b.hnswUs,
                    "itemCount", b.itemCount));
        });

        server.createContext("/hnsw-info", exchange -> {
            withCors(exchange);
            if (!requireMethod(exchange, "GET")) return;

            HNSWIndex.GraphInfo gi = db.hnswInfo();

            List<Object> nodes = new ArrayList<>();
            for (HNSWIndex.NodeView n : gi.nodes) {
                nodes.add(Json.object(
                        "id", n.id, "metadata", n.metadata,
                        "category", n.category, "maxLyr", n.maxLayer));
            }
            List<Object> edges = new ArrayList<>();
            for (HNSWIndex.EdgeView e : gi.edges) {
                edges.add(Json.object("src", e.src, "dst", e.dst, "lyr", e.layer));
            }

            sendJson(exchange, 200, Json.object(
                    "topLayer", gi.topLayer,
                    "nodeCount", gi.nodeCount,
                    "nodesPerLayer", gi.nodesPerLayer,
                    "edgesPerLayer", gi.edgesPerLayer,
                    "nodes", nodes,
                    "edges", edges));
        });
    }

    // -----------------------------------------------------------------
    // Document + RAG endpoints
    // -----------------------------------------------------------------

    private static void registerDocumentAndRagRoutes(HttpServer server, DocumentDatabase docDb, OllamaClient ollama) {

        // POST /doc/insert {"title":"...","text":"..."} -> chunk, embed each chunk, store
        server.createContext("/doc/insert", exchange -> {
            withCors(exchange);
            if (!requireMethod(exchange, "POST")) return;

            String body = readBody(exchange);
            String title = Json.extractString(body, "title");
            String text = Json.extractString(body, "text");
            if (title.isEmpty() || text.isEmpty()) {
                sendJson(exchange, 400, Json.object("error", "need title and text"));
                return;
            }

            List<String> chunks = TextChunker.chunk(text, 250, 30);
            List<Integer> ids = new ArrayList<>();
            for (int i = 0; i < chunks.size(); i++) {
                float[] emb = ollama.embed(chunks.get(i));
                if (emb.length == 0) {
                    sendJson(exchange, 502, Json.object("error",
                            "Ollama unavailable. Install from https://ollama.com then run: "
                                    + "ollama pull nomic-embed-text && ollama pull llama3.2"));
                    return;
                }
                String chunkTitle = (chunks.size() > 1)
                        ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                        : title;
                ids.add(docDb.insert(chunkTitle, chunks.get(i), emb));
            }

            sendJson(exchange, 200, Json.object(
                    "ids", ids, "chunks", chunks.size(), "dims", docDb.getDims()));
        });

        server.createContext("/doc/delete/", exchange -> {
            withCors(exchange);
            if (!requireMethod(exchange, "DELETE")) return;

            Integer id = trailingIntSegment(exchange.getRequestURI(), "/doc/delete/");
            if (id == null) {
                sendJson(exchange, 400, Json.object("error", "invalid id"));
                return;
            }
            boolean ok = docDb.remove(id);
            sendJson(exchange, 200, Json.object("ok", ok));
        });

        server.createContext("/doc/list", exchange -> {
            withCors(exchange);
            if (!requireMethod(exchange, "GET")) return;

            List<Object> docs = new ArrayList<>();
            for (DocumentDatabase.DocItem d : docDb.all()) {
                String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "\u2026" : d.text;
                int words = countWords(d.text);
                docs.add(Json.object(
                        "id", d.id, "title", d.title, "preview", preview, "words", words));
            }
            sendJson(exchange, 200, docs);
        });

        // POST /doc/search {"question":"...","k":3} -> fast retrieval only, for the UI's context viewer
        server.createContext("/doc/search", exchange -> {
            withCors(exchange);
            if (!requireMethod(exchange, "POST")) return;

            String body = readBody(exchange);
            String question = Json.extractString(body, "question");
            int k = Json.extractInt(body, "k", 3);
            if (question.isEmpty()) {
                sendJson(exchange, 400, Json.object("error", "need question"));
                return;
            }
            float[] qEmb = ollama.embed(question);
            if (qEmb.length == 0) {
                sendJson(exchange, 502, Json.object("error", "Ollama unavailable"));
                return;
            }
            List<DocumentDatabase.ScoredDoc> hits = docDb.search(qEmb, k, 0.7f);
            List<Object> contexts = new ArrayList<>();
            for (DocumentDatabase.ScoredDoc s : hits) {
                contexts.add(Json.object("id", s.doc.id, "title", s.doc.title, "distance", s.distance));
            }
            sendJson(exchange, 200, Json.object("contexts", contexts));
        });

        // POST /doc/ask {"question":"...","k":3} -> full RAG: embed -> retrieve -> generate
        server.createContext("/doc/ask", exchange -> {
            withCors(exchange);
            if (!requireMethod(exchange, "POST")) return;

            String body = readBody(exchange);
            String question = Json.extractString(body, "question");
            int k = Json.extractInt(body, "k", 3);
            if (question.isEmpty()) {
                sendJson(exchange, 400, Json.object("error", "need question"));
                return;
            }

            float[] qEmb = ollama.embed(question);
            if (qEmb.length == 0) {
                sendJson(exchange, 502, Json.object("error",
                        "Ollama unavailable. Install from https://ollama.com then run: "
                                + "ollama pull nomic-embed-text && ollama pull llama3.2"));
                return;
            }

            List<DocumentDatabase.ScoredDoc> hits = docDb.search(qEmb, k, 0.7f);
            if (hits.isEmpty()) {
                sendJson(exchange, 200, Json.object(
                        "answer", "I don't have any relevant documents to answer that. "
                                + "Try inserting some documents first.",
                        "contexts", List.of()));
                return;
            }

            StringBuilder context = new StringBuilder();
            List<Object> contexts = new ArrayList<>();
            for (DocumentDatabase.ScoredDoc s : hits) {
                context.append("[").append(s.doc.title).append("]\n").append(s.doc.text).append("\n\n");
                contexts.add(Json.object("id", s.doc.id, "title", s.doc.title, "distance", s.distance));
            }

            String prompt = "Answer the question using ONLY the context below. "
                    + "If the answer isn't in the context, say so.\n\n"
                    + "Context:\n" + context + "\nQuestion: " + question + "\nAnswer:";

            String answer = ollama.generate(prompt);
            sendJson(exchange, 200, Json.object("answer", answer, "contexts", contexts));
        });
    }

    // -----------------------------------------------------------------
    // Status + static frontend
    // -----------------------------------------------------------------

    private static void registerStatusAndStaticRoutes(HttpServer server, OllamaClient ollama) {

        server.createContext("/status", exchange -> {
            withCors(exchange);
            if (!requireMethod(exchange, "GET")) return;

            boolean up = ollama.isAvailable();
            sendJson(exchange, 200, Json.object(
                    "ollama", up ? "online" : "offline",
                    "embedModel", ollama.embedModel,
                    "genModel", ollama.genModel));
        });

        server.createContext("/", new StaticFileHandler());
    }

    // -----------------------------------------------------------------
    // Shared HTTP plumbing
    // -----------------------------------------------------------------

    private static void withCors(HttpExchange exchange) {
        exchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        exchange.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");
    }

    /** Handles the OPTIONS preflight and enforces the expected verb; sends 204/405 and returns false if not matched. */
    private static boolean requireMethod(HttpExchange exchange, String method) throws IOException {
        if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
            exchange.sendResponseHeaders(204, -1);
            exchange.close();
            return false;
        }
        if (!method.equalsIgnoreCase(exchange.getRequestMethod())) {
            sendJson(exchange, 405, Json.object("error", "method not allowed"));
            return false;
        }
        return true;
    }

    private static String readBody(HttpExchange exchange) throws IOException {
        try (InputStream is = exchange.getRequestBody()) {
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static void sendJson(HttpExchange exchange, int status, Object payload) throws IOException {
        byte[] bytes = Json.toJson(payload).getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static Map<String, String> queryParams(URI uri) {
        Map<String, String> params = new LinkedHashMap<>();
        String query = uri.getRawQuery();
        if (query == null || query.isEmpty()) return params;
        for (String pair : query.split("&")) {
            int eq = pair.indexOf('=');
            String key = eq >= 0 ? pair.substring(0, eq) : pair;
            String value = eq >= 0 ? pair.substring(eq + 1) : "";
            params.put(urlDecode(key), urlDecode(value));
        }
        return params;
    }

    private static String urlDecode(String s) {
        return URLDecoder.decode(s, StandardCharsets.UTF_8);
    }

    private static int parseIntOr(String s, int fallback) {
        if (s == null) return fallback;
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static Integer trailingIntSegment(URI uri, String prefix) {
        String path = uri.getPath();
        if (!path.startsWith(prefix)) return null;
        String rest = path.substring(prefix.length());
        int slash = rest.indexOf('/');
        if (slash >= 0) rest = rest.substring(0, slash);
        try {
            return Integer.parseInt(rest);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static int countWords(String text) {
        String trimmed = text.trim();
        if (trimmed.isEmpty()) return 0;
        return trimmed.split("\\s+").length;
    }

    /** Serves the bundled static frontend (index.html and friends) from the classpath. */
    private static final class StaticFileHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            withCors(exchange);
            String path = exchange.getRequestURI().getPath();
            if (path.equals("/")) path = "/index.html";

            String resourcePath = "/static" + path;
            InputStream in = getClass().getResourceAsStream(resourcePath);
            if (in == null) {
                byte[] notFound = "404 not found".getBytes(StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(404, notFound.length);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(notFound);
                }
                return;
            }

            exchange.getResponseHeaders().set("Content-Type", contentTypeFor(path));
            byte[] bytes;
            try (in) {
                bytes = in.readAllBytes();
            }
            exchange.sendResponseHeaders(200, bytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        }

        private String contentTypeFor(String path) {
            if (path.endsWith(".html")) return "text/html; charset=utf-8";
            if (path.endsWith(".css")) return "text/css; charset=utf-8";
            if (path.endsWith(".js")) return "application/javascript; charset=utf-8";
            if (path.endsWith(".json")) return "application/json; charset=utf-8";
            if (path.endsWith(".svg")) return "image/svg+xml";
            return "application/octet-stream";
        }
    }
}
