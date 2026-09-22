package com.vectordb.client;

import com.vectordb.util.Json;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;

/**
 * Thin wrapper over Ollama's local REST API.
 * Install: https://ollama.com
 * Models:  ollama pull mxbai-embed-large
 *          ollama pull gemma3:4b
 */
public final class OllamaClient {

    public String embedModel = "mxbai-embed-large";
    public String genModel = "gemma3:4b";

    private final String baseUrl;
    private final HttpClient client;

    public OllamaClient() {
        this("127.0.0.1", 11434);
    }

    public OllamaClient(String host, int port) {
        this.baseUrl = "http://" + host + ":" + port;
        this.client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(3))
                .build();
    }

    public boolean isAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/api/tags"))
                    .timeout(Duration.ofSeconds(2))
                    .GET()
                    .build();
            HttpResponse<Void> res = client.send(req, HttpResponse.BodyHandlers.discarding());
            return res.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    /** Returns an empty array if Ollama isn't running or the model isn't pulled. */
    public float[] embed(String text) {
        try {
            String body = Json.toJson(Json.object("model", embedModel, "prompt", text));
            HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/api/embeddings"))
                    .timeout(Duration.ofSeconds(30))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) {
                return new float[0];
            }
            return Json.extractFloatArray(res.body(), "embedding");
        } catch (Exception e) {
            return new float[0];
        }
    }

    /** Returns a human-readable error string if Ollama is unavailable, instead of throwing. */
    public String generate(String prompt) {
        try {
            Map<String, Object> payload = Json.object(
                    "model", genModel,
                    "prompt", prompt,
                    "stream", Boolean.FALSE);
            String body = Json.toJson(payload);
            HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/api/generate"))
                    .timeout(Duration.ofSeconds(180)) // local LLMs can be slow on CPU
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) {
                return "ERROR: Ollama unavailable. Run: ollama serve";
            }
            return Json.extractString(res.body(), "response");
        } catch (Exception e) {
            return "ERROR: Ollama unavailable. Run: ollama serve";
        }
    }
}
