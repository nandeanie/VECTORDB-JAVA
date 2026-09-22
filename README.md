# VectorDB Engine — Build a Vector Database from Scratch in Java

A fully working **Vector Database** built from scratch in plain Java (JDK only,
no frameworks) with a web UI.
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side,
plus a **RAG pipeline** powered by a local LLM via Ollama.

> An educational project showing how production vector databases like Pinecone,
> Weaviate, and Chroma actually work under the hood.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

No external Java libraries are required — the HTTP server, JSON handling, and
Ollama client are all built on the standard JDK (`com.sun.net.httpserver`,
`java.net.http`), so there's nothing to fetch beyond a JDK itself.

---

## How It Works

```
Your Text
    |
    v
Ollama (nomic-embed-text)          <- converts text to a 768-dimensional vector
    |
    v
HNSW Index (Java)                  <- indexes the vector in a multilayer graph
    |
    v
Semantic Search                    <- finds nearest neighbors in vector space
    |
    v
Ollama (llama3.2)                  <- reads retrieved chunks, generates an answer
    |
    v
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by
Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each
layer is progressively sparser — searches start at the top layer and zoom in,
achieving O(log N) complexity instead of O(N) for brute force.

---

## Prerequisites

1. **JDK 17 or newer** (JDK 21 recommended)
2. **Ollama** (runs the local AI models)

No Maven/Gradle install is required — the build script drives `javac` directly.

---

## Step-by-Step Setup

### Step 1 — Install a JDK

- **Windows/macOS/Linux:** download from <https://adoptium.net> (Temurin, JDK 21 LTS)
  or `winget install EclipseAdoptium.Temurin.21.JDK` on Windows.
- Verify:
  ```
  java -version
  javac -version
  ```

### Step 2 — Install Ollama

1. Go to **<https://ollama.com>** and download the installer for your OS.
2. Pull the two required models:
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```
3. Verify:
   ```
   ollama list
   ```

> **Minimum specs for Ollama:** 8GB RAM recommended. The models use ~3GB total.

### Step 3 — Clone / download this project

```
git clone <your-repo-url> VectorDB-Java
cd VectorDB-Java
```

### Step 4 — Build

**macOS / Linux:**
```
./build.sh
```

**Windows (PowerShell or cmd):**
```
run.bat
```
(`run.bat` compiles automatically on first run.)

This produces compiled classes plus the bundled frontend under `out/`.

### Step 5 — Run

**Terminal 1** — start Ollama if it isn't already running:
```
ollama serve
```

**Terminal 2** — start the server:
```
./run.sh
```
(or `run.bat` on Windows)

You should see:
```
=== VectorDB Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

**Open your browser** to:
```
http://localhost:8080
```

---

## Using the Application

Open <http://localhost:8080>. Use the left menu (bottom bar on phones) to move between pages.

- **Search** - type an idea (or click a suggestion such as `sushi`) to find the closest sample topics.
  Results show a match bar, and the **Topic map** highlights them. *Search options* lets you change the
  method (HNSW / KD-Tree / Brute force) and distance measure, and **Compare speed** benchmarks all three.
- **Topics** - browse, filter, add and remove the sample topics that Search uses.
- **Documents** - add notes or articles; they are chunked and embedded with Ollama. Needs Ollama running.
- **Ask AI** - ask questions; the answer is generated from your documents, with the sources shown. Needs Ollama running.

The status box in the menu shows whether Ollama is reachable.

---

## REST API Reference

The server exposes a full REST API at `http://localhost:8080`.

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all stored documents |
| `DELETE` | `/doc/delete/:id` | — | Delete document chunk |
| `POST` | `/doc/search` | `{"question":"...","k":3}` | Retrieve matching chunks only |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status and model info |

### Example: Search via curl

```
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example: Ask a question via curl

```
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Project Structure

```
VectorDB-Java/
├── build.sh / run.sh / run.bat        <- build & launch scripts
├── src/main/java/com/vectordb/
│   ├── core/          <- VectorItem, ScoredId, DistanceFn, DistanceMetrics
│   ├── index/          <- BruteForceIndex, KDTreeIndex, HNSWIndex
│   ├── db/              <- VectorDatabase, DocumentDatabase
│   ├── client/          <- OllamaClient (embeddings + generation)
│   ├── text/             <- TextChunker
│   ├── util/              <- Json (dependency-free JSON read/write)
│   └── server/             <- VectorDbServer (main + REST routes), DemoData
└── src/main/resources/static/
    └── index.html      <- frontend (scatter plot, chat UI, benchmark)
```

### Architecture

```
BruteForceIndex     O(N*d)      Exact, baseline
KDTreeIndex          O(log N)    Exact, axis-aligned partitioning
HNSWIndex             O(log N)    Approximate, multilayer small-world graph

VectorDatabase        Unified interface over all 3 (16D demo vectors)
DocumentDatabase       HNSW-only index for real Ollama embeddings (768D)
OllamaClient            HTTP client -> /api/embeddings + /api/generate
```

---

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)

Nodes are inserted into a multilayer graph. Each node randomly gets assigned
a maximum layer. Layer 0 has all nodes with many connections; higher layers
have exponentially fewer nodes with longer-range connections.

**Insert:** Start at the top layer, greedily find the nearest node, drop a
layer, repeat. At each layer from the assigned max layer down to 0, run a
beam search (`efConstruction = 200`) and connect to the M nearest neighbors
bidirectionally.

**Search:** Same greedy descent from the top layer. At layer 0, expand to
`ef` nearest candidates using a priority queue.

**Why it's fast:** The upper layers act like a highway — you quickly reach
the right neighbourhood, then zoom in at layer 0.

### KD-Tree

Binary space partitioning. Each node splits space along one dimension
(cycling through all dimensions). Search prunes entire subtrees when the
closest possible point in that subtree can't beat the current best.

**Weakness:** Degrades in high dimensions (curse of dimensionality). Works
well for ≤20D, becomes close to brute force at 768D.

### Why HNSW Wins at High Dimensions

KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions,
almost all the space sits near the boundary of the hypersphere — no
subtrees get pruned. HNSW's graph-based approach doesn't have this problem.

---

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` in header | Run `ollama serve` in a terminal |
| Embedding takes forever | Ollama is downloading the model on first use, wait ~2 min |
| `javac: command not found` | Install a JDK (not just a JRE) — see Step 1 |
| Port 8080 already in use | Free the port, or change `8080` in `VectorDbServer.main()` |
| LLM answer is slow | Normal — `llama3.2` takes 10-30s on a laptop CPU. Use `llama3.2:1b` for faster answers |

### Use a Smaller/Faster LLM

```
ollama pull llama3.2:1b
```

Then edit `src/main/java/com/vectordb/client/OllamaClient.java`:

```java
public String genModel = "llama3.2:1b";   // change this
```

Rebuild and restart.

---

## License

MIT — use this however you want.
