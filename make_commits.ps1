function Commit-FileInChunks {
    param(
        [string]$File,
        [string]$MsgPrefix,
        [int]$Chunks = 3
    )

    $lines = Get-Content $File
    $total = $lines.Count
    $step = [math]::Ceiling($total / $Chunks)

    for ($i = 1; $i -le $Chunks; $i++) {
        $end = [math]::Min($i * $step, $total)
        $lines[0..($end-1)] | Set-Content $File
        git add $File
        git commit -m "$MsgPrefix (part $i/$Chunks)" --quiet
        Start-Sleep -Seconds 1
    }
}

git add .gitignore, run.sh, run.bat, build.sh
git commit -m "Initial project scaffold with build scripts" --quiet

Commit-FileInChunks "src/main/java/com/vectordb/core/VectorItem.java" "Add VectorItem data structure" 3
Commit-FileInChunks "src/main/java/com/vectordb/core/ScoredId.java" "Add ScoredId result wrapper" 3
Commit-FileInChunks "src/main/java/com/vectordb/core/DistanceFn.java" "Add DistanceFn interface" 3
Commit-FileInChunks "src/main/java/com/vectordb/core/DistanceMetrics.java" "Implement distance metrics (cosine, euclidean, manhattan)" 6

Commit-FileInChunks "src/main/java/com/vectordb/index/BruteForceIndex.java" "Implement brute-force index" 5
Commit-FileInChunks "src/main/java/com/vectordb/index/KDTreeIndex.java" "Implement KD-Tree index" 8
Commit-FileInChunks "src/main/java/com/vectordb/index/HNSWIndex.java" "Implement HNSW index" 12

Commit-FileInChunks "src/main/java/com/vectordb/db/VectorDatabase.java" "Build VectorDatabase core logic" 8
Commit-FileInChunks "src/main/java/com/vectordb/db/DocumentDatabase.java" "Build DocumentDatabase for RAG storage" 6

Commit-FileInChunks "src/main/java/com/vectordb/text/TextChunker.java" "Add text chunking for RAG pipeline" 5
Commit-FileInChunks "src/main/java/com/vectordb/client/OllamaClient.java" "Add Ollama client integration" 6

Commit-FileInChunks "src/main/java/com/vectordb/util/Json.java" "Add lightweight JSON utility" 3
Commit-FileInChunks "src/main/java/com/vectordb/server/DemoData.java" "Add demo dataset for testing" 4
Commit-FileInChunks "src/main/java/com/vectordb/server/VectorDbServer.java" "Build HTTP server and API endpoints" 10

Commit-FileInChunks "src/main/resources/static/index.html" "Add web UI with live visualization" 7

git add README.md
git commit -m "Add README with project overview and usage" --quiet

$count = (git log --oneline | Measure-Object -Line).Lines
Write-Host "Total commits: $count"