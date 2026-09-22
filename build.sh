#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "Compiling VectorDB Engine..."
rm -rf out
mkdir -p out

SOURCES=$(find src/main/java -name "*.java")
javac -d out $SOURCES

# Bundle the static frontend onto the classpath alongside the .class files
cp -r src/main/resources/static out/static

echo "Build complete -> out/"
