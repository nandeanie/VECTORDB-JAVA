#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d out ]; then
  bash ./build.sh
fi

java -cp out com.vectordb.server.VectorDbServer
