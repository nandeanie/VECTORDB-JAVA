@echo off
setlocal

if not exist out (
  echo Compiling VectorDB Engine...
  mkdir out
  dir /s /b src\main\java\*.java > sources.txt
  javac -d out @sources.txt
  del sources.txt
  xcopy /E /I src\main\resources\static out\static
)

java -cp out com.vectordb.server.VectorDbServer
