#!/usr/bin/env bash
# ABOUTME: Tải trọng số ASR streaming từ HuggingFace về model_repository - chạy trên Thor, cần mạng ra huggingface.co
# ABOUTME: Chạy lại nhiều lần được - file đã có thì bỏ qua. TTS chưa nằm trong repo này, xem thor-voice-serving TODO

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$ROOT/model_repository"

dl() {  # dl <url> <đích>
  if [ -f "$2" ]; then echo "bỏ qua $2"; return; fi
  mkdir -p "$(dirname "$2")"
  echo "tải $2"
  curl -fL --progress-bar "$1" -o "$2"
}

# Biến thể chunk export sẵn: 16 (latency thấp nhất, mặc định), 32, 64.
CHUNK_VARIANT="${CHUNK_VARIANT:-16}"
STREAM=https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h/resolve/main
SFX="epoch-31-avg-11-chunk-${CHUNK_VARIANT}-left-128.fp16.onnx"
dl "$STREAM/encoder-$SFX" "$REPO/asr_streaming/1/encoder.onnx"
dl "$STREAM/decoder-$SFX" "$REPO/asr_streaming/1/decoder.onnx"
dl "$STREAM/joiner-$SFX"  "$REPO/asr_streaming/1/joiner.onnx"
dl "$STREAM/bpe.model"    "$REPO/asr_streaming/1/bpe.model"

echo "xong"
