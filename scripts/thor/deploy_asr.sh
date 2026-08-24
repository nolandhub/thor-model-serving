#!/usr/bin/env bash
# ABOUTME: Build và chạy ASR Triton trên Thor - cổng 9000-9002, KHÔNG dùng --net host
# ABOUTME: Thor là máy dùng chung: voice-agent-asr-triton-1 (đồng nghiệp) đã chiếm 8000/8001, script này né hẳn dải đó

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

HTTP_PORT="${ASR_HTTP_PORT:-9000}"
GRPC_PORT="${ASR_GRPC_PORT:-9001}"
METRICS_PORT="${ASR_METRICS_PORT:-9002}"
NAME="${ASR_CONTAINER_NAME:-thor-asr-triton}"

# Xoá container cũ CỦA CHÍNH MÌNH trước khi kiểm cổng - không thì lần chạy thứ
# hai luôn tự báo "cổng bị chiếm" vì thấy đúng container mình vừa tạo lần trước,
# rồi thoát trước khi kịp xoá nó. Kiểm cổng phải diễn ra SAU bước dọn này.
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "xoá container cũ tên $NAME..."
  docker rm -f "$NAME" >/dev/null
fi

for p in "$HTTP_PORT" "$GRPC_PORT" "$METRICS_PORT"; do
  if ss -ltn "sport = :$p" 2>/dev/null | grep -q LISTEN; then
    echo "dừng: cổng $p đang bị chiếm (chạy 'ss -ltnp sport = :$p' để xem ai)." >&2
    echo "      Container khác trên máy này (vd voice-agent-asr-triton-1) có thể đã dùng dải 8000-8001." >&2
    exit 1
  fi
done

WEIGHTS="$ROOT/model_repository/asr_streaming/1"
for f in encoder.onnx decoder.onnx joiner.onnx bpe.model; do
  if [ ! -f "$WEIGHTS/$f" ]; then
    echo "dừng: thiếu $WEIGHTS/$f - chạy ./scripts/fetch_models.sh trước." >&2
    exit 1
  fi
done

echo "build image..."
docker build -t thor-voice-serving-asr -f "$ROOT/docker/Dockerfile.thor" "$ROOT/docker"

echo "chạy tritonserver (http=$HTTP_PORT grpc=$GRPC_PORT metrics=$METRICS_PORT)..."
docker run -d --gpus all \
  --name "$NAME" \
  --restart unless-stopped \
  -p "$HTTP_PORT:8000" \
  -p "$GRPC_PORT:8001" \
  -p "$METRICS_PORT:8002" \
  -v "$ROOT/model_repository:/models" \
  -v "$ROOT/serving:/opt/serving/serving:ro" \
  thor-voice-serving-asr \
  tritonserver --model-repository=/models \
    --metrics-config summary_latencies=true \
    --metrics-config 'summary_quantiles=0.5:0.05,0.95:0.01,0.99:0.001'

echo
echo "đang lên, đợi health check..."
for _ in $(seq 1 30); do
  if curl -sf "http://localhost:$HTTP_PORT/v2/health/ready" >/dev/null 2>&1; then
    echo "READY  http://localhost:$HTTP_PORT/v2/health/ready"
    echo "gRPC   localhost:$GRPC_PORT"
    echo "metrics http://localhost:$METRICS_PORT/metrics"
    exit 0
  fi
  sleep 2
done

echo "chưa ready sau 60s - xem log: docker logs $NAME" >&2
exit 1
