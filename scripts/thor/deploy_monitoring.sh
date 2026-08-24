#!/usr/bin/env bash
# ABOUTME: Dựng Prometheus (9090) + Grafana (3000) trên Thor, bind loopback - vào qua SSH tunnel
# ABOUTME: Dùng: ./scripts/thor/deploy_monitoring.sh | ./scripts/thor/deploy_monitoring.sh down

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COMPOSE="$ROOT/docker/monitoring/docker-compose.yml"

if [ "${1:-up}" = "down" ]; then
  docker compose -f "$COMPOSE" down
  exit 0
fi

for port in 9090 3000; do
  if ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
    echo "dừng: cổng $port đang bị chiếm." >&2
    echo "      ss -ltnp 'sport = :$port'  để xem tiến trình nào." >&2
    exit 1
  fi
done

docker compose -f "$COMPOSE" up -d

echo
echo "Bind loopback - từ máy dev, mở tunnel trước khi xem:"
echo "  ssh -L 3300:localhost:3000 -L 9900:localhost:9090 <user>@<thor>"
echo "Rồi vào  http://localhost:3300  (Grafana)  và  http://localhost:9900/targets  (Prometheus)"
echo
echo "asr Triton phải chạy trước thì target mới UP: ./scripts/thor/deploy_asr.sh"
