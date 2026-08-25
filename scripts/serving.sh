#!/usr/bin/env bash
# ABOUTME: Vòng đời container cho toàn stack - build/up/down/logs/health, compose làm phần nặng
# ABOUTME: Chỉ cần bash + docker, KHÔNG cần Python, để up/health vẫn chạy khi Thor thiếu package

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# .env là nguồn sự thật duy nhất cho cổng. compose tự đọc nó, nhưng health cần
# các biến này trong shell nên phải export thêm ở đây.
if [ ! -f .env ]; then
  echo "error: .env not found. Run 'cp .env.example .env' first." >&2
  exit 1
fi
set -a; . ./.env; set +a

compose() { docker compose "$@"; }

WEIGHTS="model_repository/asr_streaming/1"

require_weights() {
  local missing=0
  for f in encoder.onnx decoder.onnx joiner.onnx bpe.model; do
    [ -f "$WEIGHTS/$f" ] || { echo "error: missing $WEIGHTS/$f" >&2; missing=1; }
  done
  [ "$missing" -eq 0 ] && return 0
  # Thor bị chặn không ra internet - bảo người ta chạy script tải là chỉ dẫn
  # sai, họ sẽ ngồi đợi timeout. Cách khắc phục thật là copy từ máy dev sang.
  cat >&2 <<'MSG'

Thor has no outbound internet access. Copy the weights from a dev machine:
  rsync -avP --include='*/' --include='*.onnx' --exclude='*' \
    model_repository/ <user>@thor:~/thor-voice-serving/model_repository/
MSG
  exit 1
}

cmd_build() { compose --profile asr build; }

cmd_up() {
  case "${1:-all}" in
    asr)        require_weights; compose --profile asr up -d --wait ;;
    monitoring) compose --profile monitoring up -d --wait ;;
    all)        require_weights; compose --profile asr --profile monitoring up -d --wait ;;
    *)          echo "error: unknown profile '$1' (expected: asr|monitoring)" >&2; exit 1 ;;
  esac
  echo
  cmd_health
}

cmd_down() { compose --profile asr --profile monitoring down "$@"; }

cmd_logs() { compose --profile asr --profile monitoring logs -f "$@"; }

# "container đang chạy" và "scrape thật sự thành công" là hai chuyện khác nhau.
# compose ps chỉ trả lời được vế đầu, nên health hỏi thẳng từng endpoint.
probe() {  # probe <nhãn> <url>
  if curl -sf --max-time 5 "$2" >/dev/null 2>&1; then
    printf '%-12s %-6s %s\n' "$1" "ok" "$2"
  else
    printf '%-12s %-6s %s\n' "$1" "FAIL" "$2"
    return 1
  fi
}

cmd_health() {
  local rc=0
  probe ASR "http://$BIND_ADDR:$ASR_HTTP_PORT/v2/health/ready" || rc=1
  probe Prometheus "http://$BIND_ADDR:$PROM_PORT/-/healthy" || rc=1
  probe Grafana "http://$BIND_ADDR:$GRAFANA_PORT/api/health" || rc=1

  # Vế thứ hai: prometheus có thật sự scrape được từng target không. Hỏi riêng
  # từng job chứ không tìm "có target nào up không" - ASR sống mà node-exporter
  # chết thì kiểu tìm chung sẽ báo xanh, đúng loại nói dối tệ nhất ở health check.
  for job in triton node; do
    local r
    if ! r="$(curl -sf --max-time 5 --get \
        "http://$BIND_ADDR:$PROM_PORT/api/v1/query" \
        --data-urlencode "query=up{job=\"$job\"}" 2>/dev/null)"; then
      printf '%-12s %-6s %s\n' "prom:$job" "?" "prometheus unreachable"
      rc=1
      continue
    fi
    case "$r" in
      *'"value":['*',"1"]'*)
        printf '%-12s %-6s %s\n' "prom:$job" "UP" "scrape ok" ;;
      *'"result":[]'*)
        printf '%-12s %-6s %s\n' "prom:$job" "EMPTY" "no samples yet for this job"
        rc=1 ;;
      *)
        printf '%-12s %-6s %s\n' "prom:$job" "DOWN" "scrape failing (check: ./scripts/serving.sh logs)"
        rc=1 ;;
    esac
  done
  return $rc
}

cmd_help() {
  cat <<'MSG'
Usage: ./scripts/serving.sh <command>

  build                 build the ASR image (host proxy forwarded as build-arg)
  up [asr|monitoring]   start the stack; no argument starts both
  down [-v]             stop the stack; -v also removes prometheus/grafana volumes
  logs [service]        follow logs; no argument follows all
  health                probe each endpoint and the prometheus scrape target
  help                  this message

Parity test is separate (needs Python): ./tests/test_parity.sh
MSG
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift

case "$cmd" in
  build|up|down|logs|health|help) "cmd_$cmd" "$@" ;;
  *) echo "error: unknown command '$cmd'" >&2; echo >&2; cmd_help >&2; exit 1 ;;
esac
