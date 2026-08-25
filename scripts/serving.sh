#!/usr/bin/env bash
# ABOUTME: Vòng đời container cho toàn stack - build/up/down/logs/health, compose làm phần nặng
# ABOUTME: Chỉ cần bash + docker, KHÔNG cần Python, để up/health vẫn chạy khi Thor thiếu package

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# .env là nguồn sự thật duy nhất cho cổng. compose tự đọc nó, nhưng health cần
# các biến này trong shell nên phải export thêm ở đây.
if [ ! -f .env ]; then
  echo "dừng: chưa có .env - chạy 'cp .env.example .env' rồi sửa nếu cần." >&2
  exit 1
fi
set -a; . ./.env; set +a

compose() { docker compose "$@"; }

WEIGHTS="model_repository/asr_streaming/1"

require_weights() {
  local missing=0
  for f in encoder.onnx decoder.onnx joiner.onnx bpe.model; do
    [ -f "$WEIGHTS/$f" ] || { echo "thiếu $WEIGHTS/$f" >&2; missing=1; }
  done
  [ "$missing" -eq 0 ] && return 0
  # Thor bị chặn không ra internet - bảo người ta chạy script tải là chỉ dẫn
  # sai, họ sẽ ngồi đợi timeout. Cách khắc phục thật là copy từ máy dev sang.
  cat >&2 <<'MSG'

Thor không ra được internet nên không tải trực tiếp được. Từ máy dev:
  rsync -av model_repository/ thor:~/thor-voice-serving/model_repository/
MSG
  exit 1
}

cmd_build() { compose --profile asr build; }

cmd_up() {
  case "${1:-all}" in
    asr)        require_weights; compose --profile asr up -d --wait ;;
    monitoring) compose --profile monitoring up -d --wait ;;
    all)        require_weights; compose --profile asr --profile monitoring up -d --wait ;;
    *)          echo "up: profile không hợp lệ '$1' (asr|monitoring)" >&2; exit 1 ;;
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
    printf '%-12s %-6s %s\n' "$1" "HỎNG" "$2"
    return 1
  fi
}

cmd_health() {
  local rc=0
  probe ASR "http://$BIND_ADDR:$ASR_HTTP_PORT/v2/health/ready" || rc=1
  probe Prometheus "http://$BIND_ADDR:$PROM_PORT/-/healthy" || rc=1
  probe Grafana "http://$BIND_ADDR:$GRAFANA_PORT/api/health" || rc=1

  # Vế thứ hai: prometheus có thật sự scrape được asr:8002 không. Target DOWN
  # thì dashboard trống trơn mà chẳng có gì báo lỗi.
  local targets
  if targets="$(curl -sf --max-time 5 "http://$BIND_ADDR:$PROM_PORT/api/v1/targets" 2>/dev/null)"; then
    if printf '%s' "$targets" | grep -q '"health":"up"'; then
      printf '%-12s %-6s %s\n' "prom→asr" "UP" "asr:8002"
    else
      printf '%-12s %-6s %s\n' "prom→asr" "DOWN" "asr:8002 - xem ./scripts/serving.sh logs asr"
      rc=1
    fi
  fi
  return $rc
}

cmd_help() {
  cat <<'MSG'
./scripts/serving.sh <lệnh>

  build                 build image ASR (forward proxy của host qua build-arg)
  up [asr|monitoring]   dựng stack, không tham số = cả hai
  down [-v]             hạ stack, -v xoá luôn volume prometheus/grafana
  logs [service]        follow log, không tham số = tất cả
  health                kiểm từng endpoint + prometheus target có UP không
  help                  bản này


MSG
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift

case "$cmd" in
  build|up|down|logs|health|help) "cmd_$cmd" "$@" ;;
  *) echo "lệnh không rõ: $cmd" >&2; echo >&2; cmd_help >&2; exit 1 ;;
esac
