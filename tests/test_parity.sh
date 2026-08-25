#!/usr/bin/env bash
# ABOUTME: Gửi sample_vi.wav qua ASR đang chạy trên Thor, so transcript với golden fixture dump từ máy dev
# ABOUTME: Đây là bằng chứng "chạy đúng như máy dev" - không phải suy đoán từ log. Cần tritonclient[grpc]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Cùng .env mà compose và serving.sh đọc - cổng chỉ khai một nơi.
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
GRPC_PORT="${ASR_GRPC_PORT:-4001}"
ADDR="${BIND_ADDR:-127.0.0.1}"

EXPECTED="ANH CÓ THÍCH TÔI GIỮ TRƯỚC QUYỂN SÁCH NÀY CHO ANH KHÔNG"

python3 -c 'import tritonclient.grpc' 2>/dev/null || {
  echo "error: tritonclient missing. Install it first:" >&2
  echo "  pip install --user \"tritonclient[grpc]\" numpy soundfile scipy" >&2
  exit 1
}

echo "streaming tests/assets/sample_vi.wav to $ADDR:$GRPC_PORT ..."

# Chạy tách khỏi pipeline. Gộp `python3 ... 2>&1 | tail -1` vào một lệnh gán là
# tự bịt miệng mình: pipefail cho pipeline trả mã lỗi, set -e giết script ngay
# tại dòng gán, mà stderr thì đã bị nuốt vào pipe nên không in ra gì hết.
set +e
RAW="$(python3 "$ROOT/client/asr_streaming_client.py" \
  --url "$ADDR:$GRPC_PORT" --fast \
  "$ROOT/tests/assets/sample_vi.wav" 2>&1)"
RC=$?
set -e

if [ "$RC" -ne 0 ]; then
  echo "error: client exited $RC" >&2
  printf '%s\n' "$RAW" >&2
  exit 1
fi

# Dòng cuối client in là \r\x1b[K{transcript} - xoá mã ANSI trước khi so, không
# chỉ \r, nếu không GOT sẽ mang cả \x1b[K lẫn vào và không bao giờ khớp EXPECTED.
GOT="$(printf '%s\n' "$RAW" | tail -1 | sed -E 's/\x1b\[[0-9;]*[a-zA-Z]//g; s/\r//g')"

echo "expected (golden, dumped from ORT on a dev machine):"
echo "  $EXPECTED"
echo "actual   (Thor):"
echo "  $GOT"

if [ "$GOT" = "$EXPECTED" ]; then
  echo "PASS - transcript matches golden"
  exit 0
else
  echo "FAIL - transcript differs. Check: ./scripts/serving.sh logs asr, CHUNK_VARIANT, model version" >&2
  exit 1
fi
