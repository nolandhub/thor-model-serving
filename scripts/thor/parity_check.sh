#!/usr/bin/env bash
# ABOUTME: Gửi sample_vi.wav qua ASR đang chạy trên Thor, so transcript với golden fixture dump từ máy dev
# ABOUTME: Đây là bằng chứng "chạy đúng như máy dev" - không phải suy đoán từ log

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GRPC_PORT="${ASR_GRPC_PORT:-9001}"

EXPECTED="ANH CÓ THÍCH TÔI GIỮ TRƯỚC QUYỂN SÁCH NÀY CHO ANH KHÔNG"

echo "gửi tests/assets/sample_vi.wav qua localhost:$GRPC_PORT..."
# Dòng cuối client in là \r\x1b[K{transcript} - xoá mã ANSI trước khi so, không
# chỉ \r, nếu không GOT sẽ mang cả \x1b[K lẫn vào và không bao giờ khớp EXPECTED.
GOT="$(python3 "$ROOT/client/asr_streaming_client.py" \
  --url "localhost:$GRPC_PORT" --fast \
  "$ROOT/tests/assets/sample_vi.wav" 2>&1 | tail -1 | sed -E 's/\x1b\[[0-9;]*[a-zA-Z]//g; s/\r//g')"

echo "kỳ vọng (golden, dump từ ORT trên máy dev):"
echo "  $EXPECTED"
echo "nhận được (Thor):"
echo "  $GOT"

if [ "$GOT" = "$EXPECTED" ]; then
  echo "KHỚP"
  exit 0
else
  echo "LỆCH - xem docker logs thor-asr-triton, kiểm CHUNK_VARIANT và version model" >&2
  exit 1
fi
