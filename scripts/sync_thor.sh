#!/usr/bin/env bash
# Đồng bộ mã nguồn + trọng số từ máy dev sang Thor.
# Thor không ra được internet: không git pull, không docker pull được ở đó,
# nên rsync một chiều dev -> Thor là đường duy nhất để code lên máy.
#
#   ./scripts/sync_thor.sh            # xem trước (dry-run), không đổi gì
#   ./scripts/sync_thor.sh --go       # đồng bộ thật
#   ./scripts/sync_thor.sh --go --delete   # xoá luôn file thừa bên Thor (mirror)
#
# Đổi đích bằng biến môi trường: THOR_HOST=thor THOR_DIR=/đường/dẫn/khác
set -euo pipefail

HOST=${THOR_HOST:-thor}
# Đường dẫn tuyệt đối, KHÔNG phải ~/thor-voice-serving như README nói: bản làm
# việc thật trên Thor nằm ở /u01 (mount riêng, không nằm trong home), và đây
# đúng là thư mục container thor-asr-triton bind-mount vào /models. Sync nhầm
# sang home thì file lên máy nhưng Triton không thấy gì hết.
DEST=${THOR_DIR:-/u01/nhandlt2/thor-model-serving}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Mặc định dry-run: gõ nhầm đường dẫn đích mà có --delete thì mất dữ liệu thật,
# nên bắt buộc phải nói --go mới ghi.
mode=(--dry-run)
extra=()
for arg in "$@"; do
  case "$arg" in
    --go)     mode=() ;;
    --delete) extra+=(--delete) ;;
    *) echo "tham số lạ: $arg" >&2; exit 2 ;;
  esac
done

# .env bị loại: cấu hình cục bộ của Thor (cổng, BIND_ADDR), ghi đè lên là hỏng
# máy đang chạy. .git 88M mà Thor không cần lịch sử. bench/results là số đo của
# chính Thor, kéo từ dev sang là ghi đè kết quả thật bằng số máy khác.
# --chmod là bắt buộc, không phải cho đẹp: máy dev có umask 0007 nên file là
# 0660/0770, mà rsync -a bê nguyên mode sang Thor. Prometheus trong container
# chạy bằng user `nobody`, Grafana bằng uid 472 - không phải chủ file, nên mất
# quyền đọc config bind-mount và crash-loop "permission denied". Ép a+r ở đây
# thì umask máy dev không đi theo sang nữa.
rsync -avhP --chmod=Da+rx,Fa+r --human-readable "${mode[@]}" "${extra[@]}" \
  --exclude='.git/' \
  --exclude='.env' \
  --exclude='__pycache__/' \
  --exclude='*.py[cod]' \
  --exclude='.venv/' \
  --exclude='.pytest_cache/' \
  --exclude='bench/results/' \
  --exclude='bench/result/' \
  --exclude='.gitnexus/' \
  --exclude='.claude/' \
  --exclude='CLAUDE.local.md' \
  "$SRC/" "$HOST:$DEST/"

if [ ${#mode[@]} -ne 0 ]; then
  echo
  echo "^ mới chỉ là dry-run. Chạy lại với --go để đồng bộ thật."
fi
