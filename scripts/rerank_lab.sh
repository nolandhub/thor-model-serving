#!/usr/bin/env bash
# ABOUTME: Dựng/tắt/soi container reranker thí nghiệm - mỗi lần đổi cờ TEI là một lần dựng lại
# ABOUTME: KHÔNG đụng vào search-engine-reranker-1 (prod của team khác, cổng 9002)

set -euo pipefail

NAME=${RERANK_LAB_NAME:-rerank-lab}
PORT=${RERANK_LAB_PORT:-9012}
MODEL=${RERANK_LAB_MODEL:-/u01/nhandlt2/reranker}
IMAGE=${RERANK_LAB_IMAGE:-ddosify/text-embeddings-inference:blackwell-1.8.3-baai-bge-reranker-v2-m3}

# Cấu hình prod, để `up` không tham số cho ra bản sao trung thực làm mốc đối chứng.
DEFAULT_ARGS=(--max-client-batch-size 128)

usage() {
    cat <<'EOF'
rerank_lab.sh - nghịch container reranker mà không đụng prod

    up [cờ TEI...]   dựng lại container với cờ mới (không tham số = giống hệt prod)
    down             tắt, giữ container
    rm               xoá hẳn
    logs             theo dõi log (mỗi request in total/queue/inference/tokenize)
    info             /info - cấu hình ĐANG chạy, hỏi từ server chứ không đoán
    metrics          /metrics thô
    status           đang chạy không, ăn bao nhiêu RAM

Ví dụ:
    ./scripts/rerank_lab.sh up
    ./scripts/rerank_lab.sh up --max-client-batch-size 128 --max-input-length 512 --auto-truncate
    ./scripts/rerank_lab.sh up --max-client-batch-size 128 --dtype float16

TEI đọc cờ ĐÚNG MỘT LẦN lúc khởi động. Sửa mà không dựng lại thì vẫn là cấu hình
cũ - luôn đối chiếu bằng `info` trước khi tin một con số nào.
EOF
}

require_model() {
    [ -d "$MODEL" ] || { echo "không thấy model: $MODEL" >&2; exit 1; }
}

cmd_up() {
    require_model
    local args=("$@")
    [ ${#args[@]} -eq 0 ] && args=("${DEFAULT_ARGS[@]}")

    docker rm -f "$NAME" >/dev/null 2>&1 || true

    # 127.0.0.1 chứ không 0.0.0.0: TEI không có auth, prod phơi ra LAN là việc
    # của prod. restart=no: container thí nghiệm không được tự sống lại rồi âm
    # thầm giữ 5GB trên máy dùng chung. Xoá sạch proxy: Docker tiêm proxy công
    # ty vào mọi container lúc tạo, urllib/grpc đều đọc nó (bẫy #3).
    docker run -d \
        --name "$NAME" \
        --runtime nvidia \
        --shm-size 1g \
        --restart no \
        -e MODEL_ID=/data/model \
        -e PORT=8000 \
        -e http_proxy= -e https_proxy= -e HTTP_PROXY= -e HTTPS_PROXY= \
        -e no_proxy= -e NO_PROXY= -e all_proxy= -e ALL_PROXY= \
        -v "$MODEL:/data/model:ro" \
        -p "127.0.0.1:$PORT:8000" \
        "$IMAGE" \
        "${args[@]}" >/dev/null

    printf 'đang nạp model'
    for _ in $(seq 1 40); do
        # Cờ sai thì TEI thoát ngay. Không chốt chặn ở đây thì phải đợi hết 120s
        # mới biết, mà lỗi đã nằm sẵn trong log từ giây đầu.
        if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
            echo " - CHẾT NGAY KHI KHỞI ĐỘNG:"
            docker logs "$NAME" 2>&1 | tail -12
            return 1
        fi
        if [ "$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' --max-time 3 \
                "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
            echo " - sẵn sàng ở 127.0.0.1:$PORT"
            echo
            echo "cấu hình ĐANG chạy (hỏi từ server):"
            cmd_info
            return 0
        fi
        printf '.'
        sleep 3
    done
    echo " - QUÁ HẠN. Log:"
    docker logs --tail 40 "$NAME"
    return 1
}

cmd_info() {
    curl -s --noproxy '*' "http://127.0.0.1:$PORT/info" | python3 -m json.tool
}

case "${1:-}" in
    up)      shift; cmd_up "$@" ;;
    down)    docker stop "$NAME" >/dev/null && echo "đã tắt $NAME (start lại: docker start $NAME)" ;;
    rm)      docker rm -f "$NAME" >/dev/null && echo "đã xoá $NAME (model ở $MODEL vẫn còn)" ;;
    logs)    docker logs -f --tail 50 "$NAME" ;;
    info)    cmd_info ;;
    metrics) curl -s --noproxy '*' "http://127.0.0.1:$PORT/metrics" ;;
    status)
        docker ps -a --filter "name=$NAME" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
        docker stats --no-stream --format '{{.Name}}: {{.MemUsage}}' "$NAME" 2>/dev/null || true
        ;;
    *)       usage; exit 1 ;;
esac
