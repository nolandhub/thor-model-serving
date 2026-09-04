#!/usr/bin/env bash
# ABOUTME: Dựng/tắt/soi container reranker thí nghiệm - mỗi lần đổi cờ TEI là một lần dựng lại
# ABOUTME: KHÔNG đụng vào search-engine-reranker-1 (prod của team khác, cổng 9002)

set -euo pipefail

NAME=${RERANK_LAB_NAME:-rerank-lab}
PORT=${RERANK_LAB_PORT:-9012}
MODEL=${RERANK_LAB_MODEL:-/u01/nhandlt2/reranker}
IMAGE=${RERANK_LAB_IMAGE:-ddosify/text-embeddings-inference:blackwell-1.8.3-baai-bge-reranker-v2-m3}

# Cấu hình prod. Nền của MỌI lần dựng, không chỉ lần `up` không tham số: cờ bạn
# truyền được trộn lên trên nền này, nên một thí nghiệm chỉ khác prod đúng chỗ
# bạn cố ý đổi. Viết thành từng cặp (cờ, giá trị) - cmd_up duyệt theo cặp.
DEFAULT_ARGS=(--max-client-batch-size 128)

usage() {
    cat <<'EOF'
rerank_lab.sh - experiment on a reranker container without touching prod

    up [TEI flags...]  recreate the container with new flags (no args = prod replica)
    down               stop, keep the container
    rm                 remove entirely
    logs               follow logs (each request prints total/queue/inference/tokenize)
    info               /info - the LIVE config, asked from the server, not guessed
    metrics            raw /metrics
    status             running or not, and how much RAM it holds

Examples:
    ./scripts/rerank_lab.sh up
    ./scripts/rerank_lab.sh up --max-batch-tokens 32768
    ./scripts/rerank_lab.sh up --max-input-length 512 --auto-truncate
    ./scripts/rerank_lab.sh up --dtype float16

Your flags are MERGED onto the prod baseline (--max-client-batch-size 128), not
substituted for it - so `up --max-batch-tokens N` differs from prod by exactly one
variable, not two. To change a baseline flag, pass it yourself; yours wins.

TEI reads its flags EXACTLY ONCE at startup. Editing without recreating leaves the
old config running - always confirm with `info` before trusting any number.
EOF
}

require_model() {
    [ -d "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
}

cmd_up() {
    require_model

    # Trộn DEFAULT_ARGS với cờ người dùng, cờ người dùng thắng.
    #
    # Trước đây là THAY THẾ: truyền một cờ bất kỳ là mất sạch mặc định, nên
    # `up --max-batch-tokens 32768` âm thầm đổi HAI biến - cờ muốn thử, VÀ
    # max-client-batch-size tụt 128 -> 32 (mặc định TEI) khiến mọi request
    # top-k > 32 fail 413. Thí nghiệm phải khác đối chứng đúng một chỗ.
    #
    # Phải khử trùng lặp chứ không chồng cờ lên nhau: clap của TEI từ chối cờ
    # lặp - "cannot be used multiple times" - đã đo, container chết ngay khi
    # khởi động. Bắt cả dạng --cờ=giá-trị vì nó cũng tính là lặp.
    #
    # Chỉ DEFAULT_ARGS mới buộc phải là cặp (cờ, giá trị); cờ người dùng nối
    # nguyên xi nên cờ boolean như --auto-truncate vẫn đi qua bình thường.
    local args=() i flag a found
    for ((i = 0; i < ${#DEFAULT_ARGS[@]}; i += 2)); do
        flag=${DEFAULT_ARGS[i]}
        found=0
        for a in "$@"; do
            if [ "$a" = "$flag" ] || [ "${a#"$flag"=}" != "$a" ]; then
                found=1
                break
            fi
        done
        if [ "$found" -eq 0 ]; then
            args+=("$flag" "${DEFAULT_ARGS[i+1]}")
        fi
    done
    args+=("$@")

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

    printf 'loading model'
    for _ in $(seq 1 40); do
        # Cờ sai thì TEI thoát ngay. Không chốt chặn ở đây thì phải đợi hết 120s
        # mới biết, mà lỗi đã nằm sẵn trong log từ giây đầu.
        if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
            echo " - DIED ON STARTUP:"
            docker logs "$NAME" 2>&1 | tail -12
            return 1
        fi
        if [ "$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' --max-time 3 \
                "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
            echo " - ready on 127.0.0.1:$PORT"
            echo
            echo "LIVE config (asked from the server):"
            cmd_info
            return 0
        fi
        printf '.'
        sleep 3
    done
    echo " - TIMED OUT. Logs:"
    docker logs --tail 40 "$NAME"
    return 1
}

cmd_info() {
    curl -s --noproxy '*' "http://127.0.0.1:$PORT/info" | python3 -m json.tool
}

case "${1:-}" in
    up)      shift; cmd_up "$@" ;;
    down)    docker stop "$NAME" >/dev/null && echo "stopped $NAME (restart with: docker start $NAME)" ;;
    rm)      docker rm -f "$NAME" >/dev/null && echo "removed $NAME (model at $MODEL is untouched)" ;;
    logs)    docker logs -f --tail 50 "$NAME" ;;
    info)    cmd_info ;;
    metrics) curl -s --noproxy '*' "http://127.0.0.1:$PORT/metrics" ;;
    status)
        docker ps -a --filter "name=$NAME" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
        docker stats --no-stream --format '{{.Name}}: {{.MemUsage}}' "$NAME" 2>/dev/null || true
        ;;
    *)       usage; exit 1 ;;
esac
