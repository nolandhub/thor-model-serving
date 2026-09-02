# ABOUTME: Đọc counter từ /metrics của TEI - nguồn số phía server cho bench rerank
# ABOUTME: Parse dùng lại của triton_metrics; phần tính là hàm thuần, test được khi Thor tắt

import urllib.request

from bench.triton_metrics import parse_exposition

# TEI xuất giây, không phải micro giây như Triton. Đổi sang ms lúc báo cáo.
REQUESTS = "te_request_count"                      # số lời gọi /rerank
PAIRS = "te_predict_count"                         # số cặp (query, doc) đã chấm
BATCH_SUM = "te_batch_next_size_sum"               # tổng số cặp trong mọi batch
BATCH_N = "te_batch_next_size_count"               # số lần chạy model
QUEUE_S = "te_request_queue_duration_sum"
INFER_S = "te_request_inference_duration_sum"
TOKENIZE_S = "te_request_tokenization_duration_sum"
TOTAL_S = "te_request_duration_sum"

NEEDED = (REQUESTS, PAIRS, BATCH_SUM, BATCH_N, QUEUE_S, INFER_S, TOKENIZE_S, TOTAL_S)


def counters(samples):
    """Gộp counter TEI, cộng dồn qua nhãn.

    te_request_count mang nhãn method="batch"; cộng qua nhãn để nếu TEI thêm
    method khác thì vẫn ra tổng số request chứ không im lặng bỏ sót một nhánh.

    Thiếu counter thì báo lỗi chứ không trả {}: gõ nhầm cổng sang một service
    Prometheus khác cũng cho ra text hợp lệ, và bench sẽ báo "xong" trong khi
    mọi Δ đều bằng 0.
    """
    out = {}
    for name, _labels, value in samples:
        if name in NEEDED:
            out[name] = out.get(name, 0.0) + value
    missing = [n for n in NEEDED if n not in out]
    if missing:
        raise ValueError(
            f"/metrics thiếu counter {missing} - kiểm cổng có đúng là TEI không"
        )
    return out


def snapshot_http(url):
    """Chụp counter qua HTTP. Không đi qua proxy - xem bẫy #3 trong bench/README."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as resp:
        return counters(parse_exposition(resp.read().decode()))


def send_lags(pick_times, deadlines):
    """Độ lệch giữa lúc request ĐÁNG LẼ phát đi và lúc worker thật sự cầm tới.

    Đây là drift của bench open-loop. Khác drift ASR ở chỗ đồng hồ không do
    audio áp đặt mà do chính ta chọn (--qps), nhưng ý nghĩa y hệt: đứng yên
    quanh 0 = server theo kịp nhịp; TĂNG DẦN = mọi worker đều bận, việc dồn lại
    và không bao giờ đuổi kịp - đó là sập, dù p99 của từng request vẫn đẹp.
    """
    return [p - d for p, d in zip(pick_times, deadlines)]


def server_side(delta):
    """Δ counter -> các số phía server, quy về mỗi-request.

    Chia 0 trả 0 chứ không ném: run rỗng là chuyện caller phải phát hiện (nó
    biết đã gửi bao nhiêu), không phải việc của hàm thuần này.
    """
    reqs = delta[REQUESTS]
    batches = delta[BATCH_N]
    return {
        # Số cặp trung bình mỗi lần chạy model. Đây là chỉ số quyết định: gần
        # bằng 1 nghĩa là bộ gom batch không gom được gì và GPU đang chạy
        # không tải, y hệt kết luận avg_batch của bench ASR.
        "avg_batch": delta[BATCH_SUM] / batches if batches else 0.0,
        "pairs_per_request": delta[PAIRS] / reqs if reqs else 0.0,
        "queue_ms_per_req": delta[QUEUE_S] / reqs * 1e3 if reqs else 0.0,
        "infer_ms_per_req": delta[INFER_S] / reqs * 1e3 if reqs else 0.0,
        "tokenize_ms_per_req": delta[TOKENIZE_S] / reqs * 1e3 if reqs else 0.0,
    }
