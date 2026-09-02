# ABOUTME: Metric quyết định của bench rerank - hàm thuần, không chạm mạng lẫn file
# ABOUTME: Tách khỏi driver để test chạy được khi Thor tắt và khi chưa có bộ eval

import math
import statistics

# Sàn nhiễu ĐO ĐƯỢC trên bf16: cùng input, cùng server, chạy lại vẫn lệch tới
# 0.063 trên điểm thô. Ngưỡng 0.1 để có biên an toàn. Hai doc chênh dưới mức này
# mà đảo chỗ là NHIỄU, không phải hồi quy.
# PHẢI ĐO LẠI khi đổi dtype/runtime - FP8 sẽ có sàn cao hơn hẳn.
TIE_EPS = 0.1


# --- chất lượng: cần nhãn relevance ----------------------------------------

def dcg_at_k(gains, k):
    """DCG với gain mũ: (2^rel - 1) / log2(i+2). Nhãn nhị phân thì thành 1/log2(i+2)."""
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_labels, k):
    """NDCG@k. ranked_labels = nhãn xếp theo THỨ TỰ MODEL TRẢ VỀ.

    Không có doc liên quan nào thì trả 0.0 chứ không chia cho 0: một query
    không có đáp án đúng là lỗi của bộ eval, và 0.0 sẽ kéo trung bình xuống
    đủ rõ để người ta đi tìm, còn ném lỗi thì giết cả lượt chạy.
    """
    ideal = dcg_at_k(sorted(ranked_labels, reverse=True), k)
    return dcg_at_k(ranked_labels, k) / ideal if ideal > 0 else 0.0


def mrr_at_k(ranked_labels, k):
    """1/thứ hạng của doc liên quan ĐẦU TIÊN trong top-k. Không có thì 0."""
    for i, g in enumerate(ranked_labels[:k]):
        if g > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_labels, k):
    """Bao nhiêu phần doc liên quan lọt vào top-k."""
    total = sum(1 for g in ranked_labels if g > 0)
    if total == 0:
        return 0.0
    return sum(1 for g in ranked_labels[:k] if g > 0) / total


# --- chất lượng: KHÔNG cần nhãn, so với baseline ---------------------------

def kendall_tau(scores_a, scores_b):
    """Tương quan thứ hạng giữa hai lần chấm cùng một tập doc. 1.0 = trùng khít.

    Dạng tau-a, O(n²) - n ở đây là số doc mỗi query, vài chục, không đáng tối ưu.
    """
    n = len(scores_a)
    if n < 2:
        return 1.0
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = (scores_a[i] - scores_a[j]) * (scores_b[i] - scores_b[j])
            if d > 0:
                con += 1
            elif d < 0:
                dis += 1
    total = n * (n - 1) / 2
    return (con - dis) / total if total else 1.0


def topk_overlap(scores_a, scores_b, k):
    """Tỷ lệ doc chung giữa top-k của hai lần chấm.

    Đây là thứ người dùng thật sự cảm nhận: họ chỉ nhìn top-k, không nhìn
    tương quan thứ hạng trên toàn danh sách.
    """
    order = lambda s: {i for i, _ in sorted(enumerate(s), key=lambda p: -p[1])[:k]}
    a, b = order(scores_a), order(scores_b)
    return len(a & b) / k if k else 1.0


def significant_swaps(scores_a, scores_b, tie_eps=TIE_EPS):
    """Số cặp doc đảo thứ tự mà KHÔNG giải thích được bằng nhiễu.

    Bỏ qua cặp mà cả hai lần chấm đều thấy chúng gần hoà (chênh < tie_eps):
    những cặp đó tự đảo chỗ ngay cả khi không đổi gì - đã đo được bằng tay.
    Đếm cả chúng thì bộ gác cổng báo động giả liên tục và sẽ bị tắt đi.
    """
    n = len(scores_a)
    swaps = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = scores_a[i] - scores_a[j], scores_b[i] - scores_b[j]
            if abs(da) < tie_eps and abs(db) < tie_eps:
                continue
            if da * db < 0:
                swaps += 1
    return swaps


def quality_summary(per_query, k):
    """Nhiều query -> trung bình các metric. Mỗi phần tử là dict của một query."""
    if not per_query:
        raise ValueError("bộ eval rỗng")
    keys = per_query[0].keys()
    return {f"{key}@{k}" if key in ("ndcg", "mrr", "recall") else key:
            statistics.mean([q[key] for q in per_query]) for key in keys}


# --- hiệu năng --------------------------------------------------------------

def pairs_per_second(n_requests, k, elapsed_s):
    """Thông lượng tính theo CẶP, không theo request: một request 60 doc nặng
    gấp bốn một request 15 doc, đếm request sẽ giấu mất chuyện đó."""
    return n_requests * k / elapsed_s if elapsed_s > 0 else 0.0


def batch_fill(avg_batch, max_batch_tokens, mean_pair_tokens):
    """avg_batch đang bằng bao nhiêu phần trần mà ngân sách token cho phép.

    TEI không có 'max_batch_size' cố định - nó gom theo ngân sách token. Trần
    thật là max_batch_tokens / độ dài một cặp. Gần 1.0 = batch đã đầy; gần 0 =
    đang chạy dưới công suất và đó là chỗ để lấy throughput.
    """
    if mean_pair_tokens <= 0:
        return 0.0
    ceiling = max_batch_tokens / mean_pair_tokens
    return avg_batch / ceiling if ceiling > 0 else 0.0


def concurrency(qps, latency_s):
    """Số request nằm trong hệ thống, suy từ định luật Little: L = λ × W.

    Bench open-loop không đặt CCU mà để nó tự hình thành, nên đây là cách đọc
    ra 'CCU tại mức tải này' mà không phải đếm tay.
    """
    return qps * latency_s


def max_qps_within_slo(agg, slo_s, key="p99_total_s"):
    """Mức QPS cao nhất còn giữ được p99 dưới SLO, quét TỪ DƯỚI LÊN.

    Dừng ở mức đầu tiên vỡ chứ không lấy max toàn bảng: sau điểm gãy, một mức
    cao hơn tình cờ đo đẹp (hàng xóm vừa rảnh) sẽ bị đọc thành sức chứa.
    """
    ok = 0.0
    for q in sorted(agg):
        if agg[q][key] > slo_s:
            break
        ok = q
    return ok
