# ABOUTME: Các metric quyết định của bench - hàm thuần, không chạm mạng lẫn file
# ABOUTME: Tách khỏi driver để test chạy được khi Thor tắt và khi thiếu tritonclient

import math
import statistics


def pct(values, p):
    """Percentile theo nearest-rank - trả về một mẫu CÓ THẬT trong dãy.

    Không nội suy: p99 của latency phải là một chunk thật sự đã chậm ngần ấy,
    không phải trung bình giữa hai chunk mà không lần gửi nào từng gặp. Nội suy
    làm số đẹp lên ở đúng chỗ đang cần nó xấu.
    """
    if not values:
        raise ValueError("dãy rỗng - không có percentile")
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def drifts(recv_times, t_start, chunk_s):
    """Độ lệch so với thời gian thực của từng chunk.

    Chunk i chứa audio [i, i+1)*chunk_s, nên hệ thống đúng nhịp phải trả kết quả
    ngay khi đoạn audio đó vừa hết: t_start + (i+1)*chunk_s. Lấy mốc đó làm gốc.

    Giá trị đứng yên quanh 0 = bám realtime. Giá trị TĂNG DẦN = server chậm hơn
    tốc độ audio vào, hàng đợi dồn vô hạn - đó là sập, dù p99 có thể vẫn đẹp vì
    p99 chỉ nhìn từng chunk rời chứ không nhìn xu hướng tích luỹ.
    """
    return [r - (t_start + (i + 1) * chunk_s) for i, r in enumerate(recv_times)]


def avg_batch(request_count, exec_count):
    """Số request trung bình Triton gộp được vào một lần chạy model.

    Đây là chỉ số quyết định của cả bài bench. Toàn bộ lý lẽ ủng hộ BLS là tách
    encoder ra để dynamic batcher gom request từ nhiều stream. Nếu số này vẫn
    xấp xỉ 1.0 thì không gom được gì và BLS chỉ còn lại phần chi phí.
    """
    if exec_count <= 0:
        raise ValueError(f"exec_count={exec_count} - model chưa chạy lần nào, không tính được")
    return request_count / exec_count


def bls_tax(compute_input_us, compute_infer_us, compute_output_us):
    """Chi phí copy tensor qua ranh giới model, tính theo bội của compute thật.

    Cache encoder phải serialize ra Triton tensor rồi nhận lại trên MỖI chunk
    của MỖI stream. Triton tính phần đó vào compute_input/compute_output, tách
    khỏi compute_infer - nên tỷ số này chính là đồng hồ đo thuế BLS.
    """
    if compute_infer_us <= 0:
        raise ValueError(f"compute_infer_us={compute_infer_us} - không có compute để so")
    return (compute_input_us + compute_output_us) / compute_infer_us


def diff_counters(before, after):
    """Δ giữa hai snapshot counter Triton.

    Counter là cộng dồn, nên Δ âm chỉ có một nghĩa: server đã restart giữa hai
    snapshot và counter tụt về 0. Phải hỏng to ở đây - trả số âm đi tiếp thì nó
    lặng lẽ chảy vào avg_batch rồi thành một kết luận sai mà không ai truy ra.
    """
    out = {}
    for name, v0 in before.items():
        if name not in after:
            raise ValueError(f"metric {name!r} có ở snapshot đầu nhưng mất ở snapshot sau")
        delta = after[name] - v0
        if delta < 0:
            raise ValueError(
                f"metric {name!r} giảm ({v0} -> {after[name]}) - counter đã reset, "
                "server restart giữa run; bỏ run này"
            )
        out[name] = delta
    return out


def max_ccu_within_budget(p99_by_ccu, budget_s):
    """CCU cao nhất còn đạt ngưỡng latency, tính từ dưới lên.

    Dừng ngay ở mức đầu tiên vỡ chứ không quét tiếp tìm mức cao hơn còn đẹp:
    một mức đã vỡ mà mức trên lại đẹp là nhiễu (throttle, run bẩn), không phải
    sức chứa. Báo cáo nó thành sức chứa là loại nói dối tệ nhất ở bench.
    """
    ok = 0
    for ccu in sorted(p99_by_ccu):
        if p99_by_ccu[ccu] > budget_s:
            break
        ok = ccu
    return ok


# Tên metric Triton mà driver phải chụp. Numerator của avg_batch là
# nv_inference_count chứ KHÔNG phải nv_inference_request_success: cái đầu đã
# nhân batch size, cái sau đếm request nên tỷ số luôn ra ~1.0 và bench sẽ kết
# luận sai rằng dynamic batcher không gom được gì.
COUNT = "nv_inference_count"
EXEC = "nv_inference_exec_count"
C_IN = "nv_inference_compute_input_duration_us"
C_INFER = "nv_inference_compute_infer_duration_us"
C_OUT = "nv_inference_compute_output_duration_us"
QUEUE = "nv_inference_queue_duration_us"


# Chỉ sáu cái này là counter đơn điệu. Triton xuất cạnh chúng cả summary
# (nv_inference_request_summary_us - cửa sổ trượt, tụt xuống là bình thường) và
# gauge (pending_request_count), nên KHÔNG được diff cả snapshot: chốt chặn
# "counter reset" sẽ bắn nhầm vào summary và giết cả run sau khi đã đo xong.
NEEDED = (COUNT, EXEC, C_IN, C_INFER, C_OUT, QUEUE)


def only_counters(snapshot):
    """Lọc snapshot còn đúng các counter đơn điệu mà bench dùng."""
    missing = [name for name in NEEDED if name not in snapshot]
    if missing:
        raise ValueError(f"snapshot thiếu counter {missing} - kiểm tên model và version Triton")
    return {name: snapshot[name] for name in NEEDED}


def run_summary(record):
    """Một run -> các metric quyết định. Ném lỗi nếu counter đã reset."""
    lat = [
        r - s
        for stream in record["streams"]
        for s, r in zip(stream["send"], stream["recv"])
    ]
    # Drift lấy stream TỆ NHẤT, không phải trung bình: một stream tụt nhịp là
    # một cuộc gọi hỏng, trung bình với các stream khoẻ sẽ giấu mất nó.
    worst_drift = max(
        drifts(stream["recv"], stream["t_start"], record["chunk_s"])[-1]
        for stream in record["streams"]
    )
    d = diff_counters(
        only_counters(record["counters_before"]), only_counters(record["counters_after"])
    )
    return {
        "p50_latency_s": pct(lat, 50),
        "p95_latency_s": pct(lat, 95),
        "p99_latency_s": pct(lat, 99),
        "max_latency_s": max(lat),
        "final_drift_s": worst_drift,
        "avg_batch": avg_batch(d[COUNT], d[EXEC]),
        "bls_tax": bls_tax(d[C_IN], d[C_INFER], d[C_OUT]),
        "queue_us_per_request": d[QUEUE] / d[COUNT] if d[COUNT] else 0.0,
        # Không phải chỉ số hiệu năng - là chứng cứ để đối chiếu hai lần đo với
        # nhau. Thor không báo được mình có bị hãm hay không (xem clock_mhz).
        "gpu_mhz_p50": record.get("clock", {}).get("p50", 0.0),
    }


def aggregate(records):
    """Nhiều run -> {ccu: metric median}. Run invalid bị loại trước khi tính.

    Median chứ không trung bình: một run dính throttle nhiệt của Thor sẽ kéo
    trung bình đi rất xa, còn median thì bỏ qua nó.
    """
    by_ccu = {}
    for rec in records:
        if rec.get("valid", True):
            by_ccu.setdefault(rec["ccu"], []).append(run_summary(rec))

    invalid_ccus = {r["ccu"] for r in records} - set(by_ccu)
    if invalid_ccus:
        raise ValueError(f"CCU {sorted(invalid_ccus)} không còn run hợp lệ nào - chạy lại")

    return {
        ccu: {k: statistics.median([s[k] for s in summaries]) for k in summaries[0]}
        for ccu, summaries in by_ccu.items()
    }
