# ABOUTME: Đọc counter từ endpoint /metrics của Triton - nguồn số phía server của bench
# ABOUTME: Phần parse là hàm thuần, test được mà không cần container nào chạy

import re
import subprocess
import urllib.request

# name{k="v",k2="v2"} 1.234e+06   - phần {...} có thể không có
_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)$')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')


def parse_exposition(text):
    """Định dạng text của Prometheus -> [(name, labels, value)].

    Đủ dùng cho output của Triton, không phải parser đầy đủ đặc tả: Triton không
    xuất label có ký tự escape, không xuất timestamp. Viết đủ cho thứ mình đọc
    thay vì kéo về prometheus_client chỉ để bóc mấy dòng text.
    """
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, raw_labels, raw_value = m.groups()
        labels = dict(_LABEL.findall(raw_labels)) if raw_labels else {}
        # float() chứ không int(): Triton xuất duration dạng 1.234e+06
        samples.append((name, labels, float(raw_value)))
    return samples


def counters_for_model(samples, model):
    """Gộp mọi counter của một model, cộng dồn qua các version.

    Rỗng thì báo lỗi chứ không trả {}: gõ sai tên model mà im lặng thì
    diff_counters cũng rỗng theo, avg_batch không bao giờ được gọi, và bench
    báo "xong" trong khi chưa đo được gì cả.
    """
    out = {}
    for name, labels, value in samples:
        if labels.get("model") == model:
            out[name] = out.get(name, 0.0) + value
    if not out:
        raise ValueError(f"không có mẫu nào cho model {model!r} - kiểm tên model và server")
    return out


def snapshot(container, model, port=8002):
    """Chụp counter của một model tại đúng thời điểm gọi.

    Đọc từ BÊN TRONG container: compose cố ý không publish 8002 ra host vì data
    plane có thể mở ra LAN mà Triton không có auth - bench không phá thế đó.

    Không đi qua Prometheus: chu kỳ scrape 15s trong khi một run chỉ ~20s, hai
    snapshot dễ rơi vào cùng một điểm dữ liệu và Δ ra 0.

    python3 chứ không curl, cùng lý do đã ghi ở healthcheck trong compose.yaml:
    image base chắc chắn có python3, không chắc có curl.
    """
    script = (
        "import urllib.request;"
        f"print(urllib.request.urlopen('http://localhost:{port}/metrics', timeout=5)"
        ".read().decode())"
    )
    text = subprocess.run(
        ["docker", "exec", container, "python3", "-c", script],
        check=True, capture_output=True, text=True,
    ).stdout
    return counters_for_model(parse_exposition(text), model)


def snapshot_http(url, model):
    """Chụp counter qua HTTP - dùng khi bench chạy TRONG network của compose.

    Từ trong network thì asr:8002 tới thẳng được, không cần publish ra host và
    cũng không có docker socket để mà exec. Cùng lý do với snapshot(): đọc
    thẳng endpoint chứ không hỏi Prometheus, vì chu kỳ scrape 15s dài hơn cả
    một run và hai snapshot sẽ rơi trúng cùng một điểm dữ liệu.
    """
    # opener KHÔNG proxy: url này là tên service trong network compose, đi qua
    # proxy công ty là chắc chắn refuse (xem without_proxy_env trong run_asr).
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as resp:
        text = resp.read().decode()
    return counters_for_model(parse_exposition(text), model)


def stage_breakdown(samples, model):
    """Phân rã thời gian theo tầng -> {stage: {count, sum_s, mean_ms, share}}.

    counters_for_model() không dùng được ở đây: nó gộp theo TÊN metric, mà cả
    ba tầng dùng chung tên voice_stage_seconds_sum và chỉ khác nhau ở label
    stage - gộp theo tên là cả ba dồn vào một số duy nhất.

    share tính trên tổng sum của các tầng, không phải trên wall-clock của
    chunk. Ba tầng cộng lại có thể KHÔNG bằng 100% thời gian một chunk - phần
    thiếu chính là overhead ngoài tầng (Triton, serialize, python), và đó mới
    là con số cần nhìn.
    """
    acc = {}
    for name, labels, value in samples:
        if labels.get("model") != model:
            continue
        stage = labels.get("stage")
        if stage is None:
            continue
        if name.endswith("_sum"):
            key = "sum_s"
        elif name.endswith("_count"):
            key = "count"
        else:
            continue   # _bucket: không cần cho bảng phân rã
        slot = acc.setdefault(stage, {"count": 0.0, "sum_s": 0.0})
        slot[key] += value
    if not acc:
        raise ValueError(
            f"không có mẫu voice_stage_seconds nào cho model {model!r} - "
            "kiểm tên model và xem model đã chạy request nào chưa"
        )
    total = sum(s["sum_s"] for s in acc.values())
    for s in acc.values():
        s["mean_ms"] = (s["sum_s"] / s["count"] * 1000) if s["count"] else 0.0
        s["share"] = (s["sum_s"] / total) if total else 0.0
    return acc
