# ABOUTME: Đọc counter từ endpoint /metrics của Triton - nguồn số phía server của bench
# ABOUTME: Phần parse là hàm thuần, test được mà không cần container nào chạy

import re
import subprocess

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
