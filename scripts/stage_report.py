# ABOUTME: In bảng phân rã thời gian theo tầng của asr_streaming_prof từ /metrics của Triton
# ABOUTME: Chạy: docker compose ... run --rm --entrypoint python3 bench scripts/stage_report.py

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.triton_metrics import parse_exposition, stage_breakdown  # noqa: E402

# Bọc trọn _handle nên nó CHỨA ba tầng kia - không nằm cùng hàng trong bảng.
WRAPPER = "chunk"


def render(bd):
    """Bảng phân rã. Hàng cuối là phần KHÔNG nằm trong tầng nào.

    Tỷ trọng tính bằng TỔNG thời gian, không phải mean/mean. Các tầng có số
    lần gọi khác nhau: encoder tiêu thụ decode_chunk_len khung mỗi bước trong
    khi một chunk chỉ nạp 20 khung, nên nó chạy ~0.6 lần mỗi chunk. Lấy mean
    của tầng chia mean của chunk sẽ ra tỷ trọng vượt 100% và phần dư âm.

    Cột "lần/chunk" giữ lại vì nó tự nói lên nhịp: <1 nghĩa là tầng đó không
    chạy ở mọi chunk, và mean của nó không so thẳng với mean chunk được.
    """
    if WRAPPER not in bd:
        raise ValueError(f"thiếu tầng {WRAPPER!r} - model đo chưa chạy request nào?")
    chunk = bd[WRAPPER]
    inner = {k: v for k, v in bd.items() if k != WRAPPER}
    inner_sum = sum(v["sum_s"] for v in inner.values())
    total_s = chunk["sum_s"]
    n = chunk["count"]

    def row(name, sum_s, mean_ms, per_chunk):
        share = sum_s / total_s * 100 if total_s else 0.0
        ms_per_chunk = sum_s / n * 1000 if n else 0.0
        return (f"| {name} | {mean_ms:.2f} | {per_chunk} | "
                f"{ms_per_chunk:.2f} | {share:.1f}% |")

    lines = [
        f"chunk quan sát được: {n:.0f}",
        "",
        "| tầng | mean/lần (ms) | lần/chunk | ms/chunk | % của chunk |",
        "|---|---|---|---|---|",
    ]
    for name, v in sorted(inner.items(), key=lambda kv: -kv[1]["sum_s"]):
        per_chunk = f"{v['count'] / n:.2f}" if n else "-"
        lines.append(row(name, v["sum_s"], v["mean_ms"], per_chunk))

    out_s = total_s - inner_sum
    lines += [
        row("**ngoài tầng**", out_s, out_s / n * 1000 if n else 0.0, "-"),
        row("_chunk (tổng)_", total_s, chunk["mean_ms"], "1.00"),
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-url", default="http://asr:8002/metrics")
    ap.add_argument("--model", default="asr_streaming_prof")
    args = ap.parse_args()

    # opener KHÔNG proxy, cùng lý do đã ghi ở bench/triton_metrics.snapshot_http
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(args.metrics_url, timeout=5) as resp:
        text = resp.read().decode()
    print(render(stage_breakdown(parse_exposition(text), args.model)))


if __name__ == "__main__":
    main()
