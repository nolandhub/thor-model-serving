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

    Con số đáng nhìn nhất là hàng 'ngoài tầng': nó là overhead của Triton +
    python + serialize. Nếu nó lớn thì tách model thành nhiều tầng sẽ nhân
    overhead đó lên chứ không giảm đi, và cả hướng kiến trúc phải đổi.
    """
    if WRAPPER not in bd:
        raise ValueError(f"thiếu tầng {WRAPPER!r} - model đo chưa chạy request nào?")
    chunk = bd[WRAPPER]
    inner = {k: v for k, v in bd.items() if k != WRAPPER}
    inner_ms = sum(v["mean_ms"] for v in inner.values())

    lines = [
        f"chunk quan sát được: {chunk['count']:.0f}",
        "",
        "| tầng | mean (ms) | % của chunk |",
        "|---|---|---|",
    ]
    for name, v in sorted(inner.items(), key=lambda kv: -kv[1]["mean_ms"]):
        share = v["mean_ms"] / chunk["mean_ms"] * 100 if chunk["mean_ms"] else 0.0
        lines.append(f"| {name} | {v['mean_ms']:.3f} | {share:.1f}% |")

    out_ms = chunk["mean_ms"] - inner_ms
    out_share = out_ms / chunk["mean_ms"] * 100 if chunk["mean_ms"] else 0.0
    lines += [
        f"| **ngoài tầng** | **{out_ms:.3f}** | **{out_share:.1f}%** |",
        f"| _chunk (tổng)_ | _{chunk['mean_ms']:.3f}_ | _100%_ |",
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
