# ABOUTME: Driver bench asr_streaming - quét CCU, gửi chunk đúng nhịp realtime, chụp counter Triton
# ABOUTME: Chạy: python bench/run_asr.py --ccu 1,2,4,8 (cần Thor đang up; hàm thuần thì không)

import argparse
import functools
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.report import aggregate, max_ccu_within_budget  # noqa: E402
from bench.schedule import send_deadlines  # noqa: E402
from bench.triton_metrics import snapshot, snapshot_http  # noqa: E402

MODEL = "asr_streaming"
# Cột nào lên bảng và in ra sao. avg_batch để 2 chữ số vì cả quyết định BLS nằm
# ở chỗ nó là 1.0 hay lớn hơn - làm tròn 1 chữ số sẽ nuốt mất khác biệt đó.
COLUMNS = [
    ("p99_latency_s", "p99 (s)", "{:.3f}"),
    ("final_drift_s", "drift cuối (s)", "{:+.2f}"),
    ("avg_batch", "avg_batch", "{:.2f}"),
    ("bls_tax", "bls_tax", "{:.2f}"),
    ("queue_us_per_request", "queue (µs/req)", "{:.0f}"),
]


def parse_ccus(spec):
    """'1,2,4,8' -> [1, 2, 4, 8]. Tăng dần, không trùng.

    Bắt buộc tăng dần vì max_ccu_within_budget quét từ dưới lên và dừng ở mức
    đầu tiên vỡ - chạy lộn thứ tự thì thứ nó đọc được không còn là sức chứa.
    """
    out = set()
    for part in spec.split(","):
        ccu = int(part)
        if ccu <= 0:
            raise ValueError(f"CCU {ccu} không dương")
        out.add(ccu)
    if not out:
        raise ValueError(f"không đọc được mức CCU nào từ {spec!r}")
    return sorted(out)


def metrics_reader(container, metrics_url):
    """Hàm chụp counter phù hợp với chỗ bench đang đứng.

    Trong container (service `bench` của compose): gọi thẳng URL. Trên host:
    docker exec, vì compose cố ý không publish 8002 - data plane có thể mở ra
    LAN mà Triton không có auth, bench không được phá thế đó chỉ để đo cho tiện.
    """
    if metrics_url:
        return functools.partial(snapshot_http, metrics_url)
    if container:
        return functools.partial(snapshot, container)
    raise ValueError("không có nguồn metrics: cần --metrics-url hoặc --container")


def throttled(reasons):
    """Trường clocks_throttle_reasons.active của nvidia-smi -> đang bị hãm?

    0x0 (không lý do) và 0x1 (GpuIdle) là bình thường; bất kỳ bit nào khác là
    clock đang bị kéo xuống và mọi số đo trong lát đó sai 2-3 lần. Thor có thể
    không xuất trường này - đọc không ra thì coi như không hãm, chứ loại sạch
    run vì một trường thiếu thì bench không bao giờ có kết quả.
    """
    if not re.fullmatch(r"0x[0-9a-fA-F]+", reasons.strip()):
        return False
    return int(reasons, 16) & ~0x1 != 0


def gpu_state():
    """(mô tả, đang bị hãm?) tại thời điểm gọi. Không có nvidia-smi thì bỏ qua."""
    q = "clocks.sm,temperature.gpu,clocks_throttle_reasons.active"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "nvidia-smi không đọc được", False
    return out, throttled(out.split(",")[-1])


def _run_stream(url, chunks, chunk_s, t_start, seq_id, out, errors):
    """Một stream: gửi chunk theo mốc tuyệt đối, ghi lại thời điểm gửi và nhận.

    Mỗi stream một connection riêng: dùng chung một InferenceServerClient thì
    các thread tranh nhau cùng một gRPC stream, và cái đo được là hàng đợi bên
    trong client chứ không phải bên trong server.
    """
    import tritonclient.grpc as grpcclient

    recv_q = queue.Queue()
    client = grpcclient.InferenceServerClient(url)
    client.start_stream(callback=lambda result, error: recv_q.put((time.monotonic(), error)))
    try:
        for i, deadline in enumerate(send_deadlines(t_start, len(chunks), chunk_s)):
            # Ngủ tới mốc TUYỆT ĐỐI. Đến muộn (server ăn hết CPU) thì gửi ngay,
            # không kéo lịch theo - kéo theo là giấu mất tình trạng quá tải.
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            part = chunks[i]
            inp = grpcclient.InferInput("AUDIO_CHUNK", [1, len(part)], "FP32")
            inp.set_data_from_numpy(part.reshape(1, -1))
            out["send"].append(time.monotonic())
            client.async_stream_infer(
                MODEL, [inp], sequence_id=seq_id,
                sequence_start=(i == 0), sequence_end=(i == len(chunks) - 1),
            )
        while len(out["recv"]) < len(chunks):
            t, error = recv_q.get(timeout=30)
            if error:
                raise RuntimeError(f"server trả lỗi: {error}")
            out["recv"].append(t)
    except Exception as exc:                      # noqa: BLE001 - gom về thread chính
        errors.append(exc)
    finally:
        client.stop_stream()


def run_once(args, read_counters, ccu, chunks, run_idx):
    """Một run ở một mức CCU -> record đúng dạng report.run_summary() nhận."""
    before = read_counters(MODEL)
    gpu_before, hot_before = gpu_state()

    # Mọi stream chung một t_start: chúng phải chồng lấn thật thì dynamic
    # batcher mới có gì để gom, và avg_batch mới nói lên điều gì.
    t_start = time.monotonic() + 0.5   # đệm để thread cuối kịp mở connection
    streams = [{"t_start": t_start, "send": [], "recv": []} for _ in range(ccu)]
    errors = []
    base_seq = int(time.time()) % 2**28 * 16 + run_idx * 64
    threads = [
        threading.Thread(
            target=_run_stream,
            args=(args.url, chunks, args.chunk_ms / 1000, t_start,
                  base_seq + i + 1, streams[i], errors),
        )
        for i in range(ccu)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]

    after = read_counters(MODEL)
    gpu_after, hot_after = gpu_state()
    valid = not (hot_before or hot_after)
    if not valid:
        print(f"    bỏ run: GPU bị hãm ({gpu_before} -> {gpu_after})")
    return {
        "ccu": ccu, "run": run_idx, "chunk_s": args.chunk_ms / 1000,
        "valid": valid, "streams": streams,
        "counters_before": before, "counters_after": after,
    }


def markdown_table(agg):
    head = "| CCU | " + " | ".join(label for _, label, _ in COLUMNS) + " |"
    sep = "|---" * (len(COLUMNS) + 1) + "|"
    rows = [
        "| " + str(ccu) + " | "
        + " | ".join(fmt.format(agg[ccu][key]) for key, _, fmt in COLUMNS) + " |"
        for ccu in sorted(agg)
    ]
    return "\n".join([head, sep, *rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--container", default="thor-asr-triton",
                    help="chạy trên host: đọc /metrics bằng docker exec vào container này")
    ap.add_argument("--metrics-url",
                    help="chạy trong network compose: đọc thẳng, vd http://asr:8002/metrics")
    ap.add_argument("--wav", default=str(ROOT / "tests/assets/sample_vi_long.wav"))
    ap.add_argument("--chunk-ms", type=int, default=200)
    # Trần 8 là max_candidate_sequences trong config.pbtxt - vượt qua thì các
    # sequence dư nằm chờ ngoài cửa và cái đo được không còn là sức chứa model.
    ap.add_argument("--ccu", default="1,2,4,8")
    ap.add_argument("--runs", type=int, default=3, help="số run mỗi mức, lấy median")
    ap.add_argument("--budget-s", type=float, default=0.5, help="ngưỡng p99 để tính CCU tối đa")
    ap.add_argument("--cooldown-s", type=float, default=10.0, help="nghỉ giữa hai run cho nguội")
    ap.add_argument("--out", default=str(ROOT / "bench/results/asr_streaming.md"))
    args = ap.parse_args()

    read_counters = metrics_reader(args.container, args.metrics_url)

    from client.common import chunk_wav, load_wav_16k

    chunks = chunk_wav(load_wav_16k(args.wav), args.chunk_ms)
    ccus = parse_ccus(args.ccu)
    print(f"{len(chunks)} chunk x {args.chunk_ms}ms = {len(chunks) * args.chunk_ms / 1000:.1f}s "
          f"audio/stream | CCU {ccus} | {args.runs} run/mức")

    records = []
    for ccu in ccus:
        for run_idx in range(args.runs):
            print(f"  CCU {ccu} run {run_idx + 1}/{args.runs}")
            records.append(run_once(args, read_counters, ccu, chunks, run_idx))
            time.sleep(args.cooldown_s)

    agg = aggregate(records)
    p99 = {ccu: m["p99_latency_s"] for ccu, m in agg.items()}
    table = markdown_table(agg)
    verdict = (
        f"- CCU tối đa còn dưới p99 {args.budget_s}s: **{max_ccu_within_budget(p99, args.budget_s)}**\n"
        f"- avg_batch cao nhất: **{max(m['avg_batch'] for m in agg.values()):.2f}** "
        "(≈1.0 nghĩa là dynamic batcher không gom được gì - BLS chỉ còn phần chi phí)"
    )
    gpu, _ = gpu_state()
    body = (
        f"# Bench `{MODEL}`\n\n"
        f"{len(chunks) * args.chunk_ms / 1000:.1f}s audio/stream, chunk {args.chunk_ms}ms, "
        f"median của {args.runs} run. GPU lúc kết thúc: `{gpu}`.\n\n"
        f"{table}\n\n{verdict}\n"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(f"\n{table}\n\n{verdict}\n\n-> {out}")


if __name__ == "__main__":
    main()
