#!/usr/bin/env python3
# ABOUTME: Bench OPEN-LOOP reranker - quét QPS tìm sức chứa, thứ closed-loop không đo được
# ABOUTME: Chạy: python3 bench/run_rerank.py --url http://127.0.0.1:9012 --qps 1,2,4,8

"""Gửi theo nhịp cố định, KHÔNG chờ response trước mới gửi tiếp.

Khác bench/rerank_bench.py đúng một chỗ, và đó là cả vấn đề: closed-loop tự
giảm tải khi server chậm (gửi -> chờ -> gửi tiếp), nên nó đo được "một request
mất bao lâu khi server rảnh" mà KHÔNG bao giờ thấy hàng đợi dồn. Open-loop giữ
nguyên nhịp gửi dù server có kịp hay không, nên nó thấy.

Số quyết định ở đây là `lag cuối`, không phải p99 - đúng bài học đã trả giá ở
bench ASR: p99 nhìn từng request rời rạc và có thể vẫn đẹp trong lúc hàng đợi
phình vô hạn, còn lag tăng dần thì chỉ có một nghĩa.

stdout chỉ có bảng; tiến độ và cảnh báo đi ra stderr để `... > bang.md` dùng
được ngay.
"""

import argparse
import queue
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench.report import diff_counters, max_ccu_within_budget, pct  # noqa: E402
from bench.rerank_bench import (  # noqa: E402
    HwSampler, load_queries, opener_no_proxy, rerank, sample, table,
)
from bench.run_asr import ClockSampler, without_proxy_env  # noqa: E402
from bench.schedule import send_deadlines  # noqa: E402
from bench.tei_metrics import send_lags, server_side, snapshot_http, tail_lag  # noqa: E402

COLUMNS = (
    "QPS set", "QPS got", "P50 (ms)", "P95 (ms)", "P99 (ms)", "tail lag (s)",
    "avg_batch", "queue ms", "infer ms", "GPU busy%",
)

# Lag để 3 chữ số và LUÔN có dấu: dấu là thông tin thật, âm nghĩa là worker
# nhận việc trước mốc (server còn dư nhịp), dương là đã trễ. Latency ms thì số
# lẻ vô nghĩa nên để 0.
FMT = ("{:.2f}", "{:.0f}", "{:.0f}", "{:.0f}", "{:+.3f}",
       "{:.1f}", "{:.1f}", "{:.1f}", "{:.1f}")


def _worker(url, jobs, payloads, deadlines, results, errors, state):
    """Một worker: nhận job, ngủ tới mốc TUYỆT ĐỐI của job đó, gửi, ghi lại.

    Opener riêng cho mỗi worker: OpenerDirector không hứa an toàn đa luồng, và
    dùng chung thì cái đo được có thể là tranh chấp bên trong client chứ không
    phải bên trong server - đúng lỗi mà bench ASR đã tránh bằng một connection
    cho mỗi stream.

    Ghi kết quả bằng MỘT append một tuple, không phải ba append vào ba list:
    ba list thì hai thread xen kẽ nhau là mốc/pick/latency lệch hàng, và bảng
    vẫn in ra bình thường với số của request khác ghép vào nhau.
    """
    opener = opener_no_proxy()
    while True:
        try:
            i = jobs.get_nowait()
        except queue.Empty:
            return
        # Ngủ tới mốc tuyệt đối. Đến muộn thì gửi ngay chứ không kéo lịch theo -
        # kéo theo là tự giấu mất tình trạng quá tải, tức là biến open-loop
        # thành closed-loop và hỏng đúng thứ bài bench này sinh ra để đo.
        delay = deadlines[i] - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        pick = time.monotonic()
        with state["lock"]:
            state["inflight"] += 1
            state["peak"] = max(state["peak"], state["inflight"])
        try:
            q, docs = payloads[i]
            t0 = time.perf_counter()
            rerank(opener, url, q, docs)
            results.append((deadlines[i], pick, time.perf_counter() - t0))
        except Exception as exc:                  # noqa: BLE001 - gom về thread chính
            errors.append(exc)
            return
        finally:
            with state["lock"]:
                state["inflight"] -= 1


def one_level(args, queries, qps):
    """Một mức QPS -> (lats giây, lags giây, sys_stats, qps đạt, chạm trần pool?).

    Dựng SẴN toàn bộ payload trước khi bấm giờ. prep_docs() có cắt và sắp xếp,
    tức là tốn CPU thật; làm nó sau mốc gửi thì chi phí của client chảy thẳng
    vào lag và bench sẽ báo server quá tải trong khi thủ phạm là chính nó.
    """
    n = max(1, round(qps * args.duration))
    payloads = [sample(args, queries, args.k, i) for i in range(n)]

    jobs = queue.Queue()
    for i in range(n):
        jobs.put(i)

    before = snapshot_http(args.metrics_url)
    clock, hw = ClockSampler(), HwSampler(args.container)
    clock.start(), hw.start()

    # Mốc bắt đầu đặt trễ một nhịp: worker cuối cùng cũng phải kịp khởi động
    # xong trước mốc đầu tiên, không thì request #0 đã muộn ngay khi sinh ra và
    # lag của cả run bị cộng thêm một hằng số không liên quan gì tới server.
    t_start = time.monotonic() + 0.5
    deadlines = send_deadlines(t_start, n, 1.0 / qps)

    results, errors = [], []
    state = {"lock": threading.Lock(), "inflight": 0, "peak": 0}
    threads = [
        threading.Thread(
            target=_worker,
            args=(args.url, jobs, payloads, deadlines, results, errors, state),
            daemon=True,
        )
        for _ in range(args.workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t_start

    hw_stats, clock_stats = hw.stop(), clock.stop()
    after = snapshot_http(args.metrics_url)

    if errors:
        raise errors[0]
    if not results:
        raise SystemExit(f"QPS {qps}: no request completed - check server")

    # Sắp theo mốc gửi để tail_lag nhìn đúng phần ĐUÔI của run. Kết quả về theo
    # thứ tự hoàn thành, không phải thứ tự phát.
    results.sort()
    lags = send_lags([r[1] for r in results], [r[0] for r in results])
    lats = [r[2] for r in results]

    sys_stats = server_side(diff_counters(before, after))
    sys_stats.update(hw_stats)
    sys_stats["gpu_mhz_p50"] = clock_stats["p50"]
    return lats, lags, sys_stats, len(results) / wall, state["peak"] >= args.workers


def row(qps, lats, lags, s, got):
    ms = [v * 1000 for v in lats]
    return (qps, got, pct(ms, 50), pct(ms, 95), pct(ms, 99), tail_lag(lags),
            s["avg_batch"], s["queue_ms_per_req"], s["infer_ms_per_req"],
            s["gpu_busy_pct"])


def verdict(p99_by_qps, lag_by_qps, args):
    """Sức chứa = mức QPS cao nhất qua CẢ HAI cửa, lấy cái thấp hơn.

    Chỉ xét p99 thì bỏ sót ca hàng đợi dồn mà từng request vẫn nhanh; chỉ xét
    lag thì bỏ sót ca server theo kịp nhịp nhưng mỗi request chậm quá ngưỡng
    dùng được. Phải qua cả hai mới gọi là chịu được mức đó.
    """
    by_p99 = max_ccu_within_budget(p99_by_qps, args.budget)
    by_lag = max_ccu_within_budget(lag_by_qps, args.lag_budget)
    return min(by_p99, by_lag), by_p99, by_lag


def main():
    ap = argparse.ArgumentParser(description="open-loop reranker bench: throughput + drift")
    ap.add_argument("--url", default="http://127.0.0.1:9012", help="9012 = lab, 9002 = PROD")
    ap.add_argument("--metrics-url", help="defaults to --url + /metrics")
    ap.add_argument("--container", default="rerank-lab",
                    help="container name - reads real GPU busy%% via nvidia-smi")
    ap.add_argument("--qps", default="1,2,4,8", help="send-rate levels to sweep")
    ap.add_argument("--k", type=int, default=100, help="top-k per request")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds per level")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--workers", type=int, default=64,
                    help="client-side in-flight cap; hitting it invalidates the level")
    ap.add_argument("--budget", type=float, default=1.0, help="p99 budget (seconds)")
    ap.add_argument("--lag-budget", type=float, default=0.5,
                    help="tail-lag budget (seconds) - exceeding it means the queue is backing up")
    ap.add_argument("--words", type=int, default=120, help="words per doc when SYNTHESIZING")
    ap.add_argument("--file", help="use real data instead of synthetic")
    ap.add_argument("--max-doc-tokens", type=int,
                    help="truncate each doc to token cap before sending (e.g. 384)")
    ap.add_argument("--sort-by-length", action="store_true",
                    help="sort docs by length to cut padding; ranking unchanged")
    args = ap.parse_args()
    args.metrics_url = args.metrics_url or args.url + "/metrics"

    # Cùng bẫy #3 của bench/README, dạng khác: docker CLI và nvidia-smi phải
    # chạy không proxy.
    import os
    clean = without_proxy_env(os.environ)
    os.environ.clear()
    os.environ.update(clean)

    # Giữ nguyên thứ tự gõ, chỉ khử trùng lặp - xem lý do ở rerank_bench.py.
    # Riêng ở đây thứ tự KHÔNG ảnh hưởng kết luận sức chứa: verdict() tự sắp
    # khoá của nó rồi quét từ dưới lên. Nó chỉ đổi thứ tự ĐO, tức là đổi mức
    # nào phải gánh phần máy còn nguội và mức nào chạy lúc máy đã nóng.
    levels = list(dict.fromkeys(float(x) for x in args.qps.split(",")))
    queries = load_queries(args.file) if args.file else None
    opener = opener_no_proxy()

    # Warmup closed-loop, kết quả vứt đi: nạp model và ramp clock GPU phải rơi
    # vào đây, không rơi vào mức QPS đo đầu tiên. Bẫy #1 của bench ASR - mức đo
    # đầu tiên thành rác, mà verdict lại quét TỪ DƯỚI LÊN nên nó tin số rác đó
    # và kết luận sức chứa bằng 0.
    for i in range(args.warmup):
        q, docs = sample(args, queries, args.k, i)
        rerank(opener, args.url, q, docs)

    rows, p99_by_qps, lag_by_qps = [], {}, {}
    for qps in levels:
        print(f"  QPS {qps:g} ({round(qps * args.duration)} requests)", file=sys.stderr)
        lats, lags, s, got, capped = one_level(args, queries, qps)
        if capped:
            print(f"    WARNING: hit the {args.workers}-worker cap - this level measures "
                  "CLIENT capacity, not server capacity. Raise --workers and re-run.",
                  file=sys.stderr)
        rows.append(row(qps, lats, lags, s, got))
        p99_by_qps[qps] = pct(lats, 99)
        lag_by_qps[qps] = tail_lag(lags)

    print(table(COLUMNS, rows, fmt=FMT))
    print()
    cap, by_p99, by_lag = verdict(p99_by_qps, lag_by_qps, args)
    print(f"**Capacity: {cap:g} QPS** at top-k {args.k} "
          f"(p99 <= {args.budget}s allows {by_p99:g}, lag <= {args.lag_budget}s allows {by_lag:g}).")
    print()
    print("`tail lag` flat near 0 = server keeps up with the send rate. RISING = every "
          "worker is busy, work piles up and never catches up - that is collapse, even "
          "when p99 on the same row still looks fine.")


if __name__ == "__main__":
    main()
