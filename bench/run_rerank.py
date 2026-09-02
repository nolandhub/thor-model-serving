# ABOUTME: Driver bench reranker - `perf` quét QPS/độ dài open-loop, `quality` chấm NDCG hoặc so baseline
# ABOUTME: Chạy: python bench/run_rerank.py perf --qps 1,4,8,16 --slo-ms 200

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench.report import diff_counters, pct  # noqa: E402
from bench.schedule import send_deadlines  # noqa: E402
from bench.tei_metrics import send_lags, server_side, snapshot_http  # noqa: E402
from bench.rerank_report import (  # noqa: E402
    TIE_EPS, batch_fill, concurrency, kendall_tau, max_qps_within_slo, mrr_at_k,
    ndcg_at_k, pairs_per_second, quality_summary, recall_at_k, significant_swaps,
    topk_overlap,
)
# ClockSampler đọc /sys/class/devfreq của Thor, không dính gì tới ASR. Import lại
# chứ không chép: run_asr chỉ nạp tritonclient bên trong _run_stream.
from bench.run_asr import ClockSampler, without_proxy_env  # noqa: E402

SAMPLE_S = 1.0
# Một mốc cho cả lượt chạy: bảng .md và file điểm .json của cùng lượt phải
# trùng hậu tố thì mới ghép lại được. Gọi strftime hai lần có thể lệch giây.
RUN_STAMP = time.strftime("%Y-%m-%d_%H%M%S")
RESULT_DIR = ROOT / "bench/result"


def default_out(what, ext="md"):
    """bench/result/rerank_<việc>_<ngày>_<giờ>.<ext> - tên tự nói nó là gì và đo lúc nào."""
    return str(RESULT_DIR / f"rerank_{what}_{RUN_STAMP}.{ext}")

COLUMNS = [
    ("p50_total_s", "p50 (s)", "{:.3f}"),
    ("p95_total_s", "p95 (s)", "{:.3f}"),
    ("p99_total_s", "p99 (s)", "{:.3f}"),
    ("final_drift_s", "drift", "{:+.3f}"),
    ("pairs_per_s", "cặp/s", "{:.0f}"),
    ("concurrency", "CCU", "{:.1f}"),
    ("avg_batch", "avg_batch", "{:.2f}"),
    ("batch_fill", "batch đầy", "{:.0%}"),
    ("queue_ms_per_req", "queue ms", "{:.1f}"),
    ("infer_ms_per_req", "infer ms", "{:.1f}"),
    ("gpu_busy_pct", "GPU bận%", "{:.0f}"),
    ("gpu_mhz_p50", "GPC MHz", "{:.0f}"),
    ("gpu_mib_peak", "GPU MiB", "{:.0f}"),
]


# --- lấy mẫu phần cứng ------------------------------------------------------

def container_pids(name):
    try:
        out = subprocess.run(["docker", "top", name, "-eo", "pid"],
                             check=True, capture_output=True, text=True, timeout=10).stdout
        return {line.strip() for line in out.splitlines()[1:] if line.strip()}
    except (OSError, subprocess.SubprocessError):
        return set()


def gpu_sample(pids):
    """(utilization %, MiB mà các pid này đang giữ trên GPU).

    Phải đi qua --query-compute-apps chứ KHÔNG dùng `docker stats`: Thor dùng
    LPDDR chung CPU/GPU, docker stats chỉ thấy RSS phía CPU và bỏ sót phần lớn
    bộ nhớ thật (đã thấy 27.6 GiB hiện ra trong khi 60 GiB nằm ở GPU).
    """
    util = 0.0
    try:
        util = float(subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10).stdout.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    mib = 0.0
    if pids:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] in pids:
                    mib += float(parts[1])
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return util, mib


class HwSampler(threading.Thread):
    """Lấy mẫu GPU LÚC ĐANG TẢI. Đọc một phát lúc xong thì GPU đã rảnh và lần
    nào cũng ra 0% - một con số đúng mà vô nghĩa."""

    def __init__(self, container):
        super().__init__(daemon=True)
        self.pids = container_pids(container)
        self.utils, self.mibs = [], []
        self._done = threading.Event()

    def run(self):
        while not self._done.is_set():
            u, m = gpu_sample(self.pids)
            self.utils.append(u)
            self.mibs.append(m)
            self._done.wait(SAMPLE_S)

    def stop(self):
        self._done.set()
        self.join(timeout=3)
        return {
            # Trung bình các mẫu LÚC CHẠY. NVML tính utilization là "có kernel
            # nào đang chạy không", KHÔNG phải "GPU đầy bao nhiêu phần" - 79%
            # với 8W đã từng xảy ra. Đọc nó là "GPU có bận không", không phải
            # "GPU đã dùng hết chưa".
            "gpu_busy_pct": statistics.mean(self.utils) if self.utils else 0.0,
            "gpu_mib_peak": max(self.mibs) if self.mibs else 0.0,
        }


# --- dữ liệu ----------------------------------------------------------------

def build_payload(args):
    from rerank_probe import load_file, make_docs
    if args.file:
        query, docs, _ = load_file(args.file, args.k)
        source = f"{args.file} (K={len(docs)})"
    else:
        from rerank_probe import QUERY
        query, docs = QUERY, make_docs(args.k or 15, args.words)
        source = f"tự sinh K={len(docs)} x {args.words} từ"
        if len(set(docs)) < len(docs):
            source += f"  [chỉ {len(set(docs))} văn bản duy nhất]"
    return source, query, docs


def opener_no_proxy():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def rerank(opener, url, query, docs, timeout=300):
    body = json.dumps({"query": query, "texts": docs, "raw_scores": True}).encode()
    req = urllib.request.Request(url + "/rerank", data=body,
                                 headers={"Content-Type": "application/json"})
    out = json.load(opener.open(req, timeout=timeout))
    scores = [0.0] * len(docs)
    for r in out:
        scores[r["index"]] = r["score"]
    return scores


def fetch_info(opener, url):
    """Cấu hình ĐANG chạy, hỏi từ server. TEI đọc cờ đúng một lần lúc khởi
    động, nên đổi cờ mà container cũ chưa chết là đo lại đúng cấu hình cũ."""
    with opener.open(url + "/info", timeout=5) as resp:
        return json.load(resp)


def environment():
    out = {}
    for key, cmd in (("containers", ["docker", "ps", "--format", "{{.Names}}"]),
                     ("gpu_procs", ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                    "--format=csv,noheader"])):
        try:
            out[key] = subprocess.run(cmd, check=True, capture_output=True,
                                      text=True, timeout=15).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out[key] = "(không đọc được)"
    return out


def write_report(path, text):
    """Số đo đắt hơn file rất nhiều - một lượt quét mất hàng chục phút. Ghi hỏng
    thì đổi chỗ ghi, tuyệt đối không ném lỗi làm mất cả lượt đo."""
    out = Path(path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"\nđã ghi {out}")
    except OSError as exc:
        fb = Path(tempfile.gettempdir()) / out.name
        fb.write_text(text, encoding="utf-8")
        print(f"\nKHÔNG ghi được {out}: {exc}\nđã ghi tạm {fb} - số đo không mất")
        print(f"sửa hẳn: sudo chown -R $(id -un):$(id -gn) {out.parent}")


# --- perf -------------------------------------------------------------------

def dispatch(opener, url, query, docs, qps, n, workers):
    """Phát n request theo mốc TUYỆT ĐỐI, không sleep cộng dồn - cộng dồn thì
    overhead client trôi vào mốc phát và drift đo được là của client."""
    picks, recvs, errors = [0.0] * n, [0.0] * n, []
    t0 = time.monotonic() + 0.5
    deadlines = send_deadlines(t0, n, 1.0 / qps)

    def one(i):
        picks[i] = time.monotonic()
        try:
            rerank(opener, url, query, docs)
        except Exception as exc:                      # noqa: BLE001
            errors.append(exc)
        recvs[i] = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, dl in enumerate(deadlines):
            gap = dl - time.monotonic()
            if gap > 0:
                time.sleep(gap)
            pool.submit(one, i)
    if errors:
        raise errors[0]
    return deadlines, picks, recvs


def perf_level(args, opener, query, docs, qps, pair_tokens, max_batch_tokens):
    n = max(1, int(round(qps * args.duration_s)))
    before = snapshot_http(args.metrics_url)
    clock, hw = ClockSampler(), HwSampler(args.container)
    clock.start(), hw.start()
    deadlines, picks, recvs = dispatch(opener, args.url, query, docs, qps, n, args.workers)
    hw_stats, clock_stats = hw.stop(), clock.stop()
    after = snapshot_http(args.metrics_url)

    # total = từ mốc ĐÁNG LẼ phát tới lúc nhận, gồm cả thời gian chờ worker.
    # recv-pick thì bỏ qua phần chờ đó và luôn đẹp, kể cả khi đang sập.
    total = [r - d for r, d in zip(recvs, deadlines)]
    elapsed = recvs[-1] - deadlines[0]
    s = {
        "p50_total_s": pct(total, 50), "p95_total_s": pct(total, 95),
        "p99_total_s": pct(total, 99), "max_total_s": max(total),
        "final_drift_s": send_lags(picks, deadlines)[-1],
        "achieved_qps": n / elapsed,
        "pairs_per_s": pairs_per_second(n, len(docs), elapsed),
        "gpu_mhz_p50": clock_stats["p50"],
    }
    s.update(hw_stats)
    s.update(server_side(diff_counters(before, after)))
    s["concurrency"] = concurrency(s["achieved_qps"], statistics.mean(total))
    s["batch_fill"] = batch_fill(s["avg_batch"], max_batch_tokens, pair_tokens)
    return s


def table(agg, first_col, columns=COLUMNS):
    head = f"| {first_col} | " + " | ".join(l for _, l, _ in columns) + " |"
    sep = "|---" * (len(columns) + 1) + "|"
    rows = [f"| {x:g} | " + " | ".join(f.format(agg[x][k]) for k, _, f in columns) + " |"
            for x in sorted(agg)]
    return "\n".join([head, sep, *rows])


def median_runs(by_level):
    """Median chứ không trung bình: Thor là máy dùng chung, một run dính hàng
    xóm nổi tải sẽ kéo trung bình đi rất xa còn median thì bỏ qua nó."""
    return {x: {k: statistics.median([s[k] for s in runs]) for k in runs[0]}
            for x, runs in by_level.items()}


def token_len(opener, url, text):
    try:
        req = urllib.request.Request(url + "/tokenize", data=json.dumps({"inputs": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        out = json.load(opener.open(req, timeout=30))
        return len(out[0]) if out and isinstance(out[0], list) else 0
    except Exception:                                  # noqa: BLE001
        return 0


def cmd_perf(args):
    opener = opener_no_proxy()
    info = fetch_info(opener, args.url)
    source, query, docs = build_payload(args)
    pair_tokens = token_len(opener, args.url, query + " " + docs[0]) or 1

    print(f"nguồn : {source} | ~{pair_tokens} token/cặp")
    print(f"server: TEI {info['version']} | dtype {info['model_dtype']} | "
          f"max_batch_tokens {info['max_batch_tokens']} | auto_truncate {info['auto_truncate']}")
    print(f"quét  : QPS {args.qps_levels} | {args.runs} run x {args.duration_s:g}s | "
          f"SLO p99 < {args.slo_ms}ms\n")

    print(f"  warmup {args.warmup} request (kết quả bỏ)")
    for _ in range(args.warmup):
        rerank(opener, args.url, query, docs)

    by_qps = {}
    for q in args.qps_levels:
        print(f"  QPS {q:g}")
        for r in range(args.runs):
            s = perf_level(args, opener, query, docs, q, pair_tokens, info["max_batch_tokens"])
            by_qps.setdefault(q, []).append(s)
            print(f"    run {r + 1}: p99 {s['p99_total_s']:.3f}s | drift {s['final_drift_s']:+.3f}s"
                  f" | avg_batch {s['avg_batch']:.2f} | {s['pairs_per_s']:.0f} cặp/s")
            time.sleep(args.cooldown_s)

    agg = median_runs(by_qps)
    qps_table = table(agg, "QPS")
    slo_s = args.slo_ms / 1000
    cap = max_qps_within_slo(agg, slo_s)
    cap_line = (f"**Sức chứa: {cap:g} QPS** với p99 < {args.slo_ms}ms"
                f" (≈ {agg[cap]['concurrency']:.1f} request đồng thời)"
                if cap else f"**Không mức nào đạt p99 < {args.slo_ms}ms**")
    print("\n" + qps_table + "\n\n" + cap_line)

    len_table = ""
    if args.lengths:
        print(f"\n  quét độ dài ở QPS {args.length_qps:g}")
        from rerank_probe import make_docs
        by_len = {}
        for w in args.lengths:
            d = make_docs(args.k or 15, w)
            pt = token_len(opener, args.url, query + " " + d[0]) or 1
            for _ in range(args.runs):
                by_len.setdefault(pt, []).append(
                    perf_level(args, opener, query, d, args.length_qps, pt,
                               info["max_batch_tokens"]))
            print(f"    {w} từ = ~{pt} token/cặp: "
                  f"p99 {by_len[pt][-1]['p99_total_s']:.3f}s")
            time.sleep(args.cooldown_s)
        len_table = "\n\n## Độ trễ theo độ dài input\n\n" + table(
            median_runs(by_len), f"token/cặp @ {args.length_qps:g} QPS")
        print(len_table)

    write_report(args.out,
                 f"# Bench rerank - hiệu năng - {time.strftime('%Y-%m-%d %H:%M')}\n\n"
                 f"- nguồn: {source} (~{pair_tokens} token/cặp)\n"
                 f"- server: TEI {info['version']}, dtype {info['model_dtype']}, "
                 f"max_batch_tokens {info['max_batch_tokens']}\n"
                 f"- {args.runs} run x {args.duration_s:g}s mỗi mức, lấy median\n\n"
                 f"{qps_table}\n\n{cap_line}\n{len_table}\n\n"
                 f"## Bối cảnh lúc đo\n\n```\n"
                 f"{json.dumps(environment(), indent=2, ensure_ascii=False)}\n```\n")


# --- quality ----------------------------------------------------------------

def load_eval(path):
    """JSONL: {"query": str, "docs": [str], "labels": [int] (tuỳ chọn)}."""
    items = []
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        d = json.loads(line)
        for key in ("query", "docs"):
            if key not in d:
                raise SystemExit(f"{path}:{ln} thiếu khoá {key!r}")
        if "labels" in d and len(d["labels"]) != len(d["docs"]):
            raise SystemExit(f"{path}:{ln} labels và docs lệch độ dài")
        items.append(d)
    if not items:
        raise SystemExit(f"{path}: không có dòng nào")
    return items


def cmd_quality(args):
    opener = opener_no_proxy()
    info = fetch_info(opener, args.url)
    items = load_eval(args.eval)
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))["queries"] \
        if args.baseline else None
    if base is not None and len(base) != len(items):
        raise SystemExit("baseline có số query khác bộ eval - không so được")

    labelled = all("labels" in it for it in items)
    print(f"bộ eval: {args.eval} | {len(items)} query | "
          f"{'CÓ nhãn' if labelled else 'KHÔNG có nhãn'}")
    print(f"server : TEI {info['version']} | dtype {info['model_dtype']}")
    if args.baseline:
        print(f"so với : {args.baseline}")
    if not labelled and not args.baseline:
        print("\nKhông có nhãn và không có baseline -> chỉ lưu điểm để làm mốc về sau.")

    per_query, saved = [], []
    for i, it in enumerate(items):
        scores = rerank(opener, args.url, it["query"], it["docs"])
        saved.append({"query": it["query"], "scores": scores})
        row = {}
        if labelled:
            order = sorted(range(len(scores)), key=lambda j: -scores[j])
            ranked = [it["labels"][j] for j in order]
            row.update(ndcg=ndcg_at_k(ranked, args.k_at),
                       mrr=mrr_at_k(ranked, args.k_at),
                       recall=recall_at_k(ranked, args.k_at))
        if base is not None:
            b = base[i]["scores"]
            row.update(tau=kendall_tau(b, scores),
                       overlap=topk_overlap(b, scores, min(args.k_at, len(scores))),
                       swaps=float(significant_swaps(b, scores)))
        if row:
            per_query.append(row)

    lines = []
    if per_query:
        summary = quality_summary(per_query, args.k_at)
        width = max(len(k) for k in summary)
        for key, val in summary.items():
            lines.append(f"  {key:<{width}}  {val:.4f}")
        print("\n" + "\n".join(lines))
        if base is not None:
            print(f"\nswaps = số cặp doc đảo chỗ mà chênh lệch vượt ngưỡng nhiễu "
                  f"{TIE_EPS}. Bằng 0 = thứ hạng không đổi ngoài nhiễu.")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text(json.dumps(
        {"config": {k: info[k] for k in ("version", "model_dtype", "max_batch_tokens")},
         "queries": saved}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nđã lưu điểm vào {args.save}")
    print(f"lần sau so với mốc này:  --baseline {args.save}")

    if per_query:
        write_report(args.out,
                     f"# Bench rerank - chất lượng - {time.strftime('%Y-%m-%d %H:%M')}\n\n"
                     f"- eval: {args.eval} ({len(items)} query, "
                     f"{'có nhãn' if labelled else 'không nhãn'})\n"
                     f"- server: TEI {info['version']}, dtype {info['model_dtype']}\n"
                     + (f"- baseline: {args.baseline}\n" if args.baseline else "")
                     + "\n```\n" + "\n".join(lines) + "\n```\n")


def main():
    ap = argparse.ArgumentParser(description="bench reranker")
    ap.add_argument("--url", default="http://127.0.0.1:9012", help="lab; 9002 là PROD")
    ap.add_argument("--metrics-url")
    ap.add_argument("--container", default="rerank-lab", help="để đọc bộ nhớ GPU thật")
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("perf", help="quét QPS và độ dài, tìm sức chứa tại SLO")
    p.add_argument("--qps", default="1,4,8,16,32")
    p.add_argument("--slo-ms", type=float, default=200.0, help="ngưỡng p99 để tính sức chứa")
    p.add_argument("--k", type=int, help="số doc mỗi request (mặc định 15)")
    p.add_argument("--words", type=int, default=120)
    p.add_argument("--file", help="dùng dữ liệu thật thay vì tự sinh")
    p.add_argument("--lengths", help="quét thêm độ dài, vd 30,60,120,240,480")
    p.add_argument("--length-qps", type=float, default=4.0)
    p.add_argument("--duration-s", type=float, default=20.0)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--workers", type=int, default=128)
    p.add_argument("--cooldown-s", type=float, default=5.0)
    p.add_argument("--out", help="mặc định: bench/result/rerank_perf_<ngày_giờ>.md")
    p.set_defaults(func=cmd_perf)

    q = sub.add_parser("quality", help="NDCG/MRR/Recall nếu có nhãn, hoặc so thứ hạng với baseline")
    q.add_argument("--eval", required=True, help="file JSONL")
    q.add_argument("--k-at", type=int, default=10, help="k trong NDCG@k / MRR@k / Recall@k")
    q.add_argument("--baseline", help="file điểm của lần đo trước, để so thứ hạng")
    q.add_argument("--save", help="mặc định: bench/result/rerank_scores_<ngày_giờ>.json")
    q.add_argument("--out", help="mặc định: bench/result/rerank_quality_<ngày_giờ>.md")
    q.set_defaults(func=cmd_quality)

    args = ap.parse_args()
    args.metrics_url = args.metrics_url or args.url + "/metrics"
    args.out = args.out or default_out(args.mode)
    if args.mode == "quality":
        args.save = args.save or default_out("scores", "json")
    if args.mode == "perf":
        args.qps_levels = sorted({float(x) for x in args.qps.split(",")})
        args.lengths = sorted({int(x) for x in args.lengths.split(",")}) if args.lengths else None

    # Phải tính TRƯỚC khi clear, và chỉ bỏ biến proxy: clear rồi update rỗng sẽ
    # xoá cả PATH, làm docker/nvidia-smi không chạy nổi (bẫy #3 ở dạng khác).
    clean = without_proxy_env(os.environ)
    os.environ.clear()
    os.environ.update(clean)
    args.func(args)


if __name__ == "__main__":
    main()
