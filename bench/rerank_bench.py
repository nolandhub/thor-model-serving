#!/usr/bin/env python3
# ABOUTME: Bảng độ trễ + system reranker theo top-k - đúng định dạng bảng của mentor, có thêm system
# ABOUTME: Chạy: python3 bench/rerank_bench.py --url http://127.0.0.1:9012 --topk 10,20,50,100

"""Closed-loop: gửi -> chờ -> gửi tiếp, mỗi mức top-k đo riêng.

Đây là độ trễ một request nhìn từ client khi server rảnh, không phải sức chứa.
Server chậm thì client tự gửi thưa ra, nên bảng LATENCY dưới đây KHÔNG thấy
được hàng đợi dồn - muốn thấy phải chạy open-loop bằng bench/run_rerank.py perf.

Bảng SYSTEM đo CÙNG một cửa sổ thời gian với bảng latency (không phải một lượt
chạy riêng): "vì sao chậm" phải đứng cạnh "chậm bao nhiêu", đo lệch nhau thì
không đối chiếu được. Theo đúng yêu cầu của mentor: đánh giá benchmark phải có
cả latency lẫn system, không chỉ mỗi độ trễ nhìn từ client.

stdout chỉ có bảng. Mọi thứ khác (tiến độ, cảnh báo) đi ra stderr để
`... > bang.md` ra file dùng được ngay.
"""

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench.report import diff_counters, pct  # noqa: E402
from bench.tei_metrics import server_side, snapshot_http, unaccounted_ms  # noqa: E402
# ClockSampler đọc /sys/class/devfreq của Thor, không dính gì tới ASR - import
# lại chứ không chép, giống cách bench/run_rerank.py (gốc) đã làm.
from bench.run_asr import ClockSampler, without_proxy_env  # noqa: E402
from rerank_probe import QUERY, make_docs  # noqa: E402

SAMPLE_S = 1.0

LAT_COLUMNS = ("Top-k", "Mean (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)")
SYS_COLUMNS = (
    "Top-k", "pairs/req", "avg_batch", "queue ms", "infer ms", "tokenize ms",
    "overhead ms", "GPU busy%", "GPU MiB", "GPC MHz",
)


def opener_no_proxy():
    """Docker tiêm proxy công ty vào mọi container; urllib đọc nó và gửi request
    tới gateway thay vì tới server. Bẫy #3 trong bench/README."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def rerank(opener, url, query, docs, timeout=600):
    body = json.dumps({"query": query, "texts": docs, "raw_scores": True}).encode()
    req = urllib.request.Request(url + "/rerank", data=body,
                                 headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as resp:
        resp.read()


# --- system: GPU -------------------------------------------------------

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
            "gpu_busy_pct": statistics.mean(self.utils) if self.utils else 0.0,
            "gpu_mib_peak": max(self.mibs) if self.mibs else 0.0,
        }


# --- đo: cùng một cửa sổ cho cả latency lẫn system ----------------------

def measure(opener, args, queries, k):
    """Độ trễ từng request (giây) + system đo CÙNG cửa sổ. Warmup bị bỏ khỏi cả
    hai, không chỉ khỏi latency: nạp model / dựng CUDA context / GPU ramp từ
    315MHz rơi vào warmup thì cả hai bảng cùng sạch (bẫy #1 của bench ASR).

    `queries` xoay vòng qua NHIỀU (query, docs) nếu file là multi-query - mỗi
    lần trong --repeat là một câu khác nhau, sát traffic thật hơn lặp một câu.
    File một-query hoặc chế độ tự sinh chỉ có 1 phần tử nên coi như lặp cố định
    - hành vi cũ giữ nguyên, không có gì đổi cho hai chế độ đó.
    """
    def one(i):
        q, docs = sample(args, queries, k, i)
        rerank(opener, args.url, q, docs)

    for i in range(args.warmup):
        one(i)

    before = snapshot_http(args.metrics_url)
    clock, hw = ClockSampler(), HwSampler(args.container)
    clock.start(), hw.start()

    lat = []
    for i in range(args.repeat):
        t0 = time.perf_counter()
        one(i)
        lat.append(time.perf_counter() - t0)

    hw_stats, clock_stats = hw.stop(), clock.stop()
    after = snapshot_http(args.metrics_url)

    sys_stats = server_side(diff_counters(before, after))
    sys_stats.update(hw_stats)
    sys_stats["gpu_mhz_p50"] = clock_stats["p50"]
    return lat, sys_stats


def lat_row(k, lat):
    ms = [v * 1000 for v in lat]
    return (k, statistics.mean(ms), pct(ms, 50), pct(ms, 95), pct(ms, 99))


def sys_row(k, s, mean_ms):
    """Hàng bảng system. Cần `mean_ms` của chính cửa sổ đo đó cho cột overhead.

    Hai cột thêm về sau, mỗi cột chữa một điểm mù:

    - `pairs/req` nói ngữ nghĩa của các cột thời gian. Ra 100 ở top-k 100 =
      te_request_count đếm lời gọi HTTP, nên queue/infer/tokenize là TỔNG cho
      cả request. Ra 1 = counter đếm từng cặp và mọi cột phải đọc lại từ đầu.
      Số này tính sẵn trong server_side() từ lâu mà chưa bao giờ được in.
    - `overhead` là phần client thấy mà server không nhận - xem unaccounted_ms.
    """
    return (k, s["pairs_per_request"], s["avg_batch"],
            s["queue_ms_per_req"], s["infer_ms_per_req"], s["tokenize_ms_per_req"],
            unaccounted_ms(mean_ms, s),
            s["gpu_busy_pct"], s["gpu_mib_peak"], s["gpu_mhz_p50"])


def table(columns, rows, fmt="{:.0f}"):
    """Bảng markdown. Cột đầu in thô, các cột sau theo `fmt`.

    `fmt` nhận một chuỗi dùng cho mọi cột, HOẶC một dãy fmt cho từng cột. Bảng
    open-loop cần dạng dãy: latency tính bằng ms nên "{:.0f}" là đủ, còn lag
    tính bằng giây - cùng một fmt sẽ làm tròn 0.05 thành 0.1 và xoá đúng cái
    tín hiệu bài bench sinh ra để nhìn.
    """
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "---|" * len(columns)
    body = []
    for k, *stats in rows:
        fmts = [fmt] * len(stats) if isinstance(fmt, str) else list(fmt)
        body.append(f"| {k} | " + " | ".join(f.format(v) for f, v in zip(fmts, stats)) + " |")
    return "\n".join([head, sep, *body])


TOKENS_PER_WORD = 1.11   # đo trên chính tokenizer của model; chỉ để ước lượng


def truncate(docs, max_tokens):
    """Cắt doc về trần token, XẤP XỈ qua số từ.

    Cắt đúng từng token thì phải hỏi /tokenize rồi ghép ngược lại, mà TEI không
    có endpoint detokenize. Sai số của quy đổi 1.11 token/từ là vài phần trăm,
    đủ để dò ngưỡng; con số token THẬT của bộ đã cắt lấy bằng
    gen_rerank_docs.py --verify-url, đừng suy từ hằng số này.
    """
    n = max(1, int(max_tokens / TOKENS_PER_WORD))
    return [" ".join(t.split()[:n]) for t in docs]


def load_queries(path):
    """Đọc file bench - CHẤP NHẬN HAI HÌNH:

        {"query": ..., "texts": [...]}            một query
        [{"query": ..., "texts": [...]}, ...]      nhiều query

    Hình sau là dữ liệu multi-query thật (vd 1000 câu hỏi khác nhau); trả về
    danh sách để measure() xoay vòng qua, thay vì lặp y hệt một câu suốt
    --repeat lần. Không kiểm trùng ở đây - đó là việc lúc SINH dữ liệu
    (gen_rerank_docs.py), không phải lúc ĐỌC để bench.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else [raw]
    out = []
    for i, d in enumerate(items):
        for key in ("query", "texts"):
            if key not in d:
                raise SystemExit(f"{path}[{i}]: missing key {key!r}")
        out.append((d["query"], d["texts"]))
    if not out:
        raise SystemExit(f"{path}: empty - no queries")
    return out


def prep_docs(args, k, texts):
    """k đoạn đầu của `texts`, rồi áp cắt/sắp nếu có cờ.

    Thứ tự áp dụng là cắt TRƯỚC rồi sắp SAU, giống hệt thứ tự mà client thật
    phải làm: cắt đổi độ dài nên sắp trước là sắp theo độ dài đã chết.
    """
    docs = list(texts[:k])
    if len(docs) < k:
        raise SystemExit(f"top-k {k} requested but this query has only {len(docs)} docs")
    if args.max_doc_tokens:
        docs = truncate(docs, args.max_doc_tokens)
    if args.sort_by_length:
        # Gom doc dài gần nhau vào cùng batch để phần pad co lại. Backend không
        # có flash-attn nên pad mọi sequence lên bằng sequence dài nhất của
        # batch - đo được 1.49x chỉ nhờ đổi thứ tự. TEI trả về index của doc
        # gốc nên thứ hạng không hề đổi; đây thuần tuý là tầng vận chuyển.
        docs = sorted(docs, key=len)
    return docs


def sample(args, queries, k, i):
    """(query, docs) cho lần lặp thứ i. Xoay vòng theo modulo nếu --repeat lớn
    hơn số query có sẵn - lặp lại query cũ còn hơn dừng bench giữa chừng."""
    if queries is not None:
        q, texts = queries[i % len(queries)]
    else:
        q, texts = QUERY, make_docs(k, args.words)
    return q, prep_docs(args, k, texts)


def main():
    ap = argparse.ArgumentParser(description="reranker latency + system tables by top-k")
    ap.add_argument("--url", default="http://127.0.0.1:9012", help="9012 = lab, 9002 = PROD")
    ap.add_argument("--metrics-url", help="defaults to --url + /metrics")
    ap.add_argument("--container", default="rerank-lab",
                    help="container name - reads real GPU busy%%/MiB via nvidia-smi")
    ap.add_argument("--topk", default="10,20,50,100")
    ap.add_argument("--repeat", type=int, default=50,
                    help="samples per level; nearest-rank, so p99 needs >=100 samples to differ from max")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--words", type=int, default=120, help="words per doc when SYNTHESIZING")
    ap.add_argument("--file", help="use real data instead of synthetic")
    ap.add_argument("--max-doc-tokens", type=int,
                    help="truncate each doc to token cap before sending (e.g. 384)")
    ap.add_argument("--sort-by-length", action="store_true",
                    help="sort docs by length to cut padding; ranking unchanged")
    args = ap.parse_args()
    args.metrics_url = args.metrics_url or args.url + "/metrics"

    # docker/nvidia-smi phải chạy KHÔNG proxy - Docker tiêm proxy công ty vào
    # mọi container lúc tạo, và biến đó rò ra cả tiến trình host gọi docker CLI
    # trong một số cấu hình. Cùng bẫy #3, dạng khác.
    import os
    clean = without_proxy_env(os.environ)
    os.environ.clear()
    os.environ.update(clean)

    # Giữ NGUYÊN thứ tự gõ vào, chỉ khử trùng lặp. Trước đây sorted() nên
    # `--topk 100,50,20,10` chạy y hệt `--topk 10,20,50,100` - thứ tự bị nuốt
    # mất lặng lẽ.
    #
    # Thứ tự ở đây không phải chuyện trình bày, nó là một núm thí nghiệm: bench
    # chạy hết mức này rồi mới sang mức kia, nên "mức k nào" trộn lẫn với "lúc
    # nào trong run", và mọi thứ trôi theo thời gian (nhiệt, team khác vào máy)
    # đều đội lốt thành hiệu ứng của top_k. Thor lại KHÔNG đọc được throttle
    # reason (bẫy #5) nên không có cách nào phát hiện sau khi đã đo.
    #
    # Chạy hai lượt xuôi và ngược rồi so hai đường cong là đo được chính hiệu
    # ứng đó: trùng nhau = bảng blocked tin được; lệch nhau = phải interleave.
    levels = list(dict.fromkeys(int(x) for x in args.topk.split(",")))
    queries = load_queries(args.file) if args.file else None
    if queries is not None:
        print(f"  {len(queries)} queries from {args.file}"
              + (" (round-robin)" if len(queries) > 1 else ""), file=sys.stderr)
    opener = opener_no_proxy()

    lat_rows, sys_rows = [], []
    for k in levels:
        print(f"  top-k {k}", file=sys.stderr)
        lat, s = measure(opener, args, queries, k)
        lat_rows.append(lat_row(k, lat))
        # Tính lại mean từ chính dãy lat chứ không bốc theo chỉ số từ lat_row:
        # bốc theo chỉ số sẽ vỡ IM LẶNG nếu sau này có ai chèn cột vào
        # LAT_COLUMNS, và cột overhead sẽ sai mà bảng vẫn in ra đẹp đẽ.
        sys_rows.append(sys_row(k, s, statistics.mean(lat) * 1000))

    print(table(LAT_COLUMNS, lat_rows))
    print()
    print(table(SYS_COLUMNS, sys_rows, fmt="{:.1f}"))


if __name__ == "__main__":
    main()
