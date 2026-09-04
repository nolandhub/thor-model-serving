#!/usr/bin/env python3
# ABOUTME: Bắn thử vào reranker - xem điểm, đo độ trễ, đếm token thật của input
# ABOUTME: Closed-loop, chỉ để nghịch và dò knob; bench open-loop có drift là việc khác

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.report import pct  # nearest-rank, không nội suy - dùng chung với bench ASR

QUERY = "Thủ tục đăng ký hộ kinh doanh cá thể cần chuẩn bị những giấy tờ gì?"

# Câu mồi trộn nhiều mức liên quan để điểm trải rộng, dễ thấy khi thứ hạng đổi.
# CHỈ 8 câu: sinh k>8 thì doc i và doc i+8 TRÙNG KHÍT nhau. Đó là lý do nên
# dùng --file với dữ liệu thật; chế độ tự sinh chỉ để đo độ trễ và dò nhiễu.
SEEDS = [
    "Hồ sơ đăng ký hộ kinh doanh gồm giấy đề nghị đăng ký, bản sao căn cước công dân của chủ hộ và biên bản họp thành viên hộ gia đình.",
    "Giấy chứng nhận đăng ký hộ kinh doanh được cấp trong thời hạn ba ngày làm việc kể từ ngày nhận đủ hồ sơ hợp lệ.",
    "Chủ hộ kinh doanh nộp hồ sơ tại cơ quan đăng ký kinh doanh cấp huyện nơi đặt địa điểm kinh doanh.",
    "Người nộp thuế có nghĩa vụ kê khai và nộp thuế đúng thời hạn theo quy định của pháp luật về quản lý thuế.",
    "Doanh nghiệp phải lưu giữ sổ sách kế toán tại trụ sở chính trong thời hạn luật định.",
    "Cách nấu phở bò Hà Nội ngon thì phải ninh xương ống nhiều giờ để nước dùng vừa trong vừa ngọt.",
    "Đội tuyển bóng đá quốc gia sẽ tập trung vào tháng sau để chuẩn bị cho vòng loại giải khu vực.",
    "Thời tiết khu vực Nam Bộ ngày mai có mưa rào và dông vài nơi, nhiệt độ cao nhất khoảng ba mươi hai độ.",
]


def make_docs(k, words):
    """k đoạn văn, mỗi đoạn khoảng `words` từ, xoay vòng qua SEEDS."""
    docs = []
    for i in range(k):
        seed = SEEDS[i % len(SEEDS)]
        text = seed
        while len(text.split()) < words:
            text += " " + SEEDS[(i + len(text)) % len(SEEDS)]
        docs.append(" ".join(text.split()[:words]))
    return docs


def load_file(path, k):
    """Đọc {"query": ..., "texts": [...]} - cùng định dạng body của /rerank."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("query", "texts"):
        if key not in data:
            raise SystemExit("%s: missing key %r" % (path, key))
    docs = data["texts"]
    if k is not None:
        if k > len(docs):
            raise SystemExit("--k %d but file has only %d docs" % (k, len(docs)))
        docs = docs[:k]
    return data["query"], docs, data.get("raw_scores")


def clip(text, n=60):
    """Cắt để bảng khỏi vỡ - dấu … báo rằng đây là cắt KHI IN, không phải khi gửi."""
    return text if len(text) <= n else text[:n - 1] + "…"


def post(port, path, payload, timeout=120):
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (port, path),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def token_len(port, text):
    """Độ dài token thật, hỏi chính tokenizer của server. None nếu bản TEI không có."""
    try:
        out = post(port, "/tokenize", {"inputs": text})
        return len(out[0]) if out and isinstance(out[0], list) else None
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser(description="quick probe against the reranker")
    ap.add_argument("--port", type=int, default=9012, help="9012 = lab, 9002 = PROD (careful)")
    ap.add_argument("--file", help="read query+texts from a JSON file, e.g. scripts/hello.json")
    ap.add_argument("--k", type=int, default=None,
                    help="docs per request (synthetic: default 15; --file: first k docs)")
    ap.add_argument("--words", type=int, default=120, help="words per doc when SYNTHESIZING")
    ap.add_argument("--repeat", type=int, default=20, help="measured iterations (after warmup)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--show", action="store_true", help="print scores and ranking instead of measuring latency")
    ap.add_argument("--raw", dest="raw", action="store_true", default=None,
                    help="raw logit scores - noise-sensitive, use when comparing two configs")
    ap.add_argument("--no-raw", dest="raw", action="store_false",
                    help="0..1 scores - readable, easy to threshold")
    args = ap.parse_args()

    if args.file:
        query, docs, raw_from_file = load_file(args.file, args.k)
        source = args.file
    else:
        query = QUERY
        docs = make_docs(args.k if args.k is not None else 15, args.words)
        raw_from_file = None
        source = "synthetic (%d words/doc)" % args.words
        if len(set(docs)) < len(docs):
            source += "  [WARNING: %d/%d docs duplicated - only %d unique texts]" % (
                len(docs) - len(set(docs)), len(docs), len(set(docs)))

    # Ưu tiên cờ dòng lệnh, rồi tới khoá trong file, cuối cùng mặc định thô.
    raw = args.raw if args.raw is not None else (
        raw_from_file if raw_from_file is not None else True)

    n_tok = token_len(args.port, query + " " + docs[0])
    print("port %d | K=%d docs | %s | %s" % (
        args.port, len(docs),
        ("~%d tokens/pair" % n_tok) if n_tok else "(/tokenize unavailable)",
        "raw scores" if raw else "0..1 scores"))
    print("source: %s" % source)
    print()

    payload = {"query": query, "texts": docs, "raw_scores": raw}

    for _ in range(args.warmup):
        post(args.port, "/rerank", payload)

    if args.show:
        out = post(args.port, "/rerank", payload)
        print("%-6s %14s  %s" % ("doc", "score", "text (... = truncated for display only; model got it all)"))
        print("-" * 88)
        for r in sorted(out, key=lambda r: -r["score"]):
            print("%-6d %14.5f  %s" % (r["index"], r["score"], clip(docs[r["index"]])))
        return

    lat = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        post(args.port, "/rerank", payload)
        lat.append(time.perf_counter() - t0)

    print("%d iterations, %d pairs each" % (len(lat), len(docs)))
    print("-" * 46)
    for label, v in (("p50", pct(lat, 50)), ("p90", pct(lat, 90)),
                     ("p99", pct(lat, 99)), ("min", min(lat)), ("max", max(lat))):
        print("  %-4s %8.1f ms" % (label, v * 1000))
    print("  %-4s %8.1f ms" % ("mean", statistics.mean(lat) * 1000))
    print()
    print("  throughput: %.1f pairs/sec" % (len(docs) / statistics.mean(lat)))
    print()
    print("This is closed-loop: send -> wait -> send. A slow server makes the client")
    print("back off, so queue build-up is INVISIBLE here. Use an open-loop bench for that.")


if __name__ == "__main__":
    main()
