#!/usr/bin/env python3
# ABOUTME: So hai file fixture rerank - độ dài doc quyết định latency qua phần pad
# ABOUTME: Chạy: python3 scripts/compare_fixture.py a.json [b.json]

"""Vì sao độ dài doc, chứ không phải số doc, mới là thứ cần so.

Backend không có flash-attn nên pad MỌI sequence trong batch lên bằng cái dài
nhất batch. Một doc 2000 token lọt vào batch 20 doc thì 19 doc kia cũng bị tính
như 2000 token. Nên cái quyết định latency không phải độ dài TRUNG BÌNH mà là
độ dài LỚN NHẤT rơi vào mỗi batch - và với batch ~20 lấy ngẫu nhiên, đó là kỳ
vọng của max trên 20 lần rút.

Ước lượng token bằng 1.11 token/từ, đúng hằng số bench đang dùng. Số token THẬT
phải hỏi /tokenize (gen_rerank_docs.py --verify-url); ở đây chỉ cần so hai bộ
với nhau nên sai số hệ thống triệt tiêu.
"""

import json
import statistics
import sys

TOKENS_PER_WORD = 1.11
MAX_INPUT_LENGTH = 8192      # trần của model, hỏi từ /info


def load(path):
    raw = json.load(open(path, encoding="utf-8"))
    return raw if isinstance(raw, list) else [raw]


def exp_max_of_n(values, n, trials=2000, seed=0):
    """Kỳ vọng của max trên n lần rút - xấp xỉ độ dài pad của một batch n doc."""
    import random
    rnd = random.Random(seed)
    return statistics.median(max(rnd.choices(values, k=n)) for _ in range(trials))


def report(path):
    items = load(path)
    toks, per_query_max, n_texts = [], [], []
    for it in items:
        t = [round(len(x.split()) * TOKENS_PER_WORD) for x in it["texts"]]
        toks += t
        per_query_max.append(max(t))
        n_texts.append(len(t))

    q = sorted(toks)
    print(f"\n=== {path}")
    print(f"  queries              : {len(items)}")
    print(f"  texts / query        : min {min(n_texts)}  max {max(n_texts)}")
    print(f"  total docs           : {len(toks)}")
    print(f"  token/doc  p50       : {statistics.median(q):.0f}")
    print(f"             p95       : {q[int(0.95 * (len(q) - 1))]:.0f}")
    print(f"             p99       : {q[int(0.99 * (len(q) - 1))]:.0f}")
    print(f"             max       : {max(q)}")
    print(f"  token/doc  mean      : {statistics.mean(q):.0f}")
    print(f"  STDDEV               : {statistics.pstdev(q):.0f}   (higher = more padding waste)")
    print()
    print(f"  effective pad of a 20-batch (expected max-of-20): {exp_max_of_n(q, 20):.0f} tokens")
    print(f"  -> padding waste = {exp_max_of_n(q, 20) / statistics.mean(q):.2f}x vs "
          "no padding at all")
    over = sum(1 for t in q if t > MAX_INPUT_LENGTH)
    if over:
        print(f"  !! {over} docs exceed max_input_length={MAX_INPUT_LENGTH} "
              "with auto_truncate=false -> server will ERROR, not just run slow")
    return {"mean": statistics.mean(q), "pad20": exp_max_of_n(q, 20)}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__ + "\nNeed at least 1 file.")
    stats = [report(p) for p in sys.argv[1:]]
    if len(stats) == 2:
        a, b = stats
        print(f"\n=== predicted infer ratio ({sys.argv[2]} vs {sys.argv[1]})")
        print(f"  by mean tokens     : {b['mean'] / a['mean']:.2f}x")
        print(f"  by real pad (max-20): {b['pad20'] / a['pad20']:.2f}x  <- the number to trust")


if __name__ == "__main__":
    main()
