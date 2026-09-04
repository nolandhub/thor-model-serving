# ABOUTME: Đo encoder.onnx chạy batch 1 so với batch N - trần lý thuyết mà BLS có thể mua
# ABOUTME: Độc lập hoàn toàn với Triton và model.py, không đụng gì vào đường phục vụ thật

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def batch_axis(shape, symbol):
    """Vị trí chiều batch trong shape, tìm theo TÊN ký hiệu động.

    onnxruntime trả chiều động dưới dạng chuỗi ('N', 'batch'...), và mọi input
    dùng chung một chiều batch thì mang cùng một tên. Đối chiếu tên là cách duy
    nhất đúng với mọi bản export: đoán "batch luôn ở axis 0" sẽ nhân bản nhầm
    chiều của state zipformer (num_layers nằm trước batch), và đo ra một con số
    trông hợp lý nhưng của một phép tính hoàn toàn khác.
    """
    for i, dim in enumerate(shape):
        if isinstance(dim, str) and dim == symbol:
            return i
    return None


def tile_state(state, axis, n):
    """Nhân bản state cho batch n. axis None = input không phụ thuộc batch."""
    if axis is None:
        return state
    return np.concatenate([state] * n, axis=axis)


def per_sample_ms(times_s, batch):
    """Trung vị thời gian mỗi mẫu, tính bằng ms - đơn vị so được giữa các batch.

    Trung vị chứ không trung bình: lần chạy đầu của mỗi cỡ batch luôn phải cấp
    bộ nhớ và chọn thuật toán conv, đắt hơn hẳn phần còn lại.
    """
    if not times_s:
        raise ValueError("no run measured yet")
    return statistics.median(times_s) / batch * 1000


def state_bytes(specs, symbol, batch):
    """Tổng byte của state encoder cho một stream. specs = [(shape, dtype)].

    Đây là thước đo bls_tax dựng trước khi viết BLS: tách encoder thành model
    riêng thì toàn bộ cache này phải serialize qua ranh giới model MỖI chunk
    MỖI stream, hai lượt vào và ra. State nặng thì thuế ăn hết phần thắng nhờ
    batch, và biết trước rẻ hơn nhiều so với biết sau khi đã refactor xong.

    Chiều động không phải batch tính là 1 - không có gì để suy ra độ dài thật,
    nên con số trả về là CẬN DƯỚI, chỗ in ra phải nói rõ điều đó.
    """
    total = 0
    for shape, dtype in specs:
        n = 1
        for dim in shape:
            if isinstance(dim, str):
                n *= batch if dim == symbol else 1
            else:
                n *= dim
        total += n * np.dtype(dtype).itemsize
    return total


def build_feeds(session, batch, symbol):
    """Feed đầy đủ cho một lần chạy encoder ở cỡ batch cho trước.

    State toàn số 0 chứ không phải state thật: bài này đo THỜI GIAN, mà thời
    gian của conv/matmul không phụ thuộc giá trị. Dựng state thật thì phải kéo
    theo cả fbank lẫn vòng streaming, tức là đo lại chính cái mình đang muốn
    tách khỏi phép đo.
    """
    feeds = {}
    for inp in session.get_inputs():
        dtype = _ORT_TO_NP[inp.type]
        axis = batch_axis(inp.shape, symbol)
        shape = [1 if isinstance(d, str) else d for d in inp.shape]
        if axis is not None:
            shape[axis] = batch
        feeds[inp.name] = np.zeros(shape, dtype=dtype)
    return feeds


def time_batch(session, batch, symbol, iters, warmup):
    feeds = build_feeds(session, batch, symbol)
    for _ in range(warmup):
        session.run(None, feeds)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, feeds)
        times.append(time.perf_counter() - t0)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=str(ROOT / "model_repository/asr_bls/1/encoder.onnx"))
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--provider", default="CUDAExecutionProvider")
    args = ap.parse_args()

    import onnxruntime as ort

    session = ort.InferenceSession(args.encoder, providers=[args.provider, "CPUExecutionProvider"])
    print(f"provider: {session.get_providers()[0]}")

    x = session.get_inputs()[0]
    symbol = x.shape[0] if isinstance(x.shape[0], str) else None
    if symbol is None:
        raise SystemExit(
            f"batch dim of {x.name} is {x.shape[0]} (fixed) - this export cannot batch, "
            "which is already the answer: BLS has nothing to gain"
        )
    print(f"input {x.name} {x.shape}, batch symbol {symbol!r}")

    specs = [(i.shape, _ORT_TO_NP[i.type]) for i in session.get_inputs()[1:]]
    if specs:
        per_stream = state_bytes(specs, symbol, 1)
        print(f"state: {len(specs)} tensors, {per_stream / 1024:.1f} KiB per stream "
              f"(lower bound) - BLS moves this across the model boundary twice per chunk")

    base = None
    print("\n| batch | ms/run | ms/sample | speedup vs batch 1 |")
    print("|---|---|---|---|")
    for batch in [int(b) for b in args.batches.split(",")]:
        times = time_batch(session, batch, symbol, args.iters, args.warmup)
        per_sample = per_sample_ms(times, batch)
        base = base or per_sample
        print(f"| {batch} | {statistics.median(times) * 1000:.2f} | {per_sample:.3f} "
              f"| {base / per_sample:.2f}x |")

    print(
        "\nLast column is the CEILING BLS can buy: batching N chunks into one encoder "
        "call is that much faster than N separate calls. Subtract bls_tax (tensor-copy "
        "cost across the model boundary, currently 0.00-0.01) to get the real gain."
    )


if __name__ == "__main__":
    main()
