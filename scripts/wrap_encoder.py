# ABOUTME: Bọc encoder.onnx để mọi state có chiều batch ở axis 0 - điều kiện để Triton batch được
# ABOUTME: Chạy: python scripts/wrap_encoder.py --out model_repository/encoder/1/model.onnx

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.encoder_batch import batch_axis  # noqa: E402


def _check(rank, batch_ax):
    if not 0 <= batch_ax < rank:
        raise ValueError(f"batch_ax {batch_ax} ngoài rank {rank}")


def to_batch_first_perm(rank, batch_ax):
    """perm cho Transpose: layout GỐC -> batch-first. Dùng ở phía OUTPUT.

    Đồ thị cũ nhả state theo layout gốc (time-major với zipformer); Triton cần
    batch ở axis 0 mới ghép được nhiều sequence vào một batch.
    """
    _check(rank, batch_ax)
    return [batch_ax] + [i for i in range(rank) if i != batch_ax]


def from_batch_first_perm(rank, batch_ax):
    """perm cho Transpose: batch-first -> layout GỐC. Dùng ở phía INPUT.

    Nghịch đảo của to_batch_first_perm. Lẫn hai hàm này thì model vẫn nạp
    được, vẫn chạy ra số, và sai âm thầm - chỉ parity test bắt được.
    """
    fwd = to_batch_first_perm(rank, batch_ax)
    inv = [0] * rank
    for new_axis, old_axis in enumerate(fwd):
        inv[old_axis] = new_axis
    return inv


def needs_trailing_dim(shape, batch_ax):
    """Bỏ chiều batch ra thì mỗi mẫu còn lại là vô hướng?

    Triton không nhận `dims: [ ]`, nên state kiểu này (processed_lens có shape
    ['N']) phải được nống thành ['N', 1] ngay trong đồ thị. Làm ở đây chứ
    không ở config: config chỉ mô tả, sửa nó mà đồ thị không đổi thì Triton
    nạp xong rồi báo lệch shape lúc chạy.
    """
    _check(len(shape), batch_ax)
    return len(shape) == 1


def batch_first_shape(shape, batch_ax):
    """Shape sau khi đưa chiều batch lên đầu."""
    _check(len(shape), batch_ax)
    return [shape[batch_ax]] + [d for i, d in enumerate(shape) if i != batch_ax]


_AXIS_INIT = "wrap_axis_1"


def _axis_const(graph, value):
    """Tensor hằng chứa axis cho Squeeze/Unsqueeze (opset 13+ nhận axes là input).

    Dùng chung một initializer cho mọi chỗ - tạo mới mỗi lần là đồ thị phình
    thêm một initializer cho mỗi state mà giá trị y hệt nhau.
    """
    from onnx import helper, numpy_helper
    import numpy as np

    if not any(i.name == _AXIS_INIT for i in graph.initializer):
        graph.initializer.append(
            numpy_helper.from_array(np.array([value], dtype=np.int64), _AXIS_INIT)
        )
    del helper
    return _AXIS_INIT


def wrap(model, symbol):
    """Thêm Transpose ở hai đầu mọi state có batch không ở axis 0.

    Không đụng vào phần tính toán: chỉ chèn Transpose giữa ranh giới đồ thị và
    thân cũ, nên trọng số và thứ tự phép tính giữ nguyên tuyệt đối. x và
    encoder_out đã batch-first sẵn, không bọc.

    Trả về (model đã bọc, số state đã xoay).
    """
    import onnx
    from onnx import helper

    graph = model.graph
    # ONNX bắt node phải sắp thứ tự tô-pô. Transpose ở đầu vào phải đứng TRƯỚC
    # mọi node cũ, Transpose ở đầu ra phải đứng SAU - nối hết vào cuối thì
    # checker từ chối với "not output of any previous nodes".
    head, tail = [], []
    rotated = 0

    def dims_of(vi):
        return [
            d.dim_param if d.HasField("dim_param") else d.dim_value
            for d in vi.type.tensor_type.shape.dim
        ]

    # --- input: Triton đưa batch-first, xoay VỀ gốc trước khi vào thân cũ ---
    for vi in list(graph.input)[1:]:
        dims = dims_of(vi)
        ax = batch_axis(dims, symbol)
        if ax is None:
            raise ValueError(f"input {vi.name!r} không có chiều batch {symbol!r}")
        if ax == 0 and not needs_trailing_dim(dims, ax):
            continue
        inner = f"{vi.name}__inner"
        # Đổi TÊN của consumer: thân cũ đọc tensor tên inner, còn tên gốc giờ
        # là cổng vào batch-first. Giữ nguyên tên cổng để config.pbtxt và
        # model.py không phải biết gì về việc bọc này.
        for node in graph.node:
            for i, name in enumerate(node.input):
                if name == vi.name:
                    node.input[i] = inner
        if needs_trailing_dim(dims, ax):
            # ['N'] -> cổng vào ['N', 1], Squeeze bỏ chiều đệm cho thân cũ
            head.append(
                helper.make_node(
                    "Squeeze", [vi.name, _axis_const(graph, 1)], [inner],
                    name=f"wrap_in_{vi.name}",
                )
            )
            new_dims = [dims[0], 1]
        else:
            head.append(
                helper.make_node(
                    "Transpose", [vi.name], [inner],
                    perm=from_batch_first_perm(len(dims), ax),
                    name=f"wrap_in_{vi.name}",
                )
            )
            new_dims = batch_first_shape(dims, ax)
        del vi.type.tensor_type.shape.dim[:]
        vi.type.tensor_type.shape.dim.extend(
            helper.make_tensor_value_info("_", 1, new_dims).type.tensor_type.shape.dim
        )
        rotated += 1

    # --- output: thân cũ nhả layout gốc, xoay THÀNH batch-first ---
    for vi in list(graph.output)[1:]:
        dims = dims_of(vi)
        ax = batch_axis(dims, symbol)
        if ax is None:
            raise ValueError(f"output {vi.name!r} không có chiều batch {symbol!r}")
        if ax == 0 and not needs_trailing_dim(dims, ax):
            continue
        inner = f"{vi.name}__inner"
        for node in graph.node:
            for i, name in enumerate(node.output):
                if name == vi.name:
                    node.output[i] = inner
        if needs_trailing_dim(dims, ax):
            tail.append(
                helper.make_node(
                    "Unsqueeze", [inner, _axis_const(graph, 1)], [vi.name],
                    name=f"wrap_out_{vi.name}",
                )
            )
            new_dims = [dims[0], 1]
        else:
            tail.append(
                helper.make_node(
                    "Transpose", [inner], [vi.name],
                    perm=to_batch_first_perm(len(dims), ax),
                    name=f"wrap_out_{vi.name}",
                )
            )
            new_dims = batch_first_shape(dims, ax)
        del vi.type.tensor_type.shape.dim[:]
        vi.type.tensor_type.shape.dim.extend(
            helper.make_tensor_value_info("_", 1, new_dims).type.tensor_type.shape.dim
        )

    body = list(graph.node)
    del graph.node[:]
    graph.node.extend(head + body + tail)
    onnx.checker.check_model(model, full_check=False)
    return model, rotated


def main():
    import onnx

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "model_repository/asr_streaming/1/encoder.onnx"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = onnx.load(args.onnx)
    symbol = None
    for d in model.graph.input[0].type.tensor_type.shape.dim:
        symbol = d.dim_param if d.HasField("dim_param") else None
        break
    if not symbol:
        raise SystemExit("chiều batch của input đầu tiên cố định - model này không batch được")

    model, rotated = wrap(model, symbol)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    print(f"đã xoay {rotated} state sang batch-first -> {out}")


if __name__ == "__main__":
    main()
