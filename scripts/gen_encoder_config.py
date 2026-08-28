# ABOUTME: Sinh config.pbtxt cho model encoder từ encoder.onnx - 74 khối state không gõ tay được
# ABOUTME: Chạy: python scripts/gen_encoder_config.py > model_repository/encoder/config.pbtxt

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.encoder_batch import batch_axis  # noqa: E402

_ORT_TO_TRITON = {
    "tensor(float)": "TYPE_FP32",
    "tensor(float16)": "TYPE_FP16",
    "tensor(int64)": "TYPE_INT64",
    "tensor(int32)": "TYPE_INT32",
}


def state_pairs(input_names, output_names):
    """Ghép state vào/ra THEO VỊ TRÍ, đúng bất biến mà model.py đang dựa vào.

    Không ghép theo tên: tên vào và tên ra khác nhau (cached_x vs new_cached_x),
    và quy ước đặt tên đổi theo bản export. Vị trí thì không đổi - model.py hiện
    tại đã khẳng định điều đó bằng assert.
    """
    if len(input_names) != len(output_names):
        raise ValueError(
            f"số state vào ({len(input_names)}) lệch số state ra ({len(output_names)})"
        )
    return list(zip(input_names, output_names))


def triton_dims(shape, batch_ax):
    """Shape của ONNX -> dims của Triton: bỏ chiều batch, cấm mọi chiều động khác.

    max_batch_size > 0 thì Triton NGẦM hiểu chiều đầu là batch, dims chỉ khai
    phần còn lại. Và chiều batch bắt buộc đứng đầu - state có batch ở giữa thì
    không dùng được max_batch_size, phải hỏng ngay ở đây chứ không phải sau khi
    Triton nạp model rồi trả shape sai lúc chạy.
    """
    if batch_ax != 0:
        raise ValueError(
            f"chiều batch nằm ở axis {batch_ax}, Triton chỉ batch được khi nó ở axis 0"
        )
    dims = []
    for dim in shape[1:]:
        if isinstance(dim, str):
            raise ValueError(f"shape {shape} còn chiều động {dim!r} ngoài batch")
        dims.append(dim)
    if not dims:
        raise ValueError(
            f"shape {shape} bỏ chiều batch xong còn rỗng - Triton không nhận "
            "dims: [ ]. Nống thêm một chiều lúc bọc (xem scripts/wrap_encoder.py)"
        )
    return dims


def render_config(name, x_name, x_dims, x_type, states, max_batch, max_candidate,
                  max_queue_delay=0):
    """Dựng nội dung config.pbtxt. Thuần chuỗi, không đọc file nào."""
    blocks = []
    for s in states:
        dims = ", ".join(str(d) for d in s["dims"])
        blocks.append(
            f"""    {{
      input_name: "{s['input_name']}"
      output_name: "{s['output_name']}"
      data_type: {s['data_type']}
      dims: [ {dims} ]
      initial_state: {{
        data_type: {s['data_type']}
        dims: [ {dims} ]
        zero_data: true
        name: "zero"
      }}
    }}"""
        )
    x_dims_str = ", ".join(str(d) for d in x_dims)
    states_str = ",\n".join(blocks)
    return f"""# SINH TỰ ĐỘNG bởi scripts/gen_encoder_config.py - đừng sửa tay.
# 74 khối state ghép theo VỊ TRÍ giữa input[1:] và output[1:] của encoder.onnx.

name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch}

input [
  {{
    name: "{x_name}"
    data_type: {x_type}
    dims: [ {x_dims_str} ]
  }}
]

output [
  {{
    name: "encoder_out"
    data_type: {x_type}
    dims: [ -1, -1 ]
  }}
]

sequence_batching {{
  # oldest là chiến lược DUY NHẤT gom được request từ nhiều sequence khác nhau
  # vào cùng một batch. direct gán mỗi sequence một slot cố định, avg_batch sẽ
  # mãi là 1.0 và toàn bộ lý do tồn tại của model này biến mất.
  oldest {{
    max_candidate_sequences: {max_candidate}
    max_queue_delay_microseconds: {max_queue_delay}
  }}
  state [
{states_str}
  ]
}}

instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "model_repository/asr_streaming/1/encoder.onnx"))
    ap.add_argument("--name", default="encoder")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-candidate", type=int, default=8)
    ap.add_argument("--max-queue-delay", type=int, default=0,
                    help="µs chờ gom batch - tham số phải QUÉT, xem spec §6")
    args = ap.parse_args()

    import onnxruntime as ort

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inputs, outputs = sess.get_inputs(), sess.get_outputs()
    x = inputs[0]
    symbol = x.shape[0]
    if not isinstance(symbol, str):
        raise SystemExit(f"chiều batch của {x.name} cố định ({symbol}) - model này không batch được")

    pairs = state_pairs([i.name for i in inputs[1:]], [o.name for o in outputs[1:]])
    states = []
    for (in_name, out_name), spec in zip(pairs, inputs[1:]):
        states.append({
            "input_name": in_name,
            "output_name": out_name,
            "data_type": _ORT_TO_TRITON[spec.type],
            "dims": triton_dims(spec.shape, batch_axis(spec.shape, symbol)),
        })

    print(render_config(
        name=args.name,
        x_name=x.name,
        x_dims=triton_dims(x.shape, batch_axis(x.shape, symbol)),
        x_type=_ORT_TO_TRITON[x.type],
        states=states,
        max_batch=args.max_batch,
        max_candidate=args.max_candidate,
        max_queue_delay=args.max_queue_delay,
    ), end="")


if __name__ == "__main__":
    main()
