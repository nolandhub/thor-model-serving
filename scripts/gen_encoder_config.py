#!/usr/bin/env python3
# ABOUTME: Sinh model_repository/encoder/config.pbtxt từ encoder.onnx - đừng sửa config tay
# ABOUTME: Chạy: python3 scripts/gen_encoder_config.py <encoder.onnx> <đường dẫn config ra>

"""Sinh config cho encoder chạy backend onnxruntime với implicit state.

Vì sao max_batch_size: 0
------------------------
Trục batch của zipformer2 KHÔNG đồng nhất:

    cached_key_0    (128, N, 128)   -> batch ở trục 1
    cached_conv1_0  (N, 192, 15)    -> batch ở trục 0
    processed_lens  (N,)            -> batch ở trục 0, rank 1

Triton luôn chèn chiều batch vào trục 0 khi max_batch_size > 0, và không có cú
pháp nào khai "batch của tensor này ở trục 1". Nên với max_batch_size: 8 thì
Triton cấp phát cached_key_0 là (B,128,128) trong khi ONNX cần (128,B,128) -
sai ngay ở B=1, không phải chỉ sai khi batch lớn. Đây là lỗi của bản sinh config
trước, và nó đúng với cả explicit state chứ không riêng implicit.

max_batch_size: 0 thì dims là shape đầy đủ, khai được nguyên vẹn. Đánh đổi:
encoder chạy batch 1, throughput lấy từ instance_group count. Chấp nhận được vì
encoder chỉ được gọi 1 lần mỗi chunk trong khi joiner được gọi 8 lần.

Muốn batch thật thì phải phẫu thuật graph: chèn Transpose vào input/output của
các cache trục-1 để đưa hết về batch-first. Đó là task riêng.

Vì sao chạy thử một lượt inference
----------------------------------
Shape output trong ONNX toàn tên symbolic ('Slicenew_cached_key_0_dim_0'), không
đọc được số. Chạy một lượt với zeros là cách duy nhất biết chắc shape output
khớp shape input - điều kiện bắt buộc để implicit state hoạt động. Lệch một
tensor là Triton báo lỗi lúc load, hoặc tệ hơn là cache trôi dần.
"""

import argparse
import sys

import numpy as np
import onnxruntime as ort

# Phải khớp STATE_TTL_S trong model.py và max_sequence_idle_microseconds của
# asr_bls. Ba nơi lệch nhau là state bị giải phóng sớm ở một tầng: chunk kế tiếp
# tới không có SEQUENCE_START, Triton nạp lại initial_state zero, transcript sai
# hoàn toàn mà không có exception nào.
IDLE_US = 60_000_000
MAX_CANDIDATE_SEQUENCES = 64
INSTANCE_COUNT = 3

_NP_TO_TRITON = {
    np.dtype("float32"): "TYPE_FP32",
    np.dtype("float16"): "TYPE_FP16",
    np.dtype("int64"): "TYPE_INT64",
    np.dtype("int32"): "TYPE_INT32",
}
_ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def concrete(shape):
    """Shape ONNX -> shape thật với batch = 1. Chiều symbolic duy nhất là N."""
    return tuple(s if isinstance(s, int) else 1 for s in shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("out", nargs="?", default="-")
    args = ap.parse_args()

    o = ort.SessionOptions()
    o.log_severity_level = 3
    sess = ort.InferenceSession(args.onnx, o, providers=["CPUExecutionProvider"])

    ins, outs = sess.get_inputs(), sess.get_outputs()
    if len(ins) != len(outs):
        sys.exit(f"encoder có {len(ins)} input và {len(outs)} output - phải bằng nhau")
    n_state = len(ins) - 1

    # Chạy thật một lượt để lấy shape output bằng số
    feeds = {
        i.name: np.zeros(concrete(i.shape), dtype=_ORT_TO_NP[i.type]) for i in ins
    }
    res = sess.run(None, feeds)

    # Kiểm shape output state khớp input state theo VỊ TRÍ. Không khớp thì
    # implicit state không dùng được và phải dừng ở đây, chứ để Triton phát hiện
    # lúc load thì thông báo lỗi khó lần hơn nhiều.
    for k in range(n_state):
        want = concrete(ins[k + 1].shape)
        got = tuple(res[k + 1].shape)
        if want != got:
            sys.exit(
                f"state lệch shape: {ins[k + 1].name} vào {want} nhưng "
                f"{outs[k + 1].name} ra {got}"
            )

    lines = [
        "# SINH TỰ ĐỘNG bởi scripts/gen_encoder_config.py - đừng sửa tay.",
        f"# {n_state} khối state ghép theo VỊ TRÍ giữa input[1:] và output[1:].",
        "# max_batch_size: 0 vì trục batch không đồng nhất - xem docstring của script.",
        "",
        'name: "encoder"',
        'platform: "onnxruntime_onnx"',
        "max_batch_size: 0",
        "",
        "input [",
        "  {",
        f'    name: "{ins[0].name}"',
        f"    data_type: {_NP_TO_TRITON[np.dtype(_ORT_TO_NP[ins[0].type])]}",
        f"    dims: {list(concrete(ins[0].shape))}",
        "  }",
        "]",
        "",
        "output [",
        "  {",
        f'    name: "{outs[0].name}"',
        f"    data_type: {_NP_TO_TRITON[res[0].dtype]}",
        f"    dims: {list(res[0].shape)}",
        "  }",
        "]",
        "",
        "sequence_batching {",
        f"  max_sequence_idle_microseconds: {IDLE_US}",
        "  oldest {",
        f"    max_candidate_sequences: {MAX_CANDIDATE_SEQUENCES}",
        "    max_queue_delay_microseconds: 0",
        "  }",
        "  state [",
    ]

    blocks = []
    for k in range(n_state):
        name_in, name_out = ins[k + 1].name, outs[k + 1].name
        dims = list(concrete(ins[k + 1].shape))
        dtype = _NP_TO_TRITON[np.dtype(_ORT_TO_NP[ins[k + 1].type])]
        blocks.append(
            "    {\n"
            f'      input_name: "{name_in}"\n'
            f'      output_name: "{name_out}"\n'
            f"      data_type: {dtype}\n"
            f"      dims: {dims}\n"
            "      initial_state: {\n"
            f"        data_type: {dtype}\n"
            f"        dims: {dims}\n"
            "        zero_data: true\n"
            '        name: "zero"\n'
            "      }\n"
            "    }"
        )
    lines.append(",\n".join(blocks))
    lines += [
        "  ]",
        "}",
        "",
        "instance_group [",
        "  {",
        "    kind: KIND_GPU",
        f"    count: {INSTANCE_COUNT}",
        "  }",
        "]",
        "",
    ]

    text = "\n".join(lines)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w") as f:
            f.write(text)
        print(
            f"đã ghi {args.out}: {n_state} state, "
            f"encoder_out {list(res[0].shape)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
