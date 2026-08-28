# ABOUTME: Kiểm bản encoder.onnx đã bọc khớp numeric với bản gốc và chạy được ở batch > 1
# ABOUTME: Cần file encoder.onnx thật; tự bỏ qua khi không có (CI không giữ trọng số)

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ONNX = ROOT / "model_repository/asr_streaming/1/encoder.onnx"
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytestmark = pytest.mark.skipif(not ONNX.exists(), reason=f"không có {ONNX}")

from bench.encoder_batch import batch_axis  # noqa: E402
from scripts.wrap_encoder import (  # noqa: E402
    needs_trailing_dim,
    to_batch_first_perm,
    wrap,
)

_NP = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int64)": np.int64}


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    import onnx

    model = onnx.load(str(ONNX))
    symbol = model.graph.input[0].type.tensor_type.shape.dim[0].dim_param
    wrapped, _ = wrap(model, symbol)
    out = tmp_path_factory.mktemp("wrap") / "wrapped.onnx"
    onnx.save(wrapped, str(out))
    p = ["CPUExecutionProvider"]
    return (
        ort.InferenceSession(str(ONNX), providers=p),
        ort.InferenceSession(str(out), providers=p),
        symbol,
    )


def _zeros(specs, b):
    return [
        np.zeros([b if isinstance(d, str) else d for d in s.shape], dtype=_NP[s.type])
        for s in specs
    ]


def test_batch_1_khop_tuyet_doi_qua_nhieu_chunk(sessions):
    """Bọc chỉ chèn Transpose/Squeeze - không đụng phép tính nào, nên phải
    khớp BIT-FOR-BIT, không phải 'trong sai số'. Lệch dù nhỏ là đã xoay nhầm
    chiều ở đâu đó và kết quả sẽ sai âm thầm khi state tích luỹ."""
    so, sw, _ = sessions
    oi, ow = so.get_inputs(), sw.get_inputs()
    t, f = oi[0].shape[1], oi[0].shape[2]
    rng = np.random.default_rng(0)
    so_st, sw_st = _zeros(oi[1:], 1), _zeros(ow[1:], 1)
    for _ in range(4):
        x = rng.standard_normal((1, t, f)).astype(_NP[oi[0].type])
        a = so.run(None, {oi[0].name: x, **{i.name: v for i, v in zip(oi[1:], so_st)}})
        b = sw.run(None, {ow[0].name: x, **{i.name: v for i, v in zip(ow[1:], sw_st)}})
        so_st, sw_st = a[1:], b[1:]
        assert np.array_equal(a[0], b[0])


def test_state_khop_sau_khi_xoay_nguoc(sessions):
    so, sw, symbol = sessions
    oi, ow = so.get_inputs(), sw.get_inputs()
    t, f = oi[0].shape[1], oi[0].shape[2]
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, t, f)).astype(_NP[oi[0].type])
    a = so.run(None, {oi[0].name: x, **{i.name: v for i, v in zip(oi[1:], _zeros(oi[1:], 1))}})
    b = sw.run(None, {ow[0].name: x, **{i.name: v for i, v in zip(ow[1:], _zeros(ow[1:], 1))}})
    for spec, vo, vw in zip(oi[1:], a[1:], b[1:]):
        ax = batch_axis(spec.shape, symbol)
        if needs_trailing_dim(spec.shape, ax):
            back = vw.reshape(vo.shape)
        elif ax:
            back = np.transpose(vw, to_batch_first_perm(vw.ndim, ax))
        else:
            back = vw
        assert np.array_equal(vo, back), spec.name


def test_moi_state_co_batch_o_axis_0(sessions):
    """Đây là toàn bộ lý do bọc: max_batch_size > 0 thì Triton nối batch theo
    axis 0 cho MỌI input/output, không có ngoại lệ."""
    _, sw, symbol = sessions
    for spec in list(sw.get_inputs())[1:] + list(sw.get_outputs())[1:]:
        assert batch_axis(spec.shape, symbol) == 0, spec.name


def test_chay_duoc_o_batch_lon_hon_1(sessions):
    """Bản gốc không chạy nổi kiểu này - state time-major thì ORT nhận, nhưng
    Triton không bao giờ gửi tới được vì nó chỉ ghép batch ở axis 0."""
    _, sw, _ = sessions
    ow = sw.get_inputs()
    t, f = ow[0].shape[1], ow[0].shape[2]
    b = 6
    x = np.random.default_rng(2).standard_normal((b, t, f)).astype(_NP[ow[0].type])
    out = sw.run(None, {ow[0].name: x, **{i.name: v for i, v in zip(ow[1:], _zeros(ow[1:], b))}})
    assert out[0].shape[0] == b
