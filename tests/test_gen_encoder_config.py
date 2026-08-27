# ABOUTME: Test phần dựng config.pbtxt cho model encoder - hàm thuần, không cần file ONNX
# ABOUTME: Chạy: pytest tests/test_gen_encoder_config.py

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.gen_encoder_config import render_config, state_pairs, triton_dims  # noqa: E402


# --- state_pairs: ghép state vào/ra THEO VỊ TRÍ ------------------------------
# Đây đúng bất biến mà model.py hiện tại đang dựa vào và đã khẳng định bằng
# assert: input[1:] khớp output[1:] theo thứ tự. Ghép theo tên sẽ hỏng vì tên
# vào và tên ra khác nhau (cached_x vs new_cached_x).


def test_state_pairs_ghep_theo_vi_tri():
    assert state_pairs(["a_in", "b_in"], ["a_out", "b_out"]) == [
        ("a_in", "a_out"), ("b_in", "b_out")
    ]


def test_state_pairs_lech_so_luong_bao_loi():
    with pytest.raises(ValueError, match="lệch"):
        state_pairs(["a_in"], ["a_out", "b_out"])


# --- triton_dims: bỏ chiều batch --------------------------------------------
# max_batch_size > 0 thì Triton NGẦM hiểu chiều đầu là batch và dims khai phần
# còn lại. Khai cả chiều batch vào dims là model nạp được nhưng shape sai.


def test_triton_dims_bo_chieu_batch():
    assert triton_dims(["N", 2, 512], 0) == [2, 512]


def test_triton_dims_batch_khong_o_axis_0_bao_loi():
    # Triton chỉ batch được khi chiều batch đứng ĐẦU. State có batch ở giữa thì
    # không dùng được max_batch_size, phải biết ngay chứ không phải sau khi nạp
    with pytest.raises(ValueError, match="axis 0"):
        triton_dims([2, "N", 512], 1)


def test_triton_dims_chieu_dong_khac_batch_bao_loi():
    with pytest.raises(ValueError, match="động"):
        triton_dims(["N", "T", 512], 0)


# --- render_config -----------------------------------------------------------


def _cfg():
    return render_config(
        name="encoder",
        x_name="x",
        x_dims=[45, 80],
        x_type="TYPE_FP32",
        states=[{"input_name": "c_in", "output_name": "c_out",
                 "data_type": "TYPE_FP32", "dims": [2, 512]}],
        max_batch=8,
        max_candidate=8,
    )


def test_render_config_co_ten_va_backend():
    cfg = _cfg()
    assert 'name: "encoder"' in cfg
    assert 'platform: "onnxruntime_onnx"' in cfg


def test_render_config_khai_state_du_ba_phan():
    cfg = _cfg()
    assert 'input_name: "c_in"' in cfg
    assert 'output_name: "c_out"' in cfg
    assert "dims: [ 2, 512 ]" in cfg


def test_render_config_initial_state_zero():
    # Đúng như _Stream.__init__ của đường cũ: state khởi tạo bằng zeros
    assert "zero_data: true" in _cfg()


def test_render_config_dung_oldest_de_gom_nhieu_sequence():
    # oldest là chiến lược DUY NHẤT gom được request từ nhiều sequence khác nhau
    # vào một batch - direct thì mỗi sequence một slot riêng, avg_batch mãi 1.0
    assert "oldest" in _cfg()
    assert "max_candidate_sequences: 8" in _cfg()
