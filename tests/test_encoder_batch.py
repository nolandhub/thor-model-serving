# ABOUTME: Test các hàm thuần của bench/encoder_batch.py - không cần GPU lẫn file ONNX
# ABOUTME: Chạy: pytest tests/test_encoder_batch.py

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.encoder_batch import batch_axis, per_sample_ms, tile_state  # noqa: E402


# --- batch_axis: tìm chiều batch bằng TÊN ký hiệu, không đoán vị trí ---------
# onnxruntime trả dim động dưới dạng chuỗi ('N', 'batch'...). Hai input cùng
# chiều batch thì dùng CHUNG một tên. Đối chiếu tên là cách duy nhất đúng với
# mọi bản export - đoán "batch luôn ở axis 0" thì state của zipformer sai ngay.


def test_batch_axis_tim_dung_vi_tri_theo_ten():
    assert batch_axis([32, "N", 512], "N") == 1


def test_batch_axis_axis_0():
    assert batch_axis(["N", 3, 4], "N") == 0


def test_batch_axis_khong_co_thi_tra_none():
    # Input không phụ thuộc batch (hằng số, độ dài...) - cứ để nguyên
    assert batch_axis([1, 2, 3], "N") is None


def test_batch_axis_ten_khac_khong_tinh_la_batch():
    # 'T' là chiều thời gian, nhân bản theo nó là tạo ra dữ liệu sai hoàn toàn
    assert batch_axis([1, "T", 80], "N") is None


# --- tile_state: nhân bản state cho batch N ---------------------------------


def test_tile_state_nhan_ban_dung_chieu():
    state = np.zeros((2, 1, 5), dtype=np.float32)
    assert tile_state(state, 1, 8).shape == (2, 8, 5)


def test_tile_state_axis_none_giu_nguyen():
    state = np.zeros((2, 1, 5), dtype=np.float32)
    assert tile_state(state, None, 8).shape == (2, 1, 5)


def test_tile_state_giu_dtype():
    # fp16 mà bị numpy nâng lên fp32 là đo sai model: kernel khác, tốc độ khác
    state = np.zeros((1, 1), dtype=np.float16)
    assert tile_state(state, 0, 4).dtype == np.float16


# --- per_sample_ms: đơn vị so sánh giữa batch 1 và batch N ------------------


def test_per_sample_ms_chia_cho_batch():
    # 80ms cho một lần chạy batch 8 = 10ms mỗi mẫu
    assert per_sample_ms([0.08], 8) == pytest.approx(10.0)


def test_per_sample_ms_lay_trung_vi():
    # Trung vị: lần chạy đầu luôn dính chi phí cấp bộ nhớ, đừng để nó kéo số
    assert per_sample_ms([0.01, 0.02, 0.03], 1) == pytest.approx(20.0)


def test_per_sample_ms_rong_bao_loi():
    with pytest.raises(ValueError):
        per_sample_ms([], 1)
