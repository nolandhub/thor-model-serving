# ABOUTME: Test phần thuần của việc bọc encoder.onnx cho state batch-first
# ABOUTME: Chạy: pytest tests/test_wrap_encoder.py

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.wrap_encoder import (  # noqa: E402
    batch_first_shape,
    from_batch_first_perm,
    to_batch_first_perm,
)


# --- shape --------------------------------------------------------------------


def test_batch_first_shape_dua_batch_len_dau():
    # state zipformer là time-major: [left_context, N, dim]
    assert batch_first_shape(["128", "N", "128"], 1) == ["N", "128", "128"]


def test_batch_first_shape_bon_chieu():
    assert batch_first_shape(["1", "N", "128", "144"], 1) == ["N", "1", "128", "144"]


def test_batch_first_shape_da_o_dau_thi_giu_nguyen():
    assert batch_first_shape(["N", "192", "15"], 0) == ["N", "192", "15"]


# --- perm ---------------------------------------------------------------------
# Hai chiều ngược nhau và RẤT dễ lẫn. Đầu vào: Triton đưa tensor batch-first,
# phải xoay VỀ layout gốc để nạp vào đồ thị cũ. Đầu ra: đồ thị cũ nhả layout
# gốc, phải xoay THÀNH batch-first cho Triton. Lẫn hai cái là model vẫn chạy,
# vẫn ra số, và sai âm thầm.


def test_to_batch_first_perm_goc_sang_batch_first():
    # [128, N, 128] -> [N, 128, 128]: lấy axis 1 lên đầu
    assert to_batch_first_perm(3, 1) == [1, 0, 2]


def test_from_batch_first_perm_la_nghich_dao():
    assert from_batch_first_perm(3, 1) == [1, 0, 2]


def test_hai_perm_bu_nhau_o_moi_rank():
    # Bất biến duy nhất đáng test: xoay đi rồi xoay lại phải về chỗ cũ.
    for rank in range(1, 6):
        for ax in range(rank):
            fwd = to_batch_first_perm(rank, ax)
            bwd = from_batch_first_perm(rank, ax)
            assert [fwd[i] for i in bwd] == list(range(rank)), (rank, ax)


def test_perm_bon_chieu_batch_o_giua():
    # [1, N, 128, 144] -> [N, 1, 128, 144]
    assert to_batch_first_perm(4, 1) == [1, 0, 2, 3]


def test_batch_axis_ngoai_pham_vi_bao_loi():
    with pytest.raises(ValueError, match="rank"):
        to_batch_first_perm(3, 3)


# --- state vô hướng mỗi mẫu ---------------------------------------------------
# processed_lens có shape ['N']: bỏ chiều batch là còn rỗng, mà Triton không
# nhận dims rỗng. Phải nống thành ['N', 1] ngay trong lúc bọc.


def test_can_them_chieu_khi_moi_mau_la_vo_huong():
    from scripts.wrap_encoder import needs_trailing_dim
    assert needs_trailing_dim(["N"], 0) is True


def test_khong_them_chieu_khi_da_du():
    from scripts.wrap_encoder import needs_trailing_dim
    assert needs_trailing_dim(["N", "192", "15"], 0) is False
    assert needs_trailing_dim(["128", "N", "128"], 1) is False
