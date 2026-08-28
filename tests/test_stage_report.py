# ABOUTME: Test hàm dựng bảng phân rã tầng - thuần, không cần server
# ABOUTME: Chạy: pytest tests/test_stage_report.py

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.stage_report import render  # noqa: E402


def _st(sum_s, count):
    return {"sum_s": sum_s, "count": count, "mean_ms": sum_s / count * 1000 if count else 0.0}


def test_ty_trong_tinh_theo_tong_khong_phai_mean():
    # Đây là con bug đã ra bảng 128% + phần dư âm trên Thor: encoder chạy 0.6
    # lần mỗi chunk nên mean/lần của nó LỚN hơn mean của cả chunk. Chia hai
    # mean cho nhau là sai; phải cộng sum.
    bd = {
        "chunk": _st(10.0, 100),      # 100ms/chunk
        "encoder": _st(8.0, 60),      # 133ms/lần, nhưng chỉ 0.6 lần/chunk
    }
    out = render(bd)
    assert "| 133.33 | 0.60 | 80.00 | 80.0% |" in out


def test_ngoai_tang_la_phan_con_lai():
    bd = {"chunk": _st(10.0, 100), "fbank": _st(1.0, 100), "greedy": _st(1.0, 100)}
    out = render(bd)
    assert "**ngoài tầng**" in out
    assert "80.0%" in out


def test_sap_xep_theo_tong_khong_theo_mean():
    # greedy mean nhỏ nhưng chạy nhiều lần -> tổng lớn hơn, phải đứng trên
    bd = {"chunk": _st(10.0, 100), "fbank": _st(1.0, 100), "greedy": _st(5.0, 500)}
    out = render(bd)
    assert out.index("| greedy ") < out.index("| fbank ")


def test_thieu_chunk_bao_loi():
    with pytest.raises(ValueError, match="chunk"):
        render({"fbank": _st(1.0, 1)})
