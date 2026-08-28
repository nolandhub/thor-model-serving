# ABOUTME: Test hàm dựng bảng phân rã tầng - thuần, không cần server
# ABOUTME: Chạy: pytest tests/test_stage_report.py

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.stage_report import render  # noqa: E402


def _bd(chunk_ms, **stages):
    bd = {"chunk": {"mean_ms": chunk_ms, "count": 100.0}}
    for k, v in stages.items():
        bd[k] = {"mean_ms": v, "count": 100.0}
    return bd


def test_hang_ngoai_tang_la_phan_con_lai():
    # 40ms chunk, ba tầng cộng lại 10ms -> 30ms không nằm trong tầng nào.
    out = render(_bd(40.0, fbank=1.0, encoder=8.0, greedy=1.0))
    assert "**30.000**" in out
    assert "**75.0%**" in out


def test_khong_cong_chunk_vao_phan_tram():
    # chunk bọc trọn ba tầng; cộng nó vào tổng thì mọi tỷ trọng bị chia đôi
    out = render(_bd(20.0, encoder=20.0))
    assert "| encoder | 20.000 | 100.0% |" in out


def test_sap_xep_tang_nang_len_dau():
    out = render(_bd(10.0, fbank=1.0, greedy=6.0))
    assert out.index("| greedy ") < out.index("| fbank ")


def test_thieu_chunk_bao_loi():
    with pytest.raises(ValueError, match="chunk"):
        render({"fbank": {"mean_ms": 1.0, "count": 1.0}})
