# ABOUTME: Test bóc phân rã tầng từ text /metrics của Triton - hàm thuần
# ABOUTME: Chạy: pytest tests/test_stage_breakdown.py

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.triton_metrics import parse_exposition, stage_breakdown  # noqa: E402

TEXT = """
# HELP voice_stage_seconds Thời gian mỗi tầng
voice_stage_seconds_count{model="asr_streaming_prof",stage="fbank"} 100
voice_stage_seconds_sum{model="asr_streaming_prof",stage="fbank"} 0.5
voice_stage_seconds_count{model="asr_streaming_prof",stage="encoder"} 100
voice_stage_seconds_sum{model="asr_streaming_prof",stage="encoder"} 1.5
voice_stage_seconds_count{model="asr_streaming_prof",stage="greedy"} 100
voice_stage_seconds_sum{model="asr_streaming_prof",stage="greedy"} 8.0
voice_stage_seconds_count{model="khac",stage="fbank"} 7
voice_stage_seconds_sum{model="khac",stage="fbank"} 7.0
"""


def _bd():
    return stage_breakdown(parse_exposition(TEXT), "asr_streaming_prof")


def test_cong_don_qua_nhieu_instance():
    # Mỗi model instance là một process riêng, cùng label -> nhiều dòng cùng
    # tên. Cộng lại chứ không lấy dòng cuối.
    samples = parse_exposition(TEXT) + parse_exposition(
        'voice_stage_seconds_count{model="asr_streaming_prof",stage="fbank"} 50\n'
        'voice_stage_seconds_sum{model="asr_streaming_prof",stage="fbank"} 0.25'
    )
    assert stage_breakdown(samples, "asr_streaming_prof")["fbank"]["count"] == 150


def test_bo_qua_model_khac():
    assert set(_bd()) == {"fbank", "encoder", "greedy"}


def test_mean_ms_va_ty_trong():
    bd = _bd()
    assert bd["encoder"]["mean_ms"] == pytest.approx(15.0)
    # 1.5 / (0.5 + 1.5 + 8.0)
    assert bd["encoder"]["share"] == pytest.approx(0.15)
    assert bd["greedy"]["share"] == pytest.approx(0.8)


def test_khong_co_mau_thi_bao_loi():
    # Im lặng trả {} nghĩa là bảng phân rã in ra rỗng và trông như "tầng nào
    # cũng 0ms" - phải hỏng ngay ở chỗ gõ sai tên model.
    with pytest.raises(ValueError, match="asr_streaming_prof"):
        stage_breakdown(parse_exposition('nv_x{model="khac"} 1'), "asr_streaming_prof")


def test_count_0_khong_chia_cho_0():
    samples = parse_exposition(
        'voice_stage_seconds_count{model="m",stage="fbank"} 0\n'
        'voice_stage_seconds_sum{model="m",stage="fbank"} 0'
    )
    assert stage_breakdown(samples, "m")["fbank"]["mean_ms"] == 0.0
