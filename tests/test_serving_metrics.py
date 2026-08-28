# ABOUTME: Test StageMetrics - phân rã thời gian theo tầng, không cần Triton chạy
# ABOUTME: Chạy: pytest tests/test_serving_metrics.py

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serving.metrics import StageMetrics  # noqa: E402


class _FakeMetric:
    def __init__(self, labels, buckets):
        self.labels = labels
        self.buckets = buckets
        self.observed = []

    def observe(self, v):
        self.observed.append(v)


class _FakeFamily:
    GAUGE = "gauge"
    HISTOGRAM = "histogram"

    def __init__(self, name, description, kind):
        self.name = name
        self.kind = kind
        self.metrics = []

    def Metric(self, labels, buckets=None):
        m = _FakeMetric(labels, buckets)
        self.metrics.append(m)
        return m


class _FakeApi:
    MetricFamily = _FakeFamily


def test_dung_histogram_moi_do_duoc_trung_binh():
    # GAUGE chỉ giữ giá trị cuối; muốn biết tầng nào ăn bao nhiêu phải có
    # sum/count cộng dồn -> bắt buộc HISTOGRAM.
    sm = StageMetrics(_FakeApi, "asr_streaming_prof", ["fbank"])
    assert sm._family.kind == _FakeFamily.HISTOGRAM


def test_moi_tang_mot_metric_rieng_theo_label():
    sm = StageMetrics(_FakeApi, "asr_streaming_prof", ["fbank", "encoder", "greedy"])
    got = {m.labels["stage"] for m in sm._family.metrics}
    assert got == {"fbank", "encoder", "greedy"}
    assert all(m.labels["model"] == "asr_streaming_prof" for m in sm._family.metrics)


def test_observe_ghi_dung_tang():
    sm = StageMetrics(_FakeApi, "m", ["fbank", "encoder"])
    sm.observe("encoder", 0.018)
    by = {m.labels["stage"]: m for m in sm._family.metrics}
    assert by["encoder"].observed == [0.018]
    assert by["fbank"].observed == []


def test_tang_la_dat_truoc_khong_tao_lazy():
    # Tạo Metric lúc chạy nghĩa là lần observe đầu của mỗi tầng đắt hơn phần
    # còn lại - đúng thứ đang đi đo thì không được tự bơm nhiễu vào.
    sm = StageMetrics(_FakeApi, "m", ["fbank"])
    with pytest.raises(KeyError):
        sm.observe("khong_ton_tai", 0.1)


def test_timer_do_va_ghi():
    sm = StageMetrics(_FakeApi, "m", ["fbank"])
    with sm.time("fbank"):
        pass
    by = {m.labels["stage"]: m for m in sm._family.metrics}
    assert len(by["fbank"].observed) == 1
    assert by["fbank"].observed[0] >= 0.0
