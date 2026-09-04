# ABOUTME: Metric runtime cho Triton Python backend - RTF và CCU, thứ nv_* không thấy được
# ABOUTME: pb_utils tiêm từ ngoài để test chạy được khi server tắt

import time

# Soi gương max_sequence_idle_microseconds trong asr_bls/config.pbtxt.
# Query Grafana dùng đúng con số này để bỏ qua instance im lặng; lệch là CCU sai
# âm thầm. test_serving_metrics.py và test_monitoring_config.py canh cả ba nơi.
CCU_TTL_S = 60.0

# Hai thang lệch hẳn một bậc (bench/: ASR quanh 0.05, TTS quanh 0.86) nên không
# dùng chung buckets được - dùng chung thì một trong hai dồn hết vào 1-2 bucket.
ASR_RTF_BUCKETS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0]
TTS_RTF_BUCKETS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]


def rtf(compute_s: float, audio_s: float) -> float:
    """Thời gian xử lý / độ dài audio. Dưới 1.0 là nhanh hơn thời gian thực.

    Cùng công thức bench/tts/metrics.py:rtf() để hai nguồn so được với nhau,
    chỉ khác: không làm tròn, vì histogram cần giá trị thô mới rơi đúng bucket.
    """
    if audio_s <= 0:
        raise ValueError(f"audio length {audio_s}s - cannot compute RTF")
    return compute_s / audio_s


class ModelMetrics:
    """Ba metric family Triton không tự có. Một đối tượng cho mỗi model instance.

    metric_api: đối tượng có .MetricFamily kèm .GAUGE/.HISTOGRAM. Production
    truyền thẳng pb_utils; test truyền fake. Tiêm thay vì import ở đầu file vì
    pb_utils chỉ tồn tại bên trong container Triton.
    """

    def __init__(self, metric_api, model: str, instance: str, rtf_buckets):
        family = metric_api.MetricFamily

        # Giữ family làm thuộc tính là BẮT BUỘC, không phải cho gọn. Stub Triton:
        # "The 'MetricFamily' object should be deleted AFTER its corresponding
        # 'Metric' objects have been deleted." Viết family(...).Metric(...) rồi
        # chỉ giữ Metric thì family bị GC và metric chết im lặng lúc chạy thật.
        self._rtf_family = family(
            name="voice_rtf",
            description="Processing time divided by audio duration",
            kind=family.HISTOGRAM,
        )
        # RTF để label chung: observe() của HISTOGRAM cộng dồn đúng qua nhiều
        # process, bucket gộp chuẩn. Tách theo instance chỉ chẻ vụn vô ích.
        self._rtf = self._rtf_family.Metric(
            labels={"model": model}, buckets=list(rtf_buckets)
        )

        # CCU thì ngược lại. GAUGE set() với label chung sẽ ghi đè giữa các
        # instance, giá trị cuối là của process nào chạy sau. asr_bls có
        # count: 2 nên đây là ca thật. Tách theo instance, cộng lại ở PromQL.
        #
        # Tên label là "model_instance" chứ không phải "instance": Prometheus
        # tự gắn label "instance" = địa chỉ target vào mọi metric lúc scrape,
        # đụng tên với label tự phát này thì bị ghi đè mất giá trị thật.
        ccu_labels = {"model": model, "model_instance": instance}
        self._ccu_family = family(
            name="voice_ccu",
            description="Live sessions on one model instance",
            kind=family.GAUGE,
        )
        self._ccu = self._ccu_family.Metric(labels=dict(ccu_labels))

        self._ccu_at_family = family(
            name="voice_ccu_updated_at",
            description="Unix timestamp of the last voice_ccu update",
            kind=family.GAUGE,
        )
        self._ccu_at = self._ccu_at_family.Metric(labels=dict(ccu_labels))

    def observe_rtf(self, compute_s: float, audio_s: float) -> None:
        self._rtf.observe(rtf(compute_s, audio_s))

    def set_ccu(self, n: int) -> None:
        """Set cả hai gauge trong một lần gọi - tách ra là chúng lệch nhau.

        time.time() chứ không monotonic: giá trị này đem so với time() của
        Prometheus ở PromQL nên phải cùng gốc unix epoch.
        """
        self._ccu.set(n)
        self._ccu_at.set(time.time())
