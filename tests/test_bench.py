# ABOUTME: Test các hàm thuần của bench/ - chạy được khi Thor tắt, không cần GPU lẫn server
# ABOUTME: Chạy: pytest tests/test_bench.py

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.report import (  # noqa: E402
    aggregate,
    avg_batch,
    bls_tax,
    diff_counters,
    drifts,
    max_ccu_within_budget,
    pct,
    run_summary,
)
from bench.run_asr import (  # noqa: E402
    markdown_table,
    metrics_reader,
    parse_ccus,
    ClockSampler,
    clock_mhz,
    model_config_summary,
    warmup_slice,
    throttled,
    without_proxy_env,
)
from bench.schedule import send_deadlines  # noqa: E402
from bench.triton_metrics import counters_for_model, parse_exposition  # noqa: E402


# --- pct: nearest-rank ------------------------------------------------------
# Latency dùng nearest-rank chứ không nội suy: p99 phải là một mẫu CÓ THẬT đã
# quan sát được, không phải số trung bình giữa hai mẫu không ai từng gặp.


def test_pct_nearest_rank_tra_ve_mau_co_that():
    values = list(range(1, 101))   # 1..100
    assert pct(values, 50) == 50
    assert pct(values, 99) == 99
    assert pct(values, 100) == 100


def test_pct_khong_noi_suy_giua_hai_mau():
    assert pct([1, 2, 3, 4], 50) == 2


def test_pct_khong_phu_thuoc_thu_tu_dau_vao():
    assert pct([9, 1, 5, 3], 50) == pct([1, 3, 5, 9], 50)


def test_pct_danh_sach_rong_bao_loi():
    with pytest.raises(ValueError):
        pct([], 50)


# --- drifts -----------------------------------------------------------------
# Mốc realtime hoàn hảo của chunk i là t_start + (i+1)*chunk_s: chunk i chứa
# audio [i, i+1) nên hệ thống đúng nhịp phải trả kết quả ngay khi audio đó hết.
# drift = 0 nghĩa là bám sát thời gian thực; drift TĂNG DẦN nghĩa là đang sập.


def test_drift_bang_khong_khi_dung_nhip_thoi_gian_thuc():
    assert drifts([0.2, 0.4, 0.6], t_start=0.0, chunk_s=0.2) == pytest.approx([0, 0, 0])


def test_drift_tang_dan_khi_server_cham_hon_realtime():
    d = drifts([0.25, 0.50, 0.75], t_start=0.0, chunk_s=0.2)
    assert d == pytest.approx([0.05, 0.10, 0.15])
    assert d[-1] > d[0], "drift phải phân kỳ khi mỗi chunk trễ thêm"


def test_drift_am_khi_server_tra_som_hon_audio_ket_thuc():
    assert drifts([0.15], t_start=0.0, chunk_s=0.2) == pytest.approx([-0.05])


# --- avg_batch --------------------------------------------------------------
# Chỉ số quyết định: BLS chỉ có nghĩa nếu Triton thật sự gộp được request.


def test_avg_batch_la_ty_so_request_tren_exec():
    assert avg_batch(request_count=40, exec_count=10) == 4.0


def test_avg_batch_bang_mot_khi_khong_gop_duoc_gi():
    assert avg_batch(request_count=37, exec_count=37) == 1.0


def test_avg_batch_exec_bang_khong_bao_loi():
    with pytest.raises(ValueError):
        avg_batch(request_count=10, exec_count=0)


# --- bls_tax ----------------------------------------------------------------
# Chi phí copy tensor qua ranh giới model nằm ở compute_input + compute_output.


def test_bls_tax_la_chi_phi_copy_chia_cho_compute_that():
    assert bls_tax(compute_input_us=30, compute_infer_us=100, compute_output_us=20) == 0.5


def test_bls_tax_bang_khong_khi_khong_co_copy():
    assert bls_tax(compute_input_us=0, compute_infer_us=100, compute_output_us=0) == 0.0


def test_bls_tax_compute_infer_bang_khong_bao_loi():
    with pytest.raises(ValueError):
        bls_tax(compute_input_us=5, compute_infer_us=0, compute_output_us=5)


# --- diff_counters ----------------------------------------------------------
# Counter Triton là cộng dồn. Server restart giữa hai snapshot làm nó tụt về 0,
# lúc đó Δ âm - phải hỏng to chứ không được lặng lẽ trả số vô nghĩa.


def test_diff_counters_tru_tung_metric():
    assert diff_counters({"a": 10.0, "b": 5.0}, {"a": 25.0, "b": 5.0}) == {"a": 15.0, "b": 0.0}


def test_diff_counters_am_bao_loi_vi_counter_da_reset():
    with pytest.raises(ValueError, match="reset"):
        diff_counters({"a": 100.0}, {"a": 3.0})


def test_diff_counters_thieu_metric_o_snapshot_sau_bao_loi():
    with pytest.raises(ValueError, match="a"):
        diff_counters({"a": 1.0}, {})


# --- max_ccu_within_budget --------------------------------------------------
# Sức chứa là điểm CCU cao nhất mà MỌI mức thấp hơn cũng còn đạt ngưỡng. Một
# mức vỡ rồi mà mức cao hơn lại đẹp thì đó là nhiễu, không phải sức chứa.


def test_max_ccu_la_diem_cao_nhat_con_dat_nguong():
    assert max_ccu_within_budget({1: 0.05, 2: 0.08, 4: 0.15, 8: 0.35}, budget_s=0.2) == 4


def test_max_ccu_bang_khong_khi_ngay_mot_stream_da_vo():
    assert max_ccu_within_budget({1: 0.5, 2: 0.6}, budget_s=0.2) == 0


def test_max_ccu_khong_nhay_qua_muc_da_vo():
    # 4 vỡ, 8 đẹp -> 8 là nhiễu, sức chứa vẫn dừng ở 2
    assert max_ccu_within_budget({1: 0.05, 2: 0.1, 4: 0.9, 8: 0.12}, budget_s=0.2) == 2


# --- send_deadlines ---------------------------------------------------------
# Lịch gửi phải TUYỆT ĐỐI. Nếu tính bằng sleep(chunk) cộng dồn thì overhead của
# chính client trôi vào mốc gửi, và drift đo được sẽ là drift của client chứ
# không phải của server - hỏng đúng thứ bài bench sinh ra để đo.


def test_send_deadlines_cach_deu_tu_moc_bat_dau():
    assert send_deadlines(t_start=100.0, n=3, chunk_s=0.2) == pytest.approx([100.0, 100.2, 100.4])


def test_send_deadlines_khong_troi_sau_nhieu_chunk():
    d = send_deadlines(t_start=0.0, n=1000, chunk_s=0.2)
    assert d[-1] == pytest.approx(999 * 0.2, abs=1e-9)


# --- chunk_wav (đã có sẵn, chưa có test) ------------------------------------
# client/common.py import soundfile ở đầu file, mà máy dev có thể chưa cài -
# skip chứ không để cả module test hỏng, phần bench thuần vẫn phải chạy được.


def test_chunk_wav_bo_phan_du_thay_vi_dem_im_lang():
    pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    from client.common import SAMPLE_RATE, chunk_wav

    wav = np.zeros(SAMPLE_RATE, dtype=np.float32)   # 1s
    wav = np.concatenate([wav, np.zeros(100, dtype=np.float32)])
    parts = chunk_wav(wav, chunk_ms=200)
    assert len(parts) == 5
    assert all(len(p) == SAMPLE_RATE * 200 // 1000 for p in parts)


# --- parser định dạng text của Triton /metrics ------------------------------
# Bench đọc thẳng localhost:8002/metrics TRONG container (compose cố ý không
# publish 8002 ra host). Không qua Prometheus: chu kỳ scrape 15s mà một run chỉ
# ~20s, hai snapshot có khi rơi vào cùng một điểm dữ liệu và Δ ra 0.

EXPOSITION = """\
# HELP nv_inference_request_success Number of successful inference requests
# TYPE nv_inference_request_success counter
nv_inference_request_success{model="asr_streaming",version="1"} 1234
nv_inference_exec_count{model="asr_streaming",version="1"} 617
nv_inference_compute_infer_duration_us{model="asr_streaming",version="1"} 1.234e+06
nv_inference_exec_count{model="encoder",version="1"} 90

# TYPE nv_gpu_utilization gauge
nv_gpu_utilization{gpu_uuid="GPU-abc"} 0.73
"""


def test_parse_doc_duoc_ten_label_va_gia_tri():
    samples = parse_exposition(EXPOSITION)
    assert ("nv_inference_exec_count", {"model": "encoder", "version": "1"}, 90.0) in samples


def test_parse_bo_qua_dong_chu_thich_va_dong_trong():
    for name, _labels, _v in parse_exposition(EXPOSITION):
        assert not name.startswith("#")
    assert len(parse_exposition(EXPOSITION)) == 5


def test_parse_hieu_ky_hieu_khoa_hoc():
    # Triton xuất duration dạng 1.234e+06 - đọc bằng int() là vỡ
    samples = parse_exposition(EXPOSITION)
    value = next(v for n, _l, v in samples if n == "nv_inference_compute_infer_duration_us")
    assert value == pytest.approx(1234000.0)


def test_parse_dong_khong_co_label():
    assert parse_exposition("nv_cpu_utilization 0.5\n") == [("nv_cpu_utilization", {}, 0.5)]


# --- counters_for_model -----------------------------------------------------


def test_counters_for_model_chi_lay_dung_model():
    c = counters_for_model(parse_exposition(EXPOSITION), "encoder")
    assert c == {"nv_inference_exec_count": 90.0}


def test_counters_for_model_cong_don_qua_nhieu_version():
    text = (
        'nv_inference_exec_count{model="m",version="1"} 10\n'
        'nv_inference_exec_count{model="m",version="2"} 5\n'
    )
    assert counters_for_model(parse_exposition(text), "m") == {"nv_inference_exec_count": 15.0}


def test_counters_for_model_khong_co_mau_nao_bao_loi():
    # Gõ sai tên model mà trả dict rỗng thì diff_counters cũng rỗng, avg_batch
    # không bao giờ được gọi, và bench báo "xong" mà không đo gì cả.
    with pytest.raises(ValueError, match="khong_ton_tai"):
        counters_for_model(parse_exposition(EXPOSITION), "khong_ton_tai")


# --- run_summary: gộp một run thành các metric quyết định --------------------
# Numerator của avg_batch là nv_inference_count (Triton đã nhân batch size),
# KHÔNG phải nv_inference_request_success (đếm request). Lấy nhầm thì tỷ số
# luôn ra ~1.0 và bench kết luận sai rằng BLS không gom được gì.


def _record(**over):
    rec = {
        "label": "baseline",
        "ccu": 2,
        "run": 0,
        "chunk_s": 0.2,
        "valid": True,
        "counters_before": {
            "nv_inference_count": 0.0,
            "nv_inference_exec_count": 0.0,
            "nv_inference_compute_input_duration_us": 0.0,
            "nv_inference_compute_infer_duration_us": 0.0,
            "nv_inference_compute_output_duration_us": 0.0,
            "nv_inference_queue_duration_us": 0.0,
        },
        "counters_after": {
            "nv_inference_count": 40.0,
            "nv_inference_exec_count": 20.0,
            "nv_inference_compute_input_duration_us": 30.0,
            "nv_inference_compute_infer_duration_us": 100.0,
            "nv_inference_compute_output_duration_us": 20.0,
            "nv_inference_queue_duration_us": 800.0,
        },
        "streams": [
            {"t_start": 0.0, "send": [0.0, 0.2], "recv": [0.25, 0.45]},
            {"t_start": 0.0, "send": [0.0, 0.2], "recv": [0.10, 0.30]},
        ],
    }
    rec.update(over)
    return rec


def test_run_summary_gop_latency_cua_moi_stream_vao_mot_phan_phoi():
    # 4 chunk: 0.25, 0.25, 0.10, 0.10 -> max = 0.25
    s = run_summary(_record())
    assert s["max_latency_s"] == pytest.approx(0.25)


def test_run_summary_lay_drift_cua_stream_te_nhat_khong_phai_trung_binh():
    # stream 0 drift cuối = 0.45 - 0.4 = 0.05 ; stream 1 = 0.30 - 0.4 = -0.10
    # Một stream tệ là một cuộc gọi tệ - trung bình sẽ giấu mất nó.
    assert run_summary(_record())["final_drift_s"] == pytest.approx(0.05)


def test_run_summary_tinh_avg_batch_tu_nv_inference_count():
    assert run_summary(_record())["avg_batch"] == pytest.approx(2.0)


def test_run_summary_tinh_bls_tax_tu_compute_input_va_output():
    assert run_summary(_record())["bls_tax"] == pytest.approx(0.5)


def test_run_summary_queue_tinh_tren_moi_request():
    assert run_summary(_record())["queue_us_per_request"] == pytest.approx(20.0)


# --- aggregate: gộp nhiều run của cùng một mức CCU ---------------------------


def test_aggregate_lay_median_giua_cac_run():
    runs = [
        _record(run=0, streams=[{"t_start": 0.0, "send": [0.0], "recv": [0.10]}]),
        _record(run=1, streams=[{"t_start": 0.0, "send": [0.0], "recv": [0.30]}]),
        _record(run=2, streams=[{"t_start": 0.0, "send": [0.0], "recv": [0.20]}]),
    ]
    assert aggregate(runs)[2]["max_latency_s"] == pytest.approx(0.20)


def test_aggregate_loai_run_danh_dau_invalid():
    runs = [
        _record(run=0, streams=[{"t_start": 0.0, "send": [0.0], "recv": [0.10]}]),
        _record(run=1, valid=False, streams=[{"t_start": 0.0, "send": [0.0], "recv": [9.9]}]),
    ]
    assert aggregate(runs)[2]["max_latency_s"] == pytest.approx(0.10)


def test_aggregate_khong_con_run_hop_le_nao_bao_loi():
    with pytest.raises(ValueError, match="hợp lệ"):
        aggregate([_record(valid=False)])


def test_aggregate_tach_theo_tung_muc_ccu():
    runs = [_record(ccu=1), _record(ccu=8)]
    assert sorted(aggregate(runs)) == [1, 8]


# --- parse_ccus: dải tải từ dòng lệnh ---------------------------------------


def test_parse_ccus_doc_danh_sach_ngan_cach_bang_phay():
    assert parse_ccus("1,2,4,8") == [1, 2, 4, 8]


def test_parse_ccus_sap_xep_tang_dan():
    # max_ccu_within_budget quét từ dưới lên, nên thứ tự chạy phải tăng dần
    assert parse_ccus("8,1,4") == [1, 4, 8]


def test_parse_ccus_bo_muc_trung_lap():
    assert parse_ccus("1,1,2") == [1, 2]


def test_parse_ccus_khong_duong_bao_loi():
    with pytest.raises(ValueError, match="0"):
        parse_ccus("1,0,2")


def test_parse_ccus_khong_phai_so_bao_loi():
    with pytest.raises(ValueError):
        parse_ccus("1,x")


# --- throttled: bẫy nhiệt, số sai 2-3 lần nếu bỏ qua ------------------------


def test_throttled_gpuidle_khong_phai_bi_ham():
    assert throttled("0x0000000000000001") is False


def test_throttled_khong_co_ly_do_nao_la_binh_thuong():
    assert throttled("0x0000000000000000") is False


def test_throttled_co_ly_do_khac_la_dang_bi_ham():
    # 0x...0004 = SwPowerCap, clock tụt và mọi số đo sau đó đều sai
    assert throttled("0x0000000000000004") is True


def test_throttled_khong_doc_duoc_thi_coi_nhu_khong_ham():
    # Jetson/Thor có thể không xuất trường này - không được vì thế mà loại sạch run
    assert throttled("[N/A]") is False
    assert throttled("") is False


# --- markdown_table: bảng kết quả -------------------------------------------


def test_markdown_table_mot_dong_moi_muc_ccu():
    agg = {
        1: {"p99_latency_s": 0.12, "final_drift_s": 0.01, "avg_batch": 1.0,
            "bls_tax": 0.5, "queue_us_per_request": 20.0, "gpu_mhz_p50": 1575.0},
        4: {"p99_latency_s": 0.40, "final_drift_s": 0.30, "avg_batch": 2.5,
            "bls_tax": 0.6, "queue_us_per_request": 90.0, "gpu_mhz_p50": 1566.0},
    }
    rows = [line for line in markdown_table(agg).splitlines() if line.startswith("| ")]
    assert len(rows) == 3          # header + 2 mức CCU
    assert rows[1].startswith("| 1 ")
    assert rows[2].startswith("| 4 ")


# --- metrics_reader: chọn nguồn counter theo chỗ bench đang đứng -------------


def test_metrics_reader_uu_tien_url_khi_chay_trong_container():
    # Trong network compose thì gọi thẳng asr:8002 - container không có docker socket
    reader = metrics_reader("thor-asr-triton", "http://asr:8002/metrics")
    assert "http://asr:8002/metrics" in repr(reader)


def test_metrics_reader_dung_docker_exec_khi_chay_tren_host():
    # Trên host thì 8002 KHÔNG publish ra ngoài, phải đọc từ bên trong container
    reader = metrics_reader("thor-asr-triton", None)
    assert "thor-asr-triton" in repr(reader)


def test_metrics_reader_khong_co_nguon_nao_bao_loi():
    with pytest.raises(ValueError, match="nguồn"):
        metrics_reader(None, None)


# --- without_proxy_env: proxy công ty nướng sẵn trong image base -------------


def test_without_proxy_env_bo_moi_bien_proxy():
    # urllib VÀ grpc đều đọc http_proxy: để nguyên thì bench gọi proxy công ty
    # chứ không gọi asr, và proxy refuse - đúng lỗi đã gặp trên Thor
    env = {"http_proxy": "http://gw:8080", "HTTPS_PROXY": "http://gw:8080", "PATH": "/bin"}
    assert without_proxy_env(env) == {"PATH": "/bin"}


def test_without_proxy_env_bo_ca_no_proxy():
    # no_proxy ở lại là vô nghĩa khi đã không còn proxy nào, mà lại dễ gây hiểu
    # nhầm là bench có đi qua proxy trong vài trường hợp
    assert without_proxy_env({"no_proxy": "localhost"}) == {}


def test_without_proxy_env_khong_dung_bien_khac():
    env = {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"}
    assert without_proxy_env(env) == env


def test_without_proxy_env_khong_sua_dict_goc():
    env = {"http_proxy": "http://gw:8080"}
    without_proxy_env(env)
    assert env == {"http_proxy": "http://gw:8080"}


# --- run_summary chỉ diff counter đơn điệu ----------------------------------


def test_run_summary_bo_qua_summary_va_gauge_khong_don_dieu():
    """Triton xuất cả summary (cửa sổ trượt) lẫn gauge cạnh counter.

    nv_inference_request_summary_us TỤT là chuyện bình thường của summary, còn
    pending_request_count là gauge. Bắt chúng phải tăng thì bench chết oan sau
    khi đã chạy xong toàn bộ phép đo - đúng loại hỏng đắt nhất.
    """
    rec = _record()
    rec["counters_before"] = {**rec["counters_before"],
                              "nv_inference_request_summary_us": 355745.0,
                              "nv_inference_pending_request_count": 3.0}
    rec["counters_after"] = {**rec["counters_after"],
                             "nv_inference_request_summary_us": 182279.0,
                             "nv_inference_pending_request_count": 0.0}
    assert run_summary(rec)["avg_batch"] == pytest.approx(2.0)


def test_run_summary_van_bat_duoc_counter_that_su_reset():
    # Chốt chặn phải còn sống với counter thật: server restart giữa run thì Δ
    # âm sẽ lặng lẽ chảy vào avg_batch thành một kết luận sai không truy ra được
    rec = _record()
    rec["counters_before"] = dict(rec["counters_after"])
    rec["counters_after"] = {**rec["counters_after"], "nv_inference_count": 0.0}
    with pytest.raises(ValueError, match="reset"):
        run_summary(rec)


def test_run_summary_thieu_counter_can_dung_bao_loi():
    rec = _record()
    del rec["counters_before"]["nv_inference_exec_count"]
    with pytest.raises(ValueError, match="nv_inference_exec_count"):
        run_summary(rec)


# --- clock_mhz: Thor không báo throttle, chỉ còn cách tự nhìn clock ----------
# nvidia-smi trên Tegra trả [N/A] cho clocks.sm lẫn throttle reasons, và
# /sys/class/thermal rỗng - không có cooling device nào để hỏi. Thứ duy nhất
# đọc được là tần số GPC, và nó chỉ có nghĩa khi lấy mẫu LÚC ĐANG TẢI.


def test_clock_mhz_doi_hz_sang_mhz():
    assert clock_mhz([1575000000, 1575000000]) == {"p50": 1575.0, "max": 1575.0}


def test_clock_mhz_lay_trung_vi_khong_phai_trung_binh():
    # Vài mẫu đầu rơi vào lúc governor chưa kịp lên - trung bình sẽ bị chúng kéo
    assert clock_mhz([315000000, 1575000000, 1566000000])["p50"] == 1566.0


def test_clock_mhz_khong_doc_duoc_thi_tra_khong():
    # Không có devfreq (chạy trên GPU rời, hoặc /sys không mount) - cột để trống
    # chứ không được bịa số, và tuyệt đối không được làm hỏng cả run
    assert clock_mhz([]) == {"p50": 0.0, "max": 0.0}


def test_clock_sampler_chay_va_dung_duoc(tmp_path):
    """Vòng đời thread thật, không mock.

    Bản đầu đặt Event tên `_stop`, đè lên method nội bộ của threading.Thread, và
    chết ngay ở join() - nhưng chỉ chết khi CHẠY THẬT, mọi test hàm thuần vẫn
    xanh. Test này là chỗ bắt được loại đó.
    """
    freq = tmp_path / "cur_freq"
    freq.write_text("1575000000\n")
    sampler = ClockSampler(freq)
    sampler.start()
    assert sampler.stop() == {"p50": 1575.0, "max": 1575.0}


def test_clock_sampler_duong_dan_khong_ton_tai_khong_giet_run(tmp_path):
    sampler = ClockSampler(tmp_path / "khong-co")
    sampler.start()
    assert sampler.stop() == {"p50": 0.0, "max": 0.0}


# --- warmup_slice: dòng CCU đầu tiên sau restart là rác nếu thiếu ------------


def test_warmup_slice_lay_dung_so_chunk_dau():
    assert warmup_slice(list(range(50)), 10) == list(range(10))


def test_warmup_slice_nhieu_hon_so_chunk_co_thi_lay_het():
    assert warmup_slice([1, 2, 3], 99) == [1, 2, 3]


def test_warmup_slice_khong_duong_thi_bo_warmup():
    # --warmup-chunks 0 là cách tắt warmup khi server đã chạy nóng sẵn
    assert warmup_slice([1, 2, 3], 0) == []


# --- model_config_summary: bảng phải tự khai mình đo cấu hình nào -----------
# Đã mất hai lần bench (30 phút) vì đo lại đúng cấu hình cũ mà không ai biết:
# Triton nạp config.pbtxt một lần lúc khởi động, sửa file mà không restart thì
# số cũ trở lại y hệt. Hỏi thẳng server đang chạy gì là hết cửa nhầm.

_CFG = {
    "name": "asr_streaming",
    "max_batch_size": 8,
    "instance_group": [{"count": 4, "kind": "KIND_GPU"}],
    "sequence_batching": {"oldest": {"max_candidate_sequences": 8}},
}


def test_model_config_summary_gom_du_bon_so_quyet_dinh():
    out = model_config_summary(_CFG)
    assert "asr_streaming" in out
    assert "4 x KIND_GPU" in out
    assert "max_batch_size 8" in out
    assert "max_candidate_sequences 8" in out


def test_model_config_summary_nhieu_instance_group():
    cfg = {**_CFG, "instance_group": [{"count": 2, "kind": "KIND_GPU"},
                                      {"count": 1, "kind": "KIND_CPU"}]}
    assert "2 x KIND_GPU + 1 x KIND_CPU" in model_config_summary(cfg)


def test_model_config_summary_khong_co_sequence_batching():
    cfg = {k: v for k, v in _CFG.items() if k != "sequence_batching"}
    assert "max_candidate_sequences" not in model_config_summary(cfg)
