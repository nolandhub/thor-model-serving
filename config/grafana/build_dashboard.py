#!/usr/bin/env python3
# ABOUTME: Sinh voice-serving.json và triton.json từ bảng panel - JSON viết tay không review nổi
# ABOUTME: Chạy: python3 docker/monitoring/build_dashboard.py [--board voice|triton|all] [--stdout]

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serving.metrics import CCU_TTL_S  # noqa: E402

DS = {"type": "prometheus", "uid": "prometheus"}
DASH_DIR = Path(__file__).parent / "dashboards"

# Instance nào không được chạm trong CCU_TTL_S giây thì nhân 0. Mô phỏng đúng
# cái _sweep() sẽ làm nếu nó được chạy - xem spec §6.
#
# Join theo model_instance chứ không phải instance: label "instance" bị
# Prometheus chiếm riêng cho địa chỉ target lúc scrape, nên serving/metrics.py
# phát label tự phát dưới tên "model_instance" để không đụng.
def opt(expr: str) -> str:
    """Bọc một vế cộng để nó thành 0 thay vì làm rỗng cả biểu thức.

    PromQL: `A + B` khớp theo label, một vế rỗng thì kết quả RỖNG chứ không
    phải A. Repo này chỉ triển khai ASR - mọi panel tổng cộng thêm vLLM/TTS
    sẽ No data vĩnh viễn nếu không bọc.
    """
    return f"({expr} or vector(0))"


def ccu(selector: str) -> str:
    return (
        f"sum(voice_ccu{selector} * on(model, model_instance) "
        f"(time() - voice_ccu_updated_at < bool {CCU_TTL_S:g}))"
    )


# sum() bắt buộc quanh failure: nó có label `reason`, không bọc thì phép cộng
# với success lệch label và Prometheus trả rỗng.
def error_rate(model: str) -> str:
    f = f'sum(rate(nv_inference_request_failure{{model="{model}"}}[1m]))'
    s = f'sum(rate(nv_inference_request_success{{model="{model}"}}[1m]))'
    return f"{f} / clamp_min({s} + {f}, 1e-9)"


def triton_rows(model: str) -> list:
    sel = f'{{model="{model}"}}'
    q = lambda p: f'nv_inference_request_summary_us{{model="{model}",quantile="{p}"}} / 1000'
    return [
        (f"{model} · RPS", [f"rate(nv_inference_request_success{sel}[1m])"], "reqps"),
        (f"{model} · Latency p50/p95/p99", [q("0.5"), q("0.95"), q("0.99")], "ms"),
        (f"{model} · CCU", [ccu(sel)], "short"),
        (f"{model} · Queue depth", [f"nv_inference_pending_request_count{sel}"], "short"),
        (
            f"{model} · RTF p50/p95/p99",
            [
                f"histogram_quantile({p}, sum by(le) (rate(voice_rtf_bucket{sel}[5m])))"
                for p in ("0.5", "0.95", "0.99")
            ],
            "none",
        ),
        (f"{model} · Error rate", [error_rate(model)], "percentunit"),
    ]


LLM_E2E = "rate(vllm:e2e_request_latency_seconds_bucket[5m])"

ROWS = [
    (
        "OVERVIEW",
        [
            (
                "RPS tổng",
                [
                    f"{opt('sum(rate(nv_inference_request_success[1m]))')}"
                    f" + {opt('sum(rate(vllm:request_success_total[1m]))')}"
                ],
                "reqps",
            ),
            (
                "Success rate (Triton)",
                [
                    "sum(rate(nv_inference_request_success[1m]))"
                    " / clamp_min(sum(rate(nv_inference_request_success[1m]))"
                    " + sum(rate(nv_inference_request_failure[1m])), 1e-9)"
                ],
                "percentunit",
            ),
            (
                "Error rate (Triton)",
                [
                    "sum(rate(nv_inference_request_failure[1m]))"
                    " / clamp_min(sum(rate(nv_inference_request_success[1m]))"
                    " + sum(rate(nv_inference_request_failure[1m])), 1e-9)"
                ],
                "percentunit",
            ),
            (
                "CCU tổng",
                [f"{opt(ccu(''))} + {opt('sum(vllm:num_requests_running)')}"],
                "short",
            ),
        ],
    ),
    ("ASR_BLS", triton_rows("asr_bls")),
    ("TTS", triton_rows("tts")),
    (
        "LLM (vLLM)",
        [
            ("llm · RPS", ["sum(rate(vllm:request_success_total[1m]))"], "reqps"),
            (
                "llm · Latency p50/p95/p99",
                [f"histogram_quantile({p}, sum by(le) ({LLM_E2E}))" for p in ("0.5", "0.95", "0.99")],
                "s",
            ),
            ("llm · CCU", ["vllm:num_requests_running"], "short"),
            ("llm · Queue depth", ["vllm:num_requests_waiting"], "short"),
            (
                "llm · TTFT p50/p95/p99",
                [
                    f"histogram_quantile({p}, sum by(le) "
                    f"(rate(vllm:time_to_first_token_seconds_bucket[5m])))"
                    for p in ("0.5", "0.95", "0.99")
                ],
                "s",
            ),
            (
                "llm · TPOT p50/p95/p99",
                [
                    f"histogram_quantile({p}, sum by(le) "
                    f"(rate(vllm:inter_token_latency_seconds_bucket[5m])))"
                    for p in ("0.5", "0.95", "0.99")
                ],
                "s",
            ),
            (
                "llm · Error rate (proxy: abort)",
                [
                    'sum(rate(vllm:request_success_total{finished_reason="abort"}[1m]))'
                    " / clamp_min(sum(rate(vllm:request_success_total[1m])), 1e-9)"
                ],
                "percentunit",
            ),
        ],
    ),
    (
        "GPU (chung cả máy)",
        [
            ("GPU Utilization", ["nv_gpu_utilization * 100"], "percent"),
            (
                "GPU Memory usage",
                ["nv_gpu_memory_used_bytes / nv_gpu_memory_total_bytes * 100"],
                "percent",
            ),
            ("GPU Memory used", ["nv_gpu_memory_used_bytes"], "bytes"),
        ],
    ),
    (
        # Trên Jetson, NVML không đọc được power/thermal (log Triton báo "Unable
        # to get power limit ... value:0"). node_exporter đọc thẳng sysfs Tegra
        # nên đây là nguồn nhiệt độ duy nhất hiện có - mà throttle nhiệt lại là
        # chế độ hỏng hay gặp nhất của thiết bị biên.
        "HOST (Thor)",
        [
            (
                "Nhiệt độ theo thermal zone",
                ["node_thermal_zone_temp"],
                "celsius",
            ),
            (
                "Đĩa còn trống",
                ['100 * node_filesystem_avail_bytes{mountpoint="/"}'
                 ' / node_filesystem_size_bytes{mountpoint="/"}'],
                "percent",
            ),
            (
                "RAM đang dùng",
                ["100 * (1 - node_memory_MemAvailable_bytes"
                 " / node_memory_MemTotal_bytes)"],
                "percent",
            ),
            (
                "CPU đang dùng",
                ['100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m])))'],
                "percent",
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Board "Triton": nội tại server, không phải view sản phẩm.
# ---------------------------------------------------------------------------

# request = queue + compute_input + compute_infer + compute_output. Bốn chặng
# này là câu trả lời cho "chậm vì xếp hàng hay vì model", thứ mà p50 request
# một mình không nói được.
STAGES = ("queue", "compute_input", "compute_infer", "compute_output")


def stage_ms(stage: str, model: str = "") -> tuple:
    """Trung bình mỗi request của một chặng, tính bằng ms.

    Dùng cặp _sum/_count chứ không phải nhánh quantile: quantile của summary
    tính trên sliding window nên trả Nan lúc không có traffic, còn _sum/_count
    là counter thường. Panel breakdown phải đọc được cả lúc hệ rảnh.
    """
    fam = f"nv_inference_{stage}_summary_us"
    sel = f'{{model="{model}"}}' if model else ""
    return (
        f"rate({fam}_sum{sel}[1m]) / clamp_min(rate({fam}_count{sel}[1m]), 1e-9) / 1000",
        stage,
    )


def error_rate_by_model() -> str:
    # `sum(...) by (model)` chứ không phải `sum by(model)(...)`: hai dạng tương
    # đương với Prometheus, nhưng chỉ dạng này giữ được chuỗi "sum(" ngay trước
    # tên metric - thứ test_moi_phep_tinh_failure_deu_boc_sum bám vào.
    f = "sum(rate(nv_inference_request_failure[1m])) by (model)"
    s = "sum(rate(nv_inference_request_success[1m])) by (model)"
    return f"{f} / clamp_min({s} + {f}, 1e-9)"


def breakdown(model: str) -> tuple:
    return (
        f"{model} · Latency breakdown",
        [stage_ms(stage, model) for stage in STAGES],
        "ms",
        {"stack": True},
    )


def summary_q(fam: str) -> list:
    return [
        (f'nv_inference_{fam}_summary_us{{quantile="{q}"}} / 1000',
         "{{model}} p" + q)
        for q in ("0.5", "0.95", "0.99")
    ]


SERVER_ROWS = [
    (
        "HEALTH",
        [
            # Không có ô này thì Triton chết và cả board cùng hiện "No data",
            # không chỗ nào phân biệt được "hệ rảnh" với "hệ tắt".
            ("Target UP", [('up{job="triton"}', "{{job}}")], "short",
             {"type": "stat", "h": 4,
              "thresholds": [("red", None), ("green", 1)]}),
            ("Error rate", [error_rate_by_model()], "percentunit",
             {"type": "stat", "h": 4,
              "thresholds": [("green", None), ("orange", 0.01), ("red", 0.05)]}),
            ("Model load", [("nv_model_load_duration_secs", "{{model}}")], "s",
             {"type": "stat", "h": 4}),
        ],
    ),
    (
        "LATENCY",
        [
            ("Request p50/p95/p99", summary_q("request"), "ms"),
            # Trống lúc hệ rảnh là ĐÚNG - quantile của summary trả Nan khi
            # sliding window rỗng. Ô kế bên là bản trung bình, luôn có số.
            ("Queue p50/p95/p99", summary_q("queue"), "ms"),
            ("Queue time trung bình", [stage_ms("queue")], "ms"),
            ("Queue depth", [("nv_inference_pending_request_count", "{{model}}")],
             "short"),
            breakdown("asr_bls"),
            breakdown("tts"),
        ],
    ),
    (
        "THROUGHPUT",
        [
            ("RPS theo model",
             [("sum(rate(nv_inference_request_success[1m])) by (model)", "{{model}}")],
             "reqps"),
            ("Inference vs execution",
             [("sum(rate(nv_inference_count[1m])) by (model)", "inference {{model}}"),
              ("sum(rate(nv_inference_exec_count[1m])) by (model)", "execution {{model}}")],
             "reqps"),
            # inference_count / exec_count = batch size trung bình. Bằng 1 nghĩa
            # là scheduler không gộp được gì, dù config khai max_candidate_sequences 8.
            ("Batch size trung bình",
             [("sum(rate(nv_inference_count[1m])) by (model)"
               " / clamp_min(sum(rate(nv_inference_exec_count[1m])) by (model), 1e-9)",
               "{{model}}")],
             "short"),
        ],
    ),
    (
        "RESOURCE",
        [
            ("GPU utilization", [("nv_gpu_utilization * 100", "GPU {{gpu_uuid}}")],
             "percent"),
            # Thor dùng LPDDR chung cho CPU và GPU: không có VRAM rời, NVML
            # không có "GPU memory" để báo và nv_gpu_memory_* không tồn tại -
            # panel cũ query nó nên vĩnh viễn No data. Nguồn thay thế là
            # node-exporter, vì trên máy này bộ nhớ GPU tiêu CHÍNH LÀ RAM.
            ("Memory (unified CPU+GPU)",
             [("(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes)"
               " / node_memory_MemTotal_bytes * 100", "used")],
             "percent",
             {"desc": "Thor dùng LPDDR chung cho CPU và GPU - không có VRAM rời, nên "
                      "NVML không có \"GPU memory\" để báo và nv_gpu_memory_* không tồn "
                      "tại. Nguồn ở đây là node-exporter: bộ nhớ GPU tiêu CHÍNH LÀ RAM hệ "
                      "thống, nên số này là 'máy còn bao nhiêu chỗ', KHÔNG tách riêng được "
                      "phần GPU. Muốn tách theo tiến trình thì đọc "
                      "/sys/kernel/debug/nvmap/iovmm/clients."}),
            # Không vẽ nv_gpu_power_limit kèm: trên 3050 Laptop này DCGM đọc
            # được 0 W (nvidia-smi cũng trả power.limit [N/A], chỉ có
            # enforced.power.limit 60 W). Một đường thẳng 0 W không nói gì mà
            # còn kéo tụt trục y của đường usage.
            ("GPU power", [("nv_gpu_power_usage", "usage")], "watt"),
            # fbank và greedy search chạy trên CPU trong Python backend, nên
            # CPU bão hoà cũng làm ASR chậm dù GPU còn rỗi.
            ("CPU utilization", [("nv_cpu_utilization * 100", "CPU")], "percent"),
            ("CPU memory",
             [("nv_cpu_memory_used_bytes / nv_cpu_memory_total_bytes * 100", "RAM")],
             "percent"),
        ],
    ),
]

# Legend mặc định: nối ba label lại, label nào không có thì Grafana bỏ trống.
# Panel nào cần chữ khác thì viết expr dạng (expr, legend).
LEGEND = "{{model}}{{instance}}{{quantile}}"


def _panel(spec, pid: int, x: int, y: int, h: int) -> dict:
    """Một ô. spec là (title, exprs, unit) hoặc (title, exprs, unit, opts).

    opts nhận: type (mặc định timeseries), h, stack, thresholds, desc.
    """
    title, exprs, unit = spec[:3]
    opts = spec[3] if len(spec) > 3 else {}

    defaults = {"unit": unit}
    if opts.get("stack"):
        # Breakdown chỉ đọc được khi chồng lên nhau - bốn đường rời rạc thì mắt
        # phải tự cộng, mà tổng mới là thứ so được với p50 request.
        defaults["custom"] = {
            "stacking": {"mode": "normal", "group": "A"}, "fillOpacity": 60,
        }
    if "thresholds" in opts:
        defaults["thresholds"] = {
            "mode": "absolute",
            "steps": [{"color": c, "value": v} for c, v in opts["thresholds"]],
        }

    targets = []
    for n, e in enumerate(exprs):
        expr, legend = e if isinstance(e, tuple) else (e, LEGEND)
        targets.append({"datasource": DS, "expr": expr, "refId": chr(65 + n),
                        "legendFormat": legend})

    panel = {
        "type": opts.get("type", "timeseries"), "title": title, "id": pid,
        "datasource": DS,
        "gridPos": {"h": h, "w": 8, "x": x, "y": y},
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "targets": targets,
    }
    if opts.get("desc"):
        panel["description"] = opts["desc"]
    if panel["type"] == "stat":
        # Không khai thì Grafana tự chọn calc, đổi giữa các bản. Chốt lại.
        panel["options"] = {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "background",
        }
    return panel


def build(rows: list, uid: str, title: str, tags: list) -> dict:
    panels, pid, y = [], 1, 0
    for row_title, specs in rows:
        panels.append(
            {
                "type": "row", "title": row_title, "id": pid,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
                "collapsed": False, "panels": [],
            }
        )
        pid += 1
        y += 1
        # Chiều cao đồng nhất trong một row: hàng stat lùn hơn hàng timeseries,
        # nhưng trộn hai chiều cao trong cùng một row thì lưới lệch.
        h = max((s[3].get("h", 8) if len(s) > 3 else 8) for s in specs)
        for i, spec in enumerate(specs):
            panels.append(_panel(spec, pid, (i % 3) * 8, y + (i // 3) * h, h))
            pid += 1
        y += ((len(specs) + 2) // 3) * h
    return {
        "uid": uid,
        "title": title,
        "tags": tags,
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "10s",
        "time": {"from": "now-30m", "to": "now"},
        "panels": panels,
    }


BOARDS = {
    "voice": ("voice-serving.json", ROWS, "voice-serving", "Voice Serving",
              ["triton", "vllm", "voice"]),
    "triton": ("triton.json", SERVER_ROWS, "triton", "Triton",
               ["triton", "server"]),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="print to stdout instead of writing files")
    ap.add_argument("--board", choices=[*BOARDS, "all"], default="all")
    args = ap.parse_args()

    names = list(BOARDS) if args.board == "all" else [args.board]
    if args.stdout and len(names) > 1:
        ap.error("--stdout requires --board because there is more than one board")

    for name in names:
        filename, rows, uid, title, tags = BOARDS[name]
        text = json.dumps(build(rows, uid, title, tags), indent=2, ensure_ascii=False) + "\n"
        if args.stdout:
            print(text, end="")
        else:
            DASH_DIR.mkdir(parents=True, exist_ok=True)
            (DASH_DIR / filename).write_text(text)
            print(f"wrote {DASH_DIR / filename}")
