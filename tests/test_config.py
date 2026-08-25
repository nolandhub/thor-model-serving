# ABOUTME: Chốt các bất biến trải trên nhiều file - lệch một nơi là dashboard sai âm thầm, không báo lỗi
# ABOUTME: Chạy: pytest tests/test_config.py (không cần Thor, không cần GPU)

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serving.metrics import CCU_TTL_S  # noqa: E402

CONFIG_PBTXT = ROOT / "model_repository" / "asr_streaming" / "config.pbtxt"
DASH_DIR = ROOT / "config" / "grafana" / "dashboards"
BUILDER = ROOT / "config" / "grafana" / "build_dashboard.py"
PROMETHEUS_YML = ROOT / "config" / "prometheus.yml"
DATASOURCE_YML = ROOT / "config" / "grafana" / "provisioning" / "datasources" / "datasource.yml"
COMPOSE = ROOT / "compose.yaml"
ENV_EXAMPLE = ROOT / ".env.example"


# --- Bất biến CCU_TTL_S: ba nơi phải khớp -----------------------------------
# serving/metrics.py dùng con số này để hiểu "phiên còn sống"; Triton dùng nó
# để quyết định khi nào dọn sequence; dashboard dùng nó để bỏ qua instance im
# lặng. Lệch nhau thì CCU hiển thị sai mà không có triệu chứng nào khác.


def test_ccu_ttl_khop_max_sequence_idle_trong_config_pbtxt():
    text = CONFIG_PBTXT.read_text()
    m = re.search(r"max_sequence_idle_microseconds:\s*(\d+)", text)
    assert m, "không tìm thấy max_sequence_idle_microseconds trong config.pbtxt"
    assert int(m.group(1)) / 1e6 == CCU_TTL_S


def test_ccu_ttl_xuat_hien_trong_query_dashboard():
    text = (DASH_DIR / "voice-serving.json").read_text()
    assert f"bool {CCU_TTL_S:g}" in text, (
        f"query CCU trong dashboard không dùng {CCU_TTL_S:g}s - "
        "chạy lại build_dashboard.py"
    )


# --- Dashboard JSON phải khớp builder sinh ra nó -----------------------------


@pytest.mark.parametrize("board,fname", [("voice", "voice-serving.json"), ("triton", "triton.json")])
def test_dashboard_json_khop_output_cua_builder(board, fname):
    out = subprocess.run(
        [sys.executable, str(BUILDER), "--board", board, "--stdout"],
        capture_output=True, text=True, check=True, cwd=ROOT,
    )
    assert json.loads(out.stdout) == json.loads((DASH_DIR / fname).read_text()), (
        f"{fname} lệch với build_dashboard.py - chạy lại builder và commit"
    )


# --- Giao tiếp nội bộ đi bằng DNS service name, không bằng cổng host ---------
# Đây chính là lớp lỗi đã xảy ra: cổng khai ở sáu nơi rồi drift. Trong compose
# network thì tên service là cố định, không có gì để drift.


def test_prometheus_scrape_asr_qua_service_name():
    text = PROMETHEUS_YML.read_text()
    assert "asr:8002" in text, "prometheus phải scrape asr:8002 trong network, không phải localhost:<cổng host>"
    assert "localhost" not in text


def test_grafana_tro_prometheus_qua_service_name():
    text = DATASOURCE_YML.read_text()
    assert "http://prometheus:9090" in text
    assert "localhost" not in text


# --- Cổng chỉ được khai đúng một nơi: .env ----------------------------------


def test_compose_khong_hardcode_cong_host():
    text = COMPOSE.read_text()
    ports = re.findall(r"^\s*-\s*\"?\$\{BIND_ADDR[^\"]*\"?$", text, re.M)
    assert ports, "không tìm thấy khai báo ports dùng ${BIND_ADDR}"
    for line in ports:
        assert "${" in line and "}" in line, f"cổng hard-code trong compose.yaml: {line.strip()}"


def test_env_example_co_du_bien_compose_dung():
    env_keys = {
        line.split("=", 1)[0].strip()
        for line in ENV_EXAMPLE.read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    used = set(re.findall(r"\$\{([A-Z_]+)(?::-[^}]*)?\}", COMPOSE.read_text()))
    # biến có giá trị mặc định trong compose thì không bắt buộc nằm trong .env
    required = {v for v in used if f"${{{v}:-" not in COMPOSE.read_text()}
    assert required <= env_keys, f"thiếu trong .env.example: {sorted(required - env_keys)}"


def test_metrics_8002_khong_publish_ra_host():
    text = COMPOSE.read_text()
    assert ":8002" not in re.sub(r"#.*", "", text), (
        "cổng metrics 8002 không được publish ra host - chỉ prometheus trong network gọi"
    )


# --- serving.sh phải chạy được ở các nhánh không cần docker ------------------
# `set -u` + `${1:-help}` trong case là cái bẫy kinh điển: case khớp nhánh
# help nhưng dòng gán ngay sau đó vẫn đọc $1 đang unset.

SERVING_SH = ROOT / "scripts" / "serving.sh"


def _run_sh(*args):
    return subprocess.run(
        ["bash", str(SERVING_SH), *args],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_serving_sh_khong_tham_so_in_help():
    r = _run_sh()
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert "build" in r.stdout and "health" in r.stdout


def test_serving_sh_help():
    r = _run_sh("help")
    assert r.returncode == 0, f"stderr: {r.stderr}"


def test_serving_sh_lenh_la_bao_loi_ro_rang():
    r = _run_sh("khong-ton-tai")
    assert r.returncode != 0
    assert "khong-ton-tai" in r.stderr


def test_log_runtime_bang_tieng_anh_ascii():
    """Log lúc deploy phải là tiếng Anh - dán đi hỏi ai cũng đọc được.

    Comment trong script vẫn tiếng Việt, chỉ output ra người vận hành mới bị
    ràng buộc. Kiểm bằng ASCII vì tiếng Việt luôn có dấu.
    """
    for r in (_run_sh("help"), _run_sh("khong-ton-tai")):
        out = r.stdout + r.stderr
        assert out.isascii(), f"log có ký tự ngoài ASCII: {[c for c in out if not c.isascii()]}"
