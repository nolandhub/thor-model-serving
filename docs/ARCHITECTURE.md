# Kiến trúc

Tài liệu này mô tả hệ thống đang chạy trên Thor. README là hướng dẫn thao tác;
file này giải thích *vì sao* mọi thứ được xếp như vậy.

## 1. Hệ thống làm gì

Phục vụ nhận dạng tiếng nói tiếng Việt theo luồng (streaming ASR) qua Triton
Inference Server, kèm hệ giám sát. Model là Zipformer 30M RNN-T export sang
ONNX, chạy trên ONNX Runtime với CUDA Execution Provider.

Client mở một gRPC stream, đẩy từng chunk audio PCM 16 kHz vào, nhận về
transcript từng phần (partial) rồi transcript cuối khi kết thúc phiên.

Phạm vi v1: **chỉ ASR**. TTS và LLM chưa nằm trong repo này, nhưng cấu trúc đã
chừa chỗ (xem §7).

## 2. Ba ràng buộc định hình mọi thứ

Đọc phần này trước, vì gần như mọi quyết định lạ trong repo đều truy về đây.

| Ràng buộc | Hệ quả |
|---|---|
| **Thor là máy dùng chung** — pipeline sản xuất của nhóm khác giữ 8000/8001 và 8010/8011 | Phải né dải cổng đó; không được tự ý dừng/xoá container của người khác; dùng compose project riêng để `down` không chạm ai |
| **Thor không ra được internet** | Không `pip install`, không `docker pull`, không NTP. Trọng số và image phải chuyển tay từ máy dev |
| **Không có registry nội bộ** | Không có nơi lưu image đã build. Rollback chỉ còn dựa vào tag local, nên `IMAGE_TAG` phải đặt theo git sha khi deploy thật |

Ràng buộc thứ hai còn kéo theo một điều dễ quên: **đồng hồ Thor trôi** vì không
đồng bộ NTP được. Hiện lệch khoảng 1 giờ 38 phút, ảnh hưởng mọi mốc thời gian
trong log và mọi biểu đồ Grafana.

## 3. Cấu trúc container

Toàn bộ stack nằm trong một compose project tên `thor-voice`, dùng bridge
network do compose tự tạo (`thor-voice_default`). **Không** dùng
`network_mode: host`.

```
                        Thor (host)
   ┌──────────────────────────────────────────────────────┐
   │  publish, bind ${BIND_ADDR} (mặc định 127.0.0.1)     │
   │                                                       │
   │   :4000        :4001        :4002        :4003        │
   │     │            │            │            │          │
   ├─────┼────────────┼────────────┼────────────┼──────────┤
   │     ▼            ▼            ▼            ▼          │
   │  ┌──────────────────────────────────────────────┐    │
   │  │  bridge network: thor-voice_default          │    │
   │  │                                               │    │
   │  │   asr          prometheus    grafana          │    │
   │  │   :8000 http   :9090         :3000            │    │
   │  │   :8001 grpc      ▲                           │    │
   │  │   :8002 metrics ──┤                           │    │
   │  │                   │                           │    │
   │  │   node-exporter ──┘                           │    │
   │  │   :9100                                       │    │
   │  └──────────────────────────────────────────────┘    │
   └──────────────────────────────────────────────────────┘
```

### Ánh xạ cổng

| Host | → Container | Tầng | Lý do publish |
|---|---|---|---|
| `4000` | `asr:8000` | data plane | Triton HTTP — health, metadata |
| `4001` | `asr:8001` | data plane | Triton gRPC — đường streaming thật |
| `4002` | `prometheus:9090` | control plane | debug `/targets` khi scrape hỏng |
| `4003` | `grafana:3000` | control plane | dashboard |
| — | `asr:8002` | nội bộ | metrics; lộ ra ngoài là lộ throughput và pattern tải |
| — | `node-exporter:9100` | nội bộ | metrics host |

Nguyên tắc: **cổng bên trong container giữ nguyên giá trị chuẩn**. Mỗi container
có network namespace riêng nên không thể va chạm với ai. Chỉ cổng phía host mới
cần né.

### Giao tiếp nội bộ đi bằng tên service

- Prometheus scrape `asr:8002` và `node-exporter:9100`
- Grafana đọc `http://prometheus:9090`

Đây là điểm quan trọng nhất của lần refactor: trước đây Prometheus chạy
`network_mode: host` nên phải biết cổng ASR ở phía host, kéo theo cả một cơ chế
template hoá `prometheus.yml` để đồng bộ con số đó. Chuyển sang DNS của network
thì con số ấy biến mất khỏi cấu hình, và cơ chế template không còn lý do tồn
tại.

**Cạm bẫy đi kèm:** khi đã publish cổng, process bên trong container *phải* nghe
`0.0.0.0`. Vì vậy `--web.listen-address=127.0.0.1` của Prometheus và
`GF_SERVER_HTTP_ADDR` của Grafana đã bị gỡ — chúng chỉ đúng ở chế độ host
network. Việc không phơi ra LAN nay do tiền tố `${BIND_ADDR}` phía host đảm
nhiệm.

## 4. Một nguồn sự thật cho cấu hình

`.env` ở gốc repo (commit `.env.example`, gitignore `.env`) là nơi **duy nhất**
khai cổng và tên.

```
COMPOSE_PROJECT_NAME   tách namespace khỏi container của nhóm khác
BIND_ADDR              127.0.0.1 = chỉ gọi được từ Thor hoặc qua SSH tunnel
ASR_HTTP_PORT          4000
ASR_GRPC_PORT          4001
PROM_PORT              4002
GRAFANA_PORT           4003
IMAGE_TAG              đặt theo git sha khi deploy thật — đường rollback duy nhất
ASR_BASE_IMAGE         ghim theo digest khi cần bản dựng lặp lại được
```

Compose đọc `.env` tự động; `scripts/serving.sh` source nó; `tests/` đọc nó.

Trước refactor, cổng được khai ở sáu nơi và **đã lệch nhau** — code dùng
1000/1001/1002, README ghi 9000-9002, `deploy_monitoring.sh` hard-code 9090/3000.
`tests/test_config.py` giờ chốt để chuyện đó không tái diễn.

## 5. Vòng đời

`scripts/serving.sh` là entrypoint duy nhất, chỉ phụ thuộc bash + docker —
không cần Python, để `up`/`health` vẫn chạy được khi Thor thiếu package.

```
build                 build image ASR, forward proxy của host qua build-arg
up [asr|monitoring]   không tham số = cả hai
down [-v]             -v xoá luôn volume prometheus/grafana
logs [service]
health                probe từng endpoint + từng scrape job
help
```

Compose lo phần nặng: tạo network, restart policy, dọn container cũ, chờ
healthcheck (`up --wait`). Đó là lý do script co từ 113 dòng bash xuống còn
khoảng 60 — phần bị xoá đúng là phần từng hai lần sinh bug.

### Vì sao `health` không dùng `docker compose ps`

"Container đang chạy" và "hệ thống đang hoạt động" là hai chuyện khác nhau.
`health` hỏi thẳng từng endpoint, rồi hỏi Prometheus `up{job="..."}` cho **từng
job riêng**. Cách tìm chung kiểu "có target nào up không" sẽ báo xanh khi ASR
sống mà node-exporter chết — health check nói dối là loại hỏng tệ nhất.

### Healthcheck của ASR dùng python3, không dùng curl

Image base chắc chắn có `python3` (Triton python backend chạy bằng nó), không
chắc có `curl`. Healthcheck gọi binary không tồn tại thì `up --wait` treo tới
hết retries rồi báo unhealthy, mà log Triton lại sạch — rất khó lần ra.

ASR mất khoảng **52 giây** để `Healthy` (nạp 2 instance GPU + khởi tạo ORT).
`start_period: 30s` cộng 12 lần retry vừa đủ; thêm model sẽ phải nới.

## 6. Giám sát

Ba tầng metric:

| Tầng | Nguồn | Nội dung | Tình trạng |
|---|---|---|---|
| Ứng dụng | `serving/metrics.py` | RTF (histogram), CCU (gauge có TTL 60s) | đủ |
| Triton nội tại | `nv_*` | RPS, latency p50/95/99, queue depth, error rate | đủ |
| Host | `node_exporter` | đĩa, RAM, CPU, nhiệt độ thermal zone | đủ |
| GPU | NVML | utilization, memory, power | **không có** |

NVML không đọc được thông số Tegra — log Triton lặp lại `Unable to get power
limit ... value:0`. Nhiệt độ đã vá được vì `node_exporter` có collector
`thermal_zone` đọc thẳng sysfs; utilization và EMC bandwidth chỉ còn đường
`tegrastats`, chưa làm.

### Hai chi tiết dễ sai âm thầm

**`CCU_TTL_S` phải khớp ba nơi**: `serving/metrics.py`,
`max_sequence_idle_microseconds` trong `config.pbtxt`, và query trong dashboard.
Lệch bất kỳ đâu thì CCU hiển thị sai mà không báo lỗi gì. `tests/test_config.py`
canh cả ba.

**PromQL: `A + B` cho vector rỗng nếu một vế rỗng.** Repo chỉ có ASR nên mọi
panel tổng cộng thêm số liệu vLLM đều phải bọc `or vector(0)`, nếu không chúng
No data vĩnh viễn kể cả khi ASR chạy bình thường.

### Prometheus không tự nạp lại config

Đổi `config/prometheus.yml` rồi `up` lại **không** đủ: compose thấy định nghĩa
service không đổi (chỉ nội dung file mount đổi) nên không tạo lại container, và
Prometheus vẫn chạy config cũ trong bộ nhớ. Phải
`docker restart thor-voice-prometheus`.

## 7. Thêm model mới

`model_repository/` của Triton **vốn đã là kiến trúc plugin** — thêm model là
thêm một thư mục `<tên>/config.pbtxt` + `1/model.py`, không sửa dòng nào ở chỗ
khác. Cùng một `tritonserver` load cả repository.

Contract cần theo:

- import `serving/metrics.py`, phát metric kèm label `model` và `model_instance`
  — **không** dùng label `instance`, Prometheus chiếm riêng tên đó cho địa chỉ
  target lúc scrape và sẽ ghi đè mất giá trị thật
- dashboard query theo label `model` nên model mới **tự hiện lên**, không sửa JSON
- `serving/metrics.py` đã có sẵn `TTS_RTF_BUCKETS` cạnh `ASR_RTF_BUCKETS` — hai
  thang lệch hẳn một bậc nên không dùng chung buckets được
- chỉ khi model cần image khác (ví dụ TTS cần `piper_phonemize`) mới thêm một
  service vào `compose.yaml`; profiles đã sẵn cho việc đó

Cố tình **không** thêm lớp trừu tượng nào. Dựng khung lúc mới có một model là
đoán mò.

## 8. Cây thư mục

Nguyên tắc: **mỗi file nằm cạnh thứ nó phục vụ**, thay vì gom vào một thư mục
`tools/` mà ai cũng phải đoán "cái này là tool hay config".

```
compose.yaml              cả 4 service, hai profile
.env.example              nguồn sự thật duy nhất cho cổng
scripts/serving.sh        vòng đời container
docker/Dockerfile.thor    base asr-triton, không cài thêm gì
config/
  prometheus.yml          scrape theo tên service
  grafana/
    provisioning/         datasource + dashboard provider
    dashboards/           JSON, trong đó node-exporter-full.json là 1860
    build_dashboard.py    sinh ra hai JSON kia — nằm cạnh output của nó
serving/metrics.py        contract dùng chung, 2 nơi import
model_repository/         copy nguyên từ triton-voice-serving, không đổi
client/                   reference implementation cho người tích hợp
tests/
  test_config.py          bất biến cấu hình, chạy được ở máy dev
  test_parity.sh          bằng chứng chạy đúng, cần GPU + weights
  dump_golden_asr.py      sinh fixture — nằm cạnh fixture nó sinh
```

`serving/` ở gốc chứ không nằm trong `model_repository/` vì cả `model.py` (chạy
trong container) lẫn `build_dashboard.py` (chạy ở máy dev) đều import nó. Đây là
thư viện dùng chung thật sự.

## 9. Giới hạn đã biết

- **Chưa có alert nào.** Dashboard là thứ phải chủ động mở ra nhìn. Đây là
  khoảng cách lớn nhất giữa hiện trạng và chữ "production".
- **Đồng hồ Thor lệch**, chưa sửa dứt điểm.
- **Data plane đang bind loopback.** Mở ra LAN cho đồng nghiệp gọi cần chốt
  phạm vi trước: Triton không có authentication, và `docker -p` chèn rule vào
  chain `DOCKER-USER`, đi vòng qua `firewalld`.
- **Chưa đo độ trễ tới partial đầu tiên** — đây mới là con số người dùng cuối
  cảm nhận ở voice agent; hiện chỉ đo thời gian suy luận mỗi chunk.
- **Coupling với `asr-triton:latest`** — image do nhóm khác kiểm soát, họ
  rebuild là bản dựng của mình đổi âm thầm.
