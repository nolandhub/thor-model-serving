# Refactor thor-voice-serving sang compose + docker network

Ngày: 2026-08-25
Trạng thái: đã chốt thiết kế, chờ viết implementation plan

## 1. Vì sao

Ba vấn đề cụ thể, không phải chuyện thẩm mỹ.

**Hai đường deploy hỏng ở thời điểm hiện tại.** `deploy_monitoring.sh` render
`prometheus.yml` bằng `sed` từ `prometheus.yml.template` — file đó đã bị xoá ở
working tree, script chạy là fail. `deploy_asr.sh` và README trỏ tới
`scripts/fetch_models.sh`, cũng đã bị xoá.

**Cổng được khai ở sáu nơi và đã drift.** `deploy_asr.sh` mặc định
1000/1001/1002; `parity_check.sh` mặc định 1001; `deploy_monitoring.sh`
hard-code 9090/3000; `prometheus.yml`; `datasource.yml`; README ghi 9000-9002.
Comment đầu `deploy_asr.sh` cũng ghi 9000-9002 trong khi code là 1000. Ba con
số khác nhau cho cùng một thứ.

**Hai cơ chế cho cùng một loại việc.** ASR chạy bằng `docker run` thủ công
trong bash, monitoring chạy bằng `docker compose`. Hệ quả là `deploy_asr.sh`
phải tự tay làm những gì compose cho không: dọn container cũ, restart policy,
port guard, vòng chờ health. Git log cho thấy đã phải sửa bug ở chính đoạn đó
hai lần (b4d974d, 281ac8c).

Ngoài ra Prometheus và Grafana đang dùng `network_mode: host` — trái với góp ý
của mentor, và là nguyên nhân khiến Prometheus phải biết cổng ASR ở phía host.

## 2. Kiến trúc mạng

Compose project `thor-voice` tự tạo một bridge network riêng
(`thor-voice_default`). Không dùng `network_mode: host`, không cần
`docker network create` thủ công, không cần `external: true` — mọi service nằm
trong cùng một compose project nên network là ngầm định.

Cổng bên trong container giữ nguyên giá trị chuẩn: mỗi container có network
namespace riêng nên không có va chạm. Chỉ cổng phía host mới cần né.

| Host (bind `127.0.0.1`) | → Container | Lý do expose |
|---|---|---|
| `4000` | `asr:8000` | Triton HTTP — client và health check |
| `4001` | `asr:8001` | Triton gRPC — đường ASR streaming |
| `4002` | `prometheus:9090` | debug `/targets`; thiếu nó thì scrape hỏng phải `docker exec` vào mò |
| `4003` | `grafana:3000` | dashboard |
| (không publish) | `asr:8002` | metrics — chỉ Prometheus trong cùng network gọi |

Giao tiếp nội bộ theo DNS service name:

- `config/prometheus.yml` → target `asr:8002`
- `config/grafana/provisioning/datasources/datasource.yml` → `http://prometheus:9090`

**Bắt buộc kèm theo:** bỏ `--web.listen-address=127.0.0.1` của Prometheus và
`GF_SERVER_HTTP_ADDR=127.0.0.1` của Grafana. Ở chế độ `network_mode: host`
chúng đúng; khi đã publish port thì process phải nghe `0.0.0.0` *bên trong*
container, nếu không port mapping trỏ vào chỗ không ai nghe. Việc không phơi ra
LAN do tiền tố `127.0.0.1:` phía host đảm nhiệm — tương đương về an toàn, và
vẫn cần thiết vì Grafana bật anonymous admin.

`4000/4001` cũng bind `127.0.0.1`: ASR chỉ gọi được từ chính Thor hoặc qua SSH
tunnel. Không có client LAN nào gọi thẳng vào Thor.

Truy cập từ máy dev:

```bash
ssh -L 4002:localhost:4002 -L 4003:localhost:4003 <user>@thor
```

## 3. Một nguồn sự thật cho cấu hình

`.env` ở root, commit `.env.example`, gitignore `.env`:

```
COMPOSE_PROJECT_NAME=thor-voice
ASR_HTTP_PORT=4000
ASR_GRPC_PORT=4001
PROM_PORT=4002
GRAFANA_PORT=4003
IMAGE_TAG=dev
ASR_BASE_IMAGE=asr-triton:latest
```

Compose đọc `.env` tự động; `scripts/serving.sh` source nó; `tests/` đọc nó.
Sáu nơi khai cổng còn lại một.

Hai hằng số biến mất khỏi cấu hình vì đã chuyển sang DNS name: cổng metrics ASR
và cổng Prometheus không còn xuất hiện ở phía host. Đây là lý do cơ chế
`prometheus.yml.template` + `sed` không còn cần — nó sinh ra chỉ để đồng bộ một
con số nay không còn tồn tại. **Xoá `prometheus.yml.template` và bước render.**

## 4. Cây thư mục đích

```
.env.example
compose.yaml
README.md
docker/Dockerfile.thor
config/
  prometheus.yml
  grafana/
    provisioning/datasources/datasource.yml
    provisioning/dashboards/voice.yml
    dashboards/{voice-serving.json,triton.json}
    build_dashboard.py
serving/{__init__.py,metrics.py}
model_repository/asr_streaming/{config.pbtxt,1/}
client/{asr_streaming_client.py,common.py}
scripts/serving.sh
tests/
  assets/
  test_config.py
  test_parity.sh
  dump_golden_asr.py
```

Nguyên tắc thay cho một thư mục `tools/`: **mỗi file nằm cạnh thứ nó phục vụ.**
`build_dashboard.py` cạnh JSON nó sinh ra; `dump_golden_asr.py` cạnh fixture nó
sinh ra. Ít thư mục hơn, và không ai phải phân vân "cái này là tool hay config".

`serving/` giữ nguyên ở root: cả `model_repository/asr_streaming/1/model.py`
(chạy trong container) lẫn `config/grafana/build_dashboard.py` (chạy ở máy dev)
đều import nó. Đây là thư viện dùng chung thật sự.

Ba file bị xoá / không giữ:

- `scripts/fetch_models.sh` — **bỏ**. Thor bị security chặn không ra internet
  được, script tải từ HuggingFace không chạy được ở đó. Weights chuyển từ máy
  dev bằng một lệnh `rsync`, ghi trong README.
- `docker/monitoring/prometheus.yml.template` — **bỏ**, xem §3.
- `.github/workflows/ci.yml` — **không tạo**. Tinh thần CI/CD nằm ở cấu hình một
  nguồn, deploy một lệnh, và test chạy được — không ở file workflow.

### Chi tiết di chuyển cần chú ý

`build_dashboard.py` hiện làm `sys.path.insert(0, parents[2])` để import
`serving.metrics`. Chuyển từ `docker/monitoring/` sang `config/grafana/` không
đổi độ sâu (đều 2 cấp so với root) nên `parents[2]` vẫn đúng — nhưng phải kiểm
lại, và `DASH_DIR = Path(__file__).parent / "grafana" / "dashboards"` thì
**phải sửa** thành `Path(__file__).parent / "dashboards"`.

Dùng `git mv` để giữ lịch sử rename.

## 5. compose.yaml

Ba service, hai profile:

- `asr` — profile `asr`
- `prometheus`, `grafana` — profile `monitoring`

Yêu cầu từng service:

**asr**
- `build`: context `./docker`, dockerfile `Dockerfile.thor`, `args` forward
  `http_proxy/https_proxy/no_proxy` (và biến hoa) từ môi trường host — apt/pip
  trong container không tự thấy proxy công ty. Giữ nguyên cơ chế đã chạy được.
- `image: thor-voice-serving-asr:${IMAGE_TAG}`
- `container_name: thor-asr-triton`
- GPU: `gpus: all`
- volumes: `../model_repository:/models`, `../serving:/opt/serving/serving:ro`
- command: `tritonserver` với `--model-repository=/models` và hai
  `--metrics-config` như hiện tại (giữ nguyên, không đổi)
- ports: `127.0.0.1:${ASR_HTTP_PORT}:8000`, `127.0.0.1:${ASR_GRPC_PORT}:8001`
- `healthcheck` gọi `/v2/health/ready` trên `localhost:8000` — thay cho vòng
  `curl` 30 lần trong bash, và cho phép `up --wait`
- `restart: unless-stopped`
- logging json-file `max-size: 50m`, `max-file: 3`

**prometheus**
- `image: prom/prometheus:latest` — giữ `:latest`, KHÔNG ghim version:
  registry-1.docker.io bị chặn từ Thor, `:latest` là bản duy nhất có sẵn trong
  cache local. Ghim version sẽ pull fail. Lý do này phải giữ lại thành comment
  trong compose.yaml.
- volumes: `./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro`,
  named volume `prometheus-data`
- ports: `127.0.0.1:${PROM_PORT}:9090`
- bỏ `--web.listen-address`
- restart + logging như trên

**grafana**
- `image: grafana/grafana:latest` — cùng lý do
- volumes: provisioning + dashboards mount `:ro`, named volume `grafana-data`
- ports: `127.0.0.1:${GRAFANA_PORT}:3000`
- env: giữ `GF_AUTH_ANONYMOUS_*` và `GF_AUTH_DISABLE_LOGIN_FORM`; **bỏ**
  `GF_SERVER_HTTP_ADDR`
- restart + logging như trên

## 6. scripts/serving.sh

Một file dispatcher, thuần vòng đời container. Chỉ phụ thuộc bash + docker —
không phụ thuộc Python, để `up`/`health` vẫn chạy được khi Thor thiếu package.

```
./scripts/serving.sh build
./scripts/serving.sh up [asr|monitoring]    # không tham số = cả hai
./scripts/serving.sh down [-v]              # -v xoá luôn volume
./scripts/serving.sh logs [service]
./scripts/serving.sh health
./scripts/serving.sh help
```

Vì sao một file thay vì `build.sh`/`up.sh`/`down.sh` riêng: phần đầu mỗi script
sẽ y hệt nhau (resolve `$ROOT`, load `.env`, trỏ `compose.yaml`). Tách ra là
nhân bản đoạn đó nhiều lần — đúng kiểu trùng lặp đã sinh ra vụ drift cổng hiện
tại. Một file thì `help` liệt kê được toàn bộ.

Thân mỗi lệnh mỏng, compose làm phần nặng:

```bash
cmd_up()    { compose ${1:+--profile "$1"} up -d --wait; }
cmd_down()  { compose down "$@"; }
cmd_logs()  { compose logs -f "$@"; }
cmd_build() { compose build; }
```

`health` là lệnh duy nhất có logic thật, và nó đáng có vì compose không trả lời
được câu hỏi này. Phải phân biệt "container đang chạy" với "scrape thật sự
thành công":

```
ASR      ready   http://127.0.0.1:4000/v2/health/ready
Grafana  up      http://127.0.0.1:4003/api/health
Prom     target  asr:8002  UP
```

Dòng cuối hỏi `/api/v1/targets` của Prometheus, không chỉ kiểm container sống.

Trước khi `up`, `serving.sh` kiểm bốn file weights tồn tại trong
`model_repository/asr_streaming/1/`. Thông điệp lỗi phải chỉ đúng cách khắc
phục cho máy không có internet:

```
dừng: thiếu model_repository/asr_streaming/1/encoder.onnx
      Thor không ra được internet. Tải ở máy dev rồi:
      rsync -av model_repository/ thor:~/thor-voice-serving/model_repository/
```

Những gì **không** làm nữa vì compose đã lo: dọn container cũ, port guard bằng
`ss`, restart policy, mapping, vòng chờ health. Ước tính 113 dòng bash → khoảng
60 dòng, và phần bị xoá đúng là phần đã hai lần sinh bug.

## 7. Test

**`tests/test_parity.sh`** — chuyển từ `scripts/thor/parity_check.sh`, giữ
nguyên logic (kể cả đoạn strip ANSI đã sửa bug), đổi nguồn cổng sang `.env`.
Tách khỏi `serving.sh` vì nó cần `tritonclient[grpc]`, `numpy`, `soundfile` —
script vận hành không nên gánh dependency Python của tầng test.

**`tests/test_config.py`** — mới. `serving/metrics.py` có comment ghi
*"test_serving_metrics.py và test_monitoring_config.py canh cả ba nơi"* nhưng
hai test đó không tồn tại trong repo này (chúng ở repo gốc
`triton-voice-serving`). Bất biến đang không ai canh:

1. `CCU_TTL_S` trong `serving/metrics.py` phải khớp
   `max_sequence_idle_microseconds` trong `model_repository/asr_streaming/config.pbtxt`
2. Con số đó phải khớp giá trị nhúng trong query của dashboard JSON
3. Dashboard JSON committed phải khớp output của `build_dashboard.py`
   (chạy builder, so với file trên đĩa)

Lệch bất kỳ chỗ nào thì CCU sai âm thầm, không báo lỗi gì.

Thêm: test khẳng định `config/prometheus.yml` trỏ `asr:8002` và
`datasource.yml` trỏ `http://prometheus:9090` — chặn đúng lớp lỗi drift đã xảy
ra.

Chạy bằng `pytest tests/` gõ tay, không bọc vào `serving.sh`.

## 8. Mở rộng cho model tương lai

Không thêm abstraction nào. `model_repository/` của Triton **vốn đã là kiến
trúc plugin**: thêm model = thêm một thư mục `<tên>/config.pbtxt` + `1/model.py`,
không sửa dòng nào ở chỗ khác. Cùng một `tritonserver` load cả repository.

Contract đã tồn tại, refactor này chỉ viết nó thành tài liệu + test:

- Model mới import `serving/metrics.py`, phát metric kèm label `model` và
  `model_instance` (không dùng label `instance` — Prometheus chiếm riêng cho địa
  chỉ target).
- Dashboard query theo label `model`, nên model mới **tự hiện lên** không cần
  sửa JSON.
- `serving/metrics.py` đã có sẵn `TTS_RTF_BUCKETS` bên cạnh `ASR_RTF_BUCKETS` —
  hai thang lệch một bậc nên không dùng chung buckets được.
- Chỉ khi model cần image khác (TTS/`piper_phonemize`) mới thêm một service vào
  `compose.yaml`; profiles đã sẵn sàng cho việc đó.

Viết mục này vào README. Thêm khung trừu tượng lúc mới có một model là đoán mò.

## 9. Tinh thần CI/CD

Không có file workflow. Thor bị chặn mạng, không có registry nội bộ, nên CD là:

```bash
git pull
./scripts/serving.sh build
./scripts/serving.sh up
./tests/test_parity.sh
```

Ba thứ làm nó lặp lại được và chuẩn production, đều là core:

1. **Log rotation** (`max-size: 50m`, `max-file: 3`). Container thường trú không
   xoay log là đầy đĩa Thor — mà Thor là máy dùng chung nhiều người.
2. **`IMAGE_TAG` trong `.env`.** Không có registry giữ bản cũ thì tag local là
   đường rollback duy nhất; tag trống thì mỗi lần build đè mất bản trước.
3. **Ghim base image.** `ASR_BASE_IMAGE` trong `.env`, mặc định
   `asr-triton:latest`. README đã tự nhận đây là rủi ro: image do team khác kiểm
   soát, họ rebuild là build của mình đổi âm thầm. Đưa ra biến để ghim được theo
   digest khi cần, không ép ngay vì hiện chỉ có `:latest` trong cache local.

## 10. README viết lại

Đang sai ở bốn chỗ: cổng 9000-9002 (thực tế 1000-1002, sắp thành 4000-4003);
`fetch_models.sh` không còn; `dump_golden_asr.py` bảo xem ở repo khác trong khi
file nằm ngay trong repo; hướng dẫn tải weights trên Thor trong khi Thor không
ra được internet.

Viết lại theo bố cục: yêu cầu → chuyển weights sang Thor → `serving.sh` →
tunnel để xem dashboard → parity → thêm model mới → phần còn lại (giữ nguyên
mục "Việc còn lại" hiện có, nó vẫn đúng).

## 11. Rủi ro

- **Đổi cổng 1000/1001 → 4000/4001** phá mọi tunnel/script đang dùng số cũ.
- **Cần Docker Compose v2** cho profiles và `up --wait`. Kiểm trên Thor bằng
  `docker compose version` trước khi bắt đầu. Nếu Thor chỉ có v1 thì thiết kế
  này không chạy và phải quay lại phương án bash.
- **Không test được ở máy dev**: build cần arm64 + base image chỉ có trên Thor;
  parity cần GPU + weights. Việc xác minh phải làm trên Thor.
- **Không đụng gì tới `model_repository/asr_streaming/`** (`model.py`,
  `streaming_search.py`, `config.pbtxt`) — chúng copy nguyên từ
  `triton-voice-serving` và đang chạy đúng. Refactor này chỉ chạm hạ tầng.
