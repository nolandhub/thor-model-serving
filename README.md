# thor-voice-serving

ASR streaming (Zipformer RNN-T) trên Jetson AGX Thor, dùng Triton Inference
Server. `docker/Dockerfile.thor` build thẳng trên `asr-triton:latest` — image
của đồng nghiệp (`voice-agent-deployment/asr-triton`), đã có sẵn
torch/torchaudio/onnxruntime-gpu/libsndfile1 và fix `LD_LIBRARY_PATH`, chạy sản
xuất 12+ ngày trên chính con Thor này. Không tự apt-get/pip gì thêm — mọi lần
thử tự cài đều bị proxy nội bộ công ty chặn ở `ports.ubuntu.com` và
`developer.download.nvidia.com`; đi qua image đã build sẵn là né hẳn.

Đây là v1: **chỉ ASR + Prometheus + Grafana**. TTS chưa nằm trong repo này.

## Kiến trúc

Toàn bộ stack nằm trong một bridge network riêng do compose tạo
(`thor-voice_default`). Service gọi nhau bằng **tên service**, không qua cổng
host — nên đổi cổng host không ảnh hưởng gì tới cấu hình bên trong.

| Cổng host (`${BIND_ADDR}`) | → Container | Tầng |
|---|---|---|
| `4000` | `asr:8000` | data plane — Triton HTTP |
| `4001` | `asr:8001` | data plane — Triton gRPC (đường streaming thật) |
| `4002` | `prometheus:9090` | control plane — debug `/targets` |
| `4003` | `grafana:3000` | control plane — dashboard |
| *(không publish)* | `node-exporter:9100` | metric host: đĩa, RAM, CPU, nhiệt độ Tegra |
| *(không publish)* | `asr:8002` | metrics — chỉ Prometheus trong network gọi |

Thor là máy dùng chung: `voice-agent-asr-triton-1` giữ 8000/8001,
`voice-agent-tts-triton-1` giữ 8010/8011. Repo này không đụng tới chúng.

## Chạy trên Thor

Repo đi bằng git, riêng trọng số thì không: Thor bị chặn ra internet nên
không tải trực tiếp được, mà `*.onnx` cũng không commit. Chỉ ba file này nằm
ngoài git (~49 MB) — `bpe.model` đã có sẵn trong repo.

```bash
# trên Thor
git clone <repo-này> ~/thor-voice-serving

# trên máy dev
rsync -avP --include='*/' --include='*.onnx' --exclude='*' \
  model_repository/ <user>@thor:~/thor-voice-serving/model_repository/
```

Rồi trên Thor:

```bash
cp .env.example .env          # sửa cổng ở đây nếu cần, chỉ một nơi duy nhất
./scripts/serving.sh build
./scripts/serving.sh up       # không tham số = cả ASR lẫn monitoring
./scripts/serving.sh health
```

```
./scripts/serving.sh build | up [asr|monitoring] | down [-v] | logs [service] | health | help
```

`health` phân biệt "container đang chạy" với "scrape thật sự thành công" —
`docker compose ps` chỉ trả lời được vế đầu.

## Kiểm "chạy đúng như máy dev"

```bash
pip install --user "tritonclient[grpc]" numpy soundfile scipy   # nếu Thor chưa có
./tests/test_parity.sh
```

So transcript của `tests/assets/sample_vi.wav` với chuẩn đã dump từ ONNX Runtime
trên máy dev (`tests/assets/golden_asr_sample_vi.npz`, sinh lại bằng
`tests/dump_golden_asr.py`). Khớp nghĩa là port đúng — khác hẳn với "server
không crash".

Tách khỏi `serving.sh` vì nó cần `tritonclient`/`numpy`/`soundfile`; script vận
hành không nên gánh dependency Python của tầng test.

Ngoài ra `pytest tests/` chốt các bất biến trải trên nhiều file (không cần GPU,
không cần Thor): `CCU_TTL_S` khớp giữa `serving/metrics.py`, `config.pbtxt` và
query dashboard; dashboard JSON khớp `build_dashboard.py`; giao tiếp nội bộ đi
bằng tên service chứ không hard-code cổng.

## Đưa image mới vào Thor

Thor không pull được từ Docker Hub. Đường chuẩn là đóng gói ở máy dev rồi nạp
vào — ví dụ với `node-exporter`:

```bash
# máy dev (nhớ --platform, Thor là arm64)
docker pull --platform linux/arm64 prom/node-exporter:latest
# --platform ở CẢ save nữa: mặc định save tìm manifest khớp host (amd64) và
# báo "content digest ... not found" vì chỉ có lớp arm64 được kéo về.
docker save --platform linux/arm64 prom/node-exporter:latest -o /tmp/node-exporter-arm64.tar
rsync -avP /tmp/node-exporter-arm64.tar <user>@thor:/tmp/

# Thor
docker load -i /tmp/node-exporter-arm64.tar
```

## Xem dashboard

Prometheus/Grafana bind theo `${BIND_ADDR}` (mặc định `127.0.0.1`) — Thor dùng
chung nhiều người, không phơi metrics/admin ra LAN. Từ máy dev:

```bash
ssh -L 4002:localhost:4002 -L 4003:localhost:4003 <user>@thor
```

Dùng `127.0.0.1` chứ đừng gõ `localhost`: SSH thường không bind được IPv6
loopback (`bind [::1]: Cannot assign requested address`), mà `localhost` lại
hay phân giải sang `::1` trước.

Ba dashboard được provision sẵn: `Voice Serving` (metric ứng dụng + host),
`Triton` (nội tại server), và `Node Exporter Full` (dashboard cộng đồng ID 1860,
~200 panel, dùng khi cần đào sâu tầng host).

Rồi mở `http://127.0.0.1:4003` (Grafana) và `http://localhost:4002/targets`
(Prometheus, để debug target UP/DOWN).

## Gọi từ máy khác

Mặc định `BIND_ADDR=127.0.0.1`, tức chỉ gọi được từ chính Thor hoặc qua tunnel.
Đổi thành IP LAN của Thor thì máy khác gọi vào inference được — **trước khi
đổi, đọc hết mục này**:

- **Triton không có authentication.** Ai trong LAN cũng gọi được. Repo giữ
  `--model-control-mode=none` tường minh để API load/unload model bị vô hiệu.
- **Cổng 8002 vẫn không publish** — metrics lộ ra LAN là lộ throughput và
  pattern tải.
- **Grafana đang bật anonymous admin.** Nếu đổi `BIND_ADDR` chung cho cả stack
  thì phải tắt `GF_AUTH_ANONYMOUS_*` trước.
- **`docker -p` đi vòng qua `firewalld`.** Rule nằm ở chain `DOCKER`/`DOCKER-USER`,
  không phải `INPUT` — chặn bằng firewalld sẽ không có tác dụng như mong đợi.

Contract cho client (model `asr_streaming` dùng `sequence_batching`, không phải
request rời):

- gRPC `thor:4001`, model `asr_streaming`, version `1`
- **bắt buộc `stream_infer` kèm `sequence_id` + `sequence_start`/`sequence_end`**
- input `AUDIO_CHUNK` FP32 `[-1]`, PCM mono **16 kHz**; output `TRANSCRIPT` BYTES `[1]`
- `max_batch_size: 8`, `instance_group: 2 GPU` → trần khoảng 8 phiên đồng thời
- phiên im lặng quá **60 s** bị dọn (`max_sequence_idle_microseconds`)
- HTTP `4000` chỉ để health/metadata; streaming phải đi gRPC
- reference implementation: `client/asr_streaming_client.py`

## Thêm model mới

`model_repository/` của Triton vốn đã là kiến trúc plugin — thêm model là thêm
một thư mục, không sửa dòng nào ở chỗ khác. Cùng một `tritonserver` load cả
repository.

Contract cần theo:

- import `serving/metrics.py`, phát metric kèm label `model` và `model_instance`
  (**không** dùng label `instance` — Prometheus chiếm riêng cho địa chỉ target)
- dashboard query theo label `model` nên model mới **tự hiện lên**, không sửa JSON
- `serving/metrics.py` đã có sẵn `TTS_RTF_BUCKETS` cạnh `ASR_RTF_BUCKETS` — hai
  thang lệch một bậc nên không dùng chung buckets được
- chỉ khi model cần image khác (TTS/`piper_phonemize`) mới thêm service vào
  `compose.yaml`; profiles đã sẵn cho việc đó

## Deploy lại

Không có registry nội bộ (Thor bị chặn mạng) nên build tại chỗ:

```bash
git pull
IMAGE_TAG=$(git rev-parse --short HEAD) ./scripts/serving.sh build
IMAGE_TAG=$(git rev-parse --short HEAD) ./scripts/serving.sh up
./tests/test_parity.sh
```

Tag theo git sha là đường rollback duy nhất khi không có registry giữ bản cũ.

## Cấu trúc

| | |
|---|---|
| `compose.yaml` | cả 3 service, hai profile (`asr`, `monitoring`) |
| `.env.example` | nguồn sự thật duy nhất cho cổng và tên |
| `scripts/serving.sh` | vòng đời container, chỉ cần bash + docker |
| `docker/Dockerfile.thor` | base `asr-triton:latest`, không cài thêm gì, chỉ kiểm CUDAExecutionProvider |
| `config/prometheus.yml` | scrape `asr:8002` và `node-exporter:9100` qua DNS network |
| `config/grafana/` | provisioning, dashboards, và `build_dashboard.py` sinh ra chúng |
| `serving/metrics.py` | contract metric dùng chung — model.py và build_dashboard.py cùng import |
| `model_repository/asr_streaming/` | copy nguyên từ `triton-voice-serving`, không đổi dòng nào |
| `tests/` | parity, bất biến cấu hình, và `dump_golden_asr.py` sinh fixture |
| `docs/ARCHITECTURE.md` | tổng quan hệ thống và vì sao mọi thứ xếp như vậy |
| `docs/superpowers/specs/` | thiết kế của lần refactor này |

## Việc còn lại

- **Rủi ro coupling với `asr-triton:latest`** — image base do team khác kiểm
  soát. Họ rebuild/xoá thì lần build sau ra khác đi mà không có cảnh báo gì.
  `ASR_BASE_IMAGE` trong `.env` cho phép ghim theo digest; đường dài thì tự dựng
  base image riêng (từ `tritonserver:r38.4.arm64-sbsa-cu130-24.04` +
  torch/onnxruntime-gpu qua `pypi.jetson-ai-lab.io/sbsa/cu130`) và xin whitelist
  proxy cho `ports.ubuntu.com`/`developer.download.nvidia.com`.
- Chốt phương án expose data plane ra LAN cho đồng nghiệp (giới hạn nguồn ở
  `DOCKER-USER`? bind IP LAN cụ thể?) — hiện để mặc định loopback.
- TTS (ZipVoice) — chưa kiểm `piper_phonemize` có wheel aarch64 hay không
- Chuyển ASR sang TensorRT — ONNX Runtime CUDA EP đã chạy được; đáng làm sau khi
  có số RTF/latency thật để so
- Alert: hiện chưa có luật nào. Dashboard là thứ phải mở ra nhìn, production
  cần thứ chủ động báo. Tối thiểu 4 luật: target down, error rate, queue depth,
  đĩa sắp đầy.
- GPU utilization và EMC bandwidth vẫn thiếu — NVML không đọc được trên Tegra
  (log Triton: "Unable to get power limit ... value:0"), phải qua `tegrastats`.
  Nhiệt độ thì node_exporter đã lo qua collector `thermal_zone`.
- Độ trễ tới partial đầu tiên chưa được đo — đây mới là con số người dùng cuối
  cảm nhận ở voice agent, `nv_inference_request_summary_us` chỉ đo mỗi chunk.
