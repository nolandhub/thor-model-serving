# thor-voice-serving

ASR streaming (Zipformer RNN-T) trên Jetson AGX Thor, dùng Triton Inference
Server. `docker/Dockerfile.thor` build thẳng trên `asr-triton:latest` —
image của đồng nghiệp (`voice-agent-deployment/asr-triton`), đã có sẵn
torch/torchaudio/onnxruntime-gpu/libsndfile1 và fix `LD_LIBRARY_PATH`, chạy
sản xuất 12+ ngày trên chính con Thor này. Không tự apt-get/pip gì thêm —
mọi lần thử tự cài đều bị proxy nội bộ công ty chặn ở `ports.ubuntu.com` và
`developer.download.nvidia.com`; đi qua image đã build sẵn là né hẳn.

Đây là v1: **chỉ ASR + Prometheus + Grafana**. TTS chưa nằm trong repo này.

## Trước khi chạy

Thor này đang chạy pipeline sản xuất của người khác
(`voice-agent-asr-triton-1` giữ cổng 8000/8001, `voice-agent-tts-triton-1`
giữ 8010/8011). Repo này **không đụng tới các container đó** — dùng dải cổng
riêng 9000-9002 cho ASR, và 3000/9090 cho Grafana/Prometheus (đang trống lúc
kiểm). Script tự chặn nếu cổng bị chiếm, không tự ý dừng gì của người khác.

## Chạy trên Thor

```bash
git clone <repo-này> && cd thor-voice-serving
./scripts/fetch_models.sh          # tải encoder/decoder/joiner/bpe.model từ HuggingFace
./scripts/thor/deploy_asr.sh       # build Dockerfile.thor, chạy tritonserver ở :9000-9002
./scripts/thor/deploy_monitoring.sh
```

## Kiểm "chạy đúng như máy dev"

```bash
pip install --user "tritonclient[grpc]" numpy soundfile scipy   # nếu Thor chưa có
./scripts/thor/parity_check.sh
```

So transcript của `tests/assets/sample_vi.wav` với chuẩn đã dump từ ONNX
Runtime trên máy dev (`tests/assets/golden_asr_sample_vi.npz` — xem
`scripts/dump_golden_asr.py` ở repo gốc `triton-voice-serving` để biết cách
dump lại nếu model đổi). Khớp nghĩa là port đúng, không phải "server không
crash" — hai chuyện khác nhau.

## Xem dashboard

Prometheus/Grafana bind loopback — Thor dùng chung nhiều người, không phơi
metrics/admin ra LAN. Từ máy dev:

```bash
ssh -L 3300:localhost:3000 -L 9900:localhost:9090 <user>@thor
```

Rồi mở `http://localhost:3300` (Grafana) và `http://localhost:9900/targets`
(Prometheus, để debug target UP/DOWN).

## Cấu trúc

| | |
|---|---|
| `docker/Dockerfile.thor` | base `asr-triton:latest` (image của đồng nghiệp, đã có sẵn trên Thor) - không cài thêm gì, chỉ kiểm CUDAExecutionProvider có mặt |
| `model_repository/asr_streaming/` | copy nguyên từ `triton-voice-serving` — `model.py`, `streaming_search.py`, `config.pbtxt` không đổi dòng nào |
| `scripts/thor/deploy_asr.sh` | build + run, cổng 9000 (http) / 9001 (gRPC) / 9002 (metrics), có port-guard |
| `scripts/thor/deploy_monitoring.sh` | Prometheus + Grafana, bind `127.0.0.1` |
| `scripts/thor/parity_check.sh` | đối chiếu transcript với golden fixture |

## Việc còn lại (chưa làm ở v1 này)

- **Rủi ro coupling với `asr-triton:latest`** — image base do team khác kiểm
  soát, không phải của repo này. Họ rebuild/xoá thì lần build sau của mình ra
  khác đi mà không có cảnh báo gì. Khi có thời gian: tự dựng một base image
  riêng (từ `tritonserver:r38.4.arm64-sbsa-cu130-24.04` + torch/onnxruntime-gpu
  qua `pypi.jetson-ai-lab.io/sbsa/cu130`) và xin whitelist proxy cho
  `ports.ubuntu.com`/`developer.download.nvidia.com` để không phụ thuộc image
  của người khác nữa.
- TTS (ZipVoice) — chưa kiểm `piper_phonemize` có wheel aarch64 hay không
- Chuyển ASR sang TensorRT — không bắt buộc, ONNX Runtime CUDA EP đã chạy được;
  đáng làm sau khi có số RTF/latency thật để so sánh
- `jetson_exporter.py` cho EMC/power/thermal — dashboard hiện chỉ có panel
  tầng ứng dụng (RPS/latency/CCU/RTF/error rate) và tầng Triton nội tại
- systemd/restart policy cho thiết bị thường trú — container hiện dùng
  `--restart unless-stopped`, chưa có health-triggered restart hay log rotation
