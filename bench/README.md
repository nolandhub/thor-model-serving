# Bench — đánh giá kiến trúc ASR trước khi chuyển sang BLS

Câu hỏi bài bench sinh ra để trả lời: **kiến trúc hiện tại (onnxruntime chạy thẳng
trong Python backend) đang nghẽn ở đâu, và tách encoder ra thành model riêng để
Triton gom batch (BLS) có đáng không.**

## Chạy

    ./scripts/serving.sh bench                    # quét CCU 1,2,4,8 - khoảng 15 phút
    ./scripts/serving.sh bench --ccu 8 --runs 1   # một mức, một run - khoảng 35 giây

    # microbench từng tầng ONNX, không qua Triton
    docker compose --profile asr --profile bench run --rm --entrypoint python3 bench \
      bench/encoder_batch.py --encoder model_repository/asr_streaming/1/encoder.onnx

Bench chạy **trong network của compose** để tới được `asr:8002` — cổng metrics cố ý
không publish ra host vì data plane có thể mở ra LAN mà Triton không có auth.

## Bố cục

    bench/run_asr.py        driver: quét CCU, gửi chunk đúng nhịp realtime, chụp counter
    bench/report.py         các metric quyết định - hàm thuần, test được khi Thor tắt
    bench/schedule.py       lịch gửi theo mốc tuyệt đối
    bench/triton_metrics.py đọc /metrics của Triton (docker exec hoặc HTTP trong network)
    bench/encoder_batch.py  microbench một file ONNX ở batch 1..8, độc lập với Triton
    bench/results/          báo cáo sinh ra (gitignore - số phụ thuộc máy đo)

## Đọc số nào

| Cột | Nói gì |
|---|---|
| `p99` | độ trễ một chunk. Vượt độ dài chunk (200ms) là không theo kịp audio |
| `drift cuối` | **quan trọng hơn p99.** Đứng yên = bám realtime; TĂNG DẦN = hàng đợi dồn vô hạn, tức là sập, dù p99 vẫn có thể đẹp |
| `avg_batch` | request/lần chạy model. ≈1.0 = dynamic batcher không gom được gì |
| `bls_tax` | (compute_input + compute_output) / compute_infer — chi phí copy tensor qua ranh giới model |
| `queue` | thời gian nằm chờ instance rảnh |
| `GPC` | trung vị clock GPU đo trong lúc chạy. **Không phải** chỉ số hãm — chỉ để đối chiếu giữa hai lần đo |

## Kết quả 2026-08-27 — mốc chuẩn

Thor AGX. `asr_streaming | instance 4 x KIND_GPU | max_batch_size 8 | max_candidate_sequences 8`.
18.4s audio/stream, chunk 200ms, median của 3 run, có warmup.

| CCU | p99 (s) | drift cuối (s) | avg_batch | bls_tax | queue (µs/req) | GPC (MHz) |
|---|---|---|---|---|---|---|
| 1 | 0.080 | -0.12 | 1.00 | 0.01 | 127 | 778 |
| 2 | 0.151 | -0.05 | 1.00 | 0.01 | 113 | 914 |
| 4 | 0.189 | -0.01 | 1.00 | 0.00 | 141 | 1332 |
| 8 | 0.205 | +0.16 | 1.04 | 0.00 | 28177 | 1575 |

**Sức chứa: 4 stream đồng thời.** Ở CCU 4 dư địa chỉ còn 5% (p99 0.189s / chunk 0.200s).

### Nâng instance 2 → 4 chữa được gì

| | 2 instance | 4 instance |
|---|---|---|
| queue @CCU4 | 17.804µs | **141µs** |
| drift @CCU4 | +0.01 | **-0.01** |
| queue @CCU8 | 23.668µs | 28.177µs |
| drift @CCU8 | +0.19 | +0.16 |

Dọn sạch hàng đợi ở mức 4, **không nới được trần**. Đáng chú ý: 2 instance gom được
`avg_batch` 2.13 ở CCU 8 mà throughput không hơn cấu hình 4 instance (`avg_batch` 1.04).
Gom được batch cũng không nhanh hơn — vì `execute()` xử lý tuần tự:

    for request in requests:
        responses.append(self._handle(request))

Batch gom xong rồi python chạy từng request một. **Kiến trúc hiện tại không biến được
batch thành throughput.**

### GPU đang bận hay đang rảnh

Đo trong lúc chạy CCU 8 (cửa sổ 2 phút, phần có tải ~50s):

    utilization  max 79%    avg 32.6%   → trong lúc chạy ≈ 65-79%
    power        max 8.7W   avg 5.5W    → trong lúc chạy ≈ 8W

79% utilization mà chỉ 8W: NVML tính utilization là *có kernel nào đang chạy không*,
không phải *GPU đầy bao nhiêu phần*. GPU hiếm khi rảnh nhưng làm rất ít việc — dấu vân
tay của chuỗi kernel nhỏ chạy nối đuôi.

### Microbench từng tầng ONNX

| tầng | batch 1 | batch 8 (mỗi mẫu) | tỷ trọng |
|---|---|---|---|
| **encoder** | **18.234ms** | 2.770ms (6.58×) | **~98%** |
| decoder | 0.169ms | 0.023ms (7.44×) | ~1% |
| joiner | 0.096ms | 0.012ms (7.71×) | ~0.5% |

Batch 8 chỉ tốn thêm 21% thời gian so với batch 1 (18.23 → 22.16ms): bảy chunk kia gần
như đi nhờ miễn phí. Encoder ở batch 1 bị chặn ở chi phí phóng kernel và băng thông đọc
trọng số, không phải ở tính toán — khớp đúng với utilization cao mà power thấp.

## Kết luận

**BLS đáng làm, và chỉ cần bọc encoder.** Decoder + joiner cộng lại là 1.5% một lần chạy
encoder; batch chúng không đáng đổi lấy phần phức tạp nhất (mỗi stream phát ra số token
khác nhau).

- trần lý thuyết: **6.58×** ở batch 8
- `bls_tax` nền hiện tại: **0.00–0.01**. Bản BLS đẩy số này quá ~0.3 thì thuế ăn hết phần thắng
- fbank và vòng greedy ở nguyên trong Python backend

## Năm cái bẫy đã trả giá

1. **Không warmup thì mức CCU đo đầu tiên là rác.** Đã thấy p99 0.434s ở CCU 1 trong khi
   CCU 2 chỉ 0.146s — chi phí nạp ONNX và dựng CUDA context của cả 4 instance rơi hết vào
   đó. Tệ hơn: `max_ccu_within_budget` quét từ dưới lên nên nó tin số rác và kết luận sức
   chứa bằng **0**.

2. **Sửa `config.pbtxt` mà không restart thì Triton vẫn chạy cấu hình cũ.** Stack chạy
   `--model-control-mode=none`, config đọc đúng một lần lúc khởi động; `up` thấy container
   Healthy thì để nguyên. Mất hai lần đo (30 phút) vì bảng mới trùng khít bảng cũ. Nay báo
   cáo tự in cấu hình **hỏi từ server**, không phải đọc từ file.

3. **Docker tiêm proxy công ty vào MỌI container lúc tạo** (mục `proxies` trong
   `~/.docker/config.json`). urllib và grpc đều đọc `http_proxy`, nên bench gửi request tới
   gateway thay vì tới `asr` và ăn `connection refused`. Cùng gốc rễ đó giết luôn Grafana
   (không query nổi Prometheus). Vá bằng anchor `x-no-proxy` trong compose, gắn cho mọi service.

4. **umask 007 trên Thor.** Mọi file `git` ghi ra là `0660`, thư mục `0770`. Prometheus chạy
   uid 65534, Grafana uid 472 — không đọc được `config/`. Prometheus chết trong vòng lặp
   restart, Grafana lên xanh nhưng **không nạp được dashboard nào và không báo lỗi gì**.
   Sửa: `chmod -R o+rX config/`. Sẽ quay lại sau mỗi `git pull` ghi đè file trong đó.

5. **`nvidia-smi` trên Tegra trả `[N/A]` cho clock và throttle reason**, `/sys/class/thermal`
   rỗng — không có tín hiệu "đang bị hãm" nào để đọc. Cột `GPC` lấy mẫu clock trong lúc chạy
   là thứ thay thế, nhưng nó **không** phải chốt chặn, chỉ là chứng cứ đối chiếu giữa hai
   lần đo. Tương tự, `nv_gpu_memory_*` không tồn tại vì Thor dùng LPDDR chung CPU/GPU.
