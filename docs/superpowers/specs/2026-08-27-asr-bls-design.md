# Tách encoder thành model riêng có batch (BLS) — thiết kế

Ngày 2026-08-27. Trạng thái: chờ duyệt.

## 1. Vì sao

Bench ngày 2026-08-27 (`bench/README.md`) đo được ba điều:

- **Sức chứa hiện tại 4 stream đồng thời.** Ở CCU 4, p99 0.189s trên chunk 0.200s —
  dư địa 5%. Ở CCU 8, drift +0.16s và dồn dần.
- **Encoder chiếm ~98% thời gian tính toán.** encoder 18.2ms/lần ở batch 1; decoder
  0.17ms; joiner 0.10ms.
- **Batch 8 chỉ tốn thêm 21% thời gian so với batch 1** (19.16 → 20.13ms), tức
  **7.62× mỗi mẫu**. Encoder ở batch 1 bị chặn ở chi phí phóng kernel và băng thông
  đọc trọng số, không phải ở tính toán — khớp với quan sát utilization 79% mà power
  chỉ 8W.

Kiến trúc hiện tại không thể khai thác điều đó: `execute()` xử lý tuần tự

    for request in requests:
        responses.append(self._handle(request))

nên dù Triton có gom được batch (đã thấy `avg_batch` 2.13 ở cấu hình 2 instance),
python vẫn gọi ONNX từng request một. Nâng instance 2→4 dọn sạch hàng đợi ở CCU 4
nhưng **không nới được trần**.

## 2. Mục tiêu

Tăng **CCU bền vững tối đa** với hai ràng buộc cứng:

- p99 độ trễ mỗi chunk **< 200ms** (bằng độ dài chunk)
- **drift không dồn** — giá trị cuối không tăng theo thời gian

Chấp nhận một lượng queue delay có kiểm soát để đổi lấy batch.

Không phải mục tiêu: giảm độ trễ ở tải hiện tại; giảm bộ nhớ.

## 3. Quyết định kiến trúc

**Chỉ bọc encoder.** Decoder + joiner cộng lại là 1.5% một lần chạy encoder. Batch
chúng buộc phải xử lý chuyện mỗi stream phát ra số token khác nhau — phần phức tạp
nhất của batched greedy search — để đổi lấy vài phần trăm mili giây. Không làm.

**State dùng implicit state của Triton**, không truyền tay.

Encoder có **74 tensor state, 1.32 MiB mỗi stream** (đo bằng `bench/encoder_batch.py`).
Truyền tay nghĩa là 74 vào + 74 ra = **148 `pb_utils.Tensor` mỗi chunk mỗi stream**.
Chi phí nằm ở overhead từng object chứ không ở số byte (băng thông chỉ ~105 MiB/s ở
CCU 8 — không đáng kể với LPDDR của Thor). Ước lượng ~3ms/chunk, ăn ~20% phần thắng.

Với `sequence_batching { state [...] }`, Triton giữ cache của từng sequence trong bộ
nhớ của nó. Python model chỉ gửi `x` và nhận `encoder_out` — **2 tensor thay vì 148**.

Phương án đã cân nhắc và loại:

| | Vì sao loại |
|---|---|
| Truyền state tay | 148 tensor/chunk/stream. **Giữ làm đường lùi** nếu implicit state không chạy được |
| Gộp state bằng phẫu thuật đồ thị ONNX | Sửa file đã export, mà pipeline export do nhóm khác giữ. Mỗi lần export lại phải làm lại, rủi ro sai lệch numeric không phát hiện được ngoài parity test đầy đủ. Đổi lấy thứ implicit state đã cho miễn phí |

## 4. Spike 0 — cổng chặn trước mọi thứ khác

Implicit state với ONNX Runtime backend **chưa từng chạy trên Thor**. Trước khi viết
bất kỳ dòng nào của model mới:

1. Dựng `encoder` model riêng với `sequence_batching` + 74 khối `state`, config sinh
   bằng script từ chính `encoder.onnx`.
2. Gọi một stream đơn từ một script throwaway: gửi vài chunk `x` liên tiếp cùng một
   `correlation_id`, so `encoder_out` với kết quả chạy trực tiếp bằng onnxruntime.
3. Đọc `decode_chunk_len` từ metadata của `encoder.onnx` và ghi lại — nó quyết định
   **tần suất gọi encoder**, tức là cơ hội gom batch (xem §6).

Kết quả spike là một câu trả lời, không phải code giữ lại.

- **Chạy được và khớp** → đi tiếp mục 5.
- **Backend không chịu 74 cặp state** → lùi về truyền tay, và bàn lại vì `bls_tax`
  lúc đó không còn gần 0.

## 5. Thiết kế

### 5.1 Model repository

    model_repository/
      asr_streaming/          giữ NGUYÊN, không sửa một dòng nào
      asr_streaming_bls/      python: fbank + greedy, gọi encoder qua BLS
      encoder/                ONNX encoder, sequence_batching + implicit state

Đường cũ giữ nguyên tên nên client, `tests/test_parity.sh` và bench hiện tại không
phải sửa gì. Hai đường **nạp cùng lúc** (~2.1 GiB mỗi bộ, Thor còn ~31 GB trống) để
so hai bảng trên cùng một lần khởi động — cùng nhiệt độ, cùng clock. Buổi đo
2026-08-27 đã cho thấy so hai bảng cách nhau một lần restart dễ nhầm tới mức nào.

### 5.2 Model `encoder`

- `platform: onnxruntime_onnx`, `max_batch_size: 8`
- `sequence_batching { oldest { max_candidate_sequences: 8 } state [ ...74... ] }`
- `state` sinh bằng script từ `encoder.onnx`: input thứ i (bỏ `x`) ghép với output thứ
  i (bỏ `encoder_out`) theo **vị trí**, đúng bất biến mà `model.py` hiện tại đang dựa
  vào và đã khẳng định bằng `assert`
- `initial_state`: zeros, đúng như `_Stream.__init__` đang làm

### 5.3 Model `asr_streaming_bls`

Sao từ `model.py` hiện tại, thay đúng một chỗ: `_encoder_step()` không gọi
`self.encoder.run()` nữa mà gửi `pb_utils.InferenceRequest` tới model `encoder` với

- `correlation_id` = CORRID nó nhận được từ client
- cờ `SEQUENCE_START` ở chunk đầu, `SEQUENCE_END` ở chunk cuối

`_Stream` không giữ `enc_states` nữa. **Cờ END là bắt buộc**: quên gửi thì slot state
trong Triton rò rỉ. Đường cũ đã có `_sweep()` phòng cho stream chết không gửi END;
đường mới phải phòng tương đương — gửi END khi `_sweep()` dọn một stream mồ côi.

`fbank` và vòng greedy (decoder/joiner) ở nguyên trong python, không đổi.

## 6. `max_queue_delay` — tham số phải quét, không phải hằng số đoán được

Batch chỉ gom được khi nhiều request tới trong cùng một cửa sổ. Tần suất gọi encoder
phụ thuộc `decode_chunk_len` (spike đọc ra): mỗi bước encoder tiêu thụ ngần ấy khung,
trong khi mỗi chunk 200ms chỉ nạp 20 khung. Nếu `decode_chunk_len` là 32 thì encoder
chỉ chạy khoảng **mỗi 320ms một lần cho mỗi stream** — 8 stream không đồng bộ thì xác
suất hai request rơi vào cùng một cửa sổ 10ms là thấp.

Nói cách khác: **không chờ thì không có batch.** Nhưng chờ ăn vào chính ngân sách p99.

Nên không chốt một con số trong spec này. Bench đã có sẵn để quét:

    max_queue_delay_microseconds ∈ {0, 5000, 10000, 20000, 40000}

Giá trị khởi điểm 10ms. Chọn giá trị cho **CCU bền vững cao nhất** thoả cả hai ràng
buộc ở §2. Nếu mọi giá trị đều không cải thiện được CCU thì kết luận là BLS không
mua được gì ở quy mô này, và đó cũng là một kết quả hợp lệ — ghi vào `bench/README.md`
rồi dừng.

## 7. Đo và tiêu chí nghiệm thu

`bench/run_asr.py` thêm `--model` (hiện đang cứng `asr_streaming`) để trỏ sang
`asr_streaming_bls`. Không đổi gì khác trong driver — cùng driver, cùng audio, cùng
mức CCU thì hai bảng mới so được.

Nghiệm thu, theo thứ tự bắt buộc:

1. **Parity**: `tests/test_parity.sh` chạy được với đường BLS và ra đúng transcript
   golden. Nhanh hơn mà sai chữ thì không phải cải tiến.
2. **CCU bền vững cao hơn 4** với p99 < 200ms và drift không dồn.
3. **`bls_tax` < 0.3.** Vượt ngưỡng này nghĩa là thuế đang ăn phần lớn phần thắng —
   dừng lại xem xét, đừng nhận.

## 8. Rủi ro

| Rủi ro | Xử lý |
|---|---|
| ORT backend không chịu 74 cặp implicit state | Spike 0 chặn trước. Lùi về truyền tay |
| Rò rỉ slot state khi stream chết không gửi END | `_sweep()` gửi END khi dọn stream mồ côi |
| Thêm một nhóm tiến trình = thêm CUDA context | Đã biết: ~529 MiB mỗi instance, phần lớn là context chứ không phải 51 MB trọng số. Tách encoder **không** tiết kiệm bộ nhớ, nhiều khả năng tốn thêm |
| Chờ để gom batch làm hỏng drift | §6 quét tham số; ràng buộc drift ở §2 là cứng |
| Thor là máy dùng chung, còn ~31 GB | Hai đường nạp cùng lúc tốn ~2.1 GiB, 7% chỗ trống. Chấp nhận |

## 9. Không làm

- Batch decoder/joiner (1.5% thời gian, đổi lấy phần phức tạp nhất)
- Phẫu thuật đồ thị ONNX
- Đổi pipeline export
- Sửa `asr_streaming` cũ — nó là đường đối chứng, phải giữ nguyên
