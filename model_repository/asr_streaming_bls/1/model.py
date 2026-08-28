# ABOUTME: Triton Python backend - như asr_streaming nhưng gọi encoder qua BLS để Triton gom batch
# ABOUTME: Cache encoder do Triton giữ theo correlation_id, không bao giờ đi qua python

import math
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import triton_python_backend_utils as pb_utils

# Triton không tự thêm thư mục model vào sys.path nên phải tự làm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streaming_search import (  # noqa: E402
    NUM_MEL_BINS,
    StreamingFbank,
    emitted_tokens,
    greedy_search_step,
    init_search_state,
)

sys.path.insert(0, "/opt/serving")
from serving.metrics import ASR_RTF_BUCKETS, ModelMetrics  # noqa: E402

BLANK_ID = 0
CONTEXT_SIZE = 2
LOG_EPS = math.log(1e-10)   # giá trị đệm khung cuối, theo quy ước icefall/sherpa
STATE_TTL_S = 60.0          # soi gương max_sequence_idle_microseconds trong config.pbtxt
SAMPLE_RATE = 16000          # mẫu số của RTF; khớp fbank và client
MODEL_NAME = "asr_streaming_bls"
ENCODER_MODEL = "encoder"    # model ONNX riêng, xem model_repository/encoder/config.pbtxt

_ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


class _Stream:
    """State của một stream đang sống.

    Cache encoder KHÔNG nằm ở đây nữa - Triton giữ nó theo correlation_id
    (xem khối `state` trong model_repository/encoder/config.pbtxt). Python chỉ
    còn gửi x và nhận encoder_out: 2 tensor mỗi bước thay vì 148.
    """

    def __init__(self, model):
        self.fbank = StreamingFbank()
        self.feat = np.zeros((0, NUM_MEL_BINS), dtype=np.float32)
        self.search = init_search_state(model.run_decoder, BLANK_ID, CONTEXT_SIZE)
        self.last_seen = time.monotonic()
        self.started = False   # đã gửi SEQUENCE_START xuống encoder chưa


class TritonPythonModel:
    def initialize(self, args):
        # Trọng số dùng chung với asr_streaming, không sao chép sang đây: khỏi
        # nhân đôi 51MB trong git, và hai đường luôn chạy đúng một bộ số thì
        # bảng so mới nói được điều gì.
        d = os.path.join(args["model_repository"], "..", "asr_streaming", "1")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.decoder = ort.InferenceSession(os.path.join(d, "decoder.onnx"), providers=providers)
        self.joiner = ort.InferenceSession(os.path.join(d, "joiner.onnx"), providers=providers)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(d, "bpe.model"))

        # Chỉ cần metadata và shape của encoder, KHÔNG chạy nó ở đây nữa -> mở
        # bằng CPU provider rồi bỏ session ngay. Giữ session sống là tốn thêm
        # một CUDA context cho mỗi instance (bench đo được context chiếm phần
        # lớn 529 MiB mỗi instance, chứ không phải 51MB trọng số).
        probe = ort.InferenceSession(
            os.path.join(d, "encoder.onnx"), providers=["CPUExecutionProvider"]
        )
        meta = probe.get_modelmeta().custom_metadata_map
        self.decode_chunk_len = int(meta["decode_chunk_len"])
        enc_in = probe.get_inputs()[0]
        self.x_name = enc_in.name
        self.x_dtype = _ORT_TO_NP[enc_in.type]
        self.T = int(enc_in.shape[1])   # khung mỗi bước; T - decode_chunk_len là lookahead
        del probe

        self.decoder_in = self.decoder.get_inputs()[0].name
        self.joiner_in = [i.name for i in self.joiner.get_inputs()]
        self.joiner_dtype = _ORT_TO_NP[self.joiner.get_inputs()[0].type]

        self.streams = {}   # corrid -> _Stream
        self.metrics = ModelMetrics(
            pb_utils, MODEL_NAME, args["model_instance_name"], ASR_RTF_BUCKETS
        )

    def run_decoder(self, context):
        y = np.array([context], dtype=np.int64)
        out = self.decoder.run(None, {self.decoder_in: y})[0]
        # Một số bản export trả (N, 1, C), bỏ chiều giữa cho khớp joiner
        out = out[:, 0, :] if out.ndim == 3 else out
        return out.astype(np.float32)

    def run_joiner(self, enc_frame, dec_out):
        feeds = {
            self.joiner_in[0]: enc_frame.astype(self.joiner_dtype),
            self.joiner_in[1]: dec_out.astype(self.joiner_dtype),
        }
        return self.joiner.run(None, feeds)[0].astype(np.float32)

    def _encoder_step(self, stream, corrid, feat_chunk, end=False):
        """Một bước encoder qua BLS. State do Triton giữ theo corrid.

        Cờ START phải đi cùng lần gọi ĐẦU TIÊN của stream và END cùng lần
        cuối. Quên END thì slot state trong Triton không bao giờ được giải
        phóng: số sequence đồng thời bò dần tới max_candidate_sequences rồi
        stream mới treo, và không có gì trong log tố cáo nguyên nhân.
        """
        flags = 0
        if not stream.started:
            flags |= pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_START
            stream.started = True
        if end:
            flags |= pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_END
        resp = pb_utils.InferenceRequest(
            model_name=ENCODER_MODEL,
            requested_output_names=["encoder_out"],
            inputs=[pb_utils.Tensor(self.x_name, feat_chunk[None].astype(self.x_dtype))],
            correlation_id=corrid,
            flags=flags,
        ).exec()
        if resp.has_error():
            raise RuntimeError(f"encoder: {resp.error().message()}")
        out = pb_utils.get_output_tensor_by_name(resp, "encoder_out").as_numpy()
        return out[0].astype(np.float32)

    def _advance(self, stream, corrid, new_feat, flush):
        """Nạp khung mới, chạy encoder đủ số bước, đi tiếp greedy search."""
        if len(new_feat):
            stream.feat = np.concatenate([stream.feat, new_feat])
        while stream.feat.shape[0] >= self.T:
            enc_out = self._encoder_step(stream, corrid, stream.feat[: self.T])
            stream.feat = stream.feat[self.decode_chunk_len :]
            greedy_search_step(
                enc_out, stream.search, self.run_decoder, self.run_joiner, BLANK_ID, CONTEXT_SIZE
            )
        if flush and stream.feat.shape[0] > 0:
            # khung cuối không đủ T - đệm LOG_EPS cho đủ một bước encoder chót
            pad = np.full(
                (self.T - stream.feat.shape[0], NUM_MEL_BINS), LOG_EPS, dtype=np.float32
            )
            enc_out = self._encoder_step(
                stream, corrid, np.concatenate([stream.feat, pad]), end=True
            )
            stream.feat = stream.feat[:0]
            greedy_search_step(
                enc_out, stream.search, self.run_decoder, self.run_joiner, BLANK_ID, CONTEXT_SIZE
            )

    def _sweep(self):
        """Xoá state của stream chết không gửi END.

        Khác đường cũ ở chỗ: ngoài dict trong python còn phải BÁO TRITON giải
        phóng slot state của encoder. Bỏ bước đó thì python sạch mà Triton giữ
        cache vĩnh viễn - rò rỉ chỉ lộ ra khi chạm max_candidate_sequences,
        tức là ở đúng mức tải mà ta đang cố nới lên.
        """
        now = time.monotonic()
        for k in [k for k, s in self.streams.items() if now - s.last_seen > STATE_TTL_S]:
            pb_utils.Logger.log_warn(f"{MODEL_NAME}: xoá state mồ côi corrid={k}")
            stream = self.streams[k]
            if stream.started:
                try:
                    self._encoder_step(
                        stream, k,
                        np.full((self.T, NUM_MEL_BINS), LOG_EPS, dtype=np.float32),
                        end=True,
                    )
                except Exception as e:   # noqa: BLE001
                    pb_utils.Logger.log_warn(f"{MODEL_NAME}: không đóng được corrid={k}: {e}")
            del self.streams[k]

    @staticmethod
    def _flag(request, name):
        t = pb_utils.get_input_tensor_by_name(request, name)
        return t is not None and bool(t.as_numpy().reshape(-1)[0])

    def _handle(self, request):
        """Xử lý trọn một chunk của một stream, trả InferenceResponse."""
        corrid = int(
            pb_utils.get_input_tensor_by_name(request, "CORRID").as_numpy().reshape(-1)[0]
        )
        start = self._flag(request, "START")
        end = self._flag(request, "END")

        if start or corrid not in self.streams:
            if not start:
                # server restart giữa stream chẳng hạn - khởi tạo lại thay vì crash
                pb_utils.Logger.log_warn(
                    f"{MODEL_NAME}: chunk không có state (corrid={corrid}), khởi tạo lại"
                )
            self.streams[corrid] = _Stream(self)
        stream = self.streams[corrid]
        stream.last_seen = time.monotonic()

        audio = (
            pb_utils.get_input_tensor_by_name(request, "AUDIO_CHUNK")
            .as_numpy()
            .reshape(-1)
            .astype(np.float32)
        )
        t0 = time.perf_counter()
        new_feat = stream.fbank.accept_waveform(audio)
        if end:
            tail = stream.fbank.flush()
            if len(tail):
                new_feat = np.concatenate([new_feat, tail]) if len(new_feat) else tail
        self._advance(stream, corrid, new_feat, flush=end)
        # Chunk rỗng (END không kèm audio) là hợp lệ - bỏ qua thay vì chia cho 0.
        if len(audio):
            self.metrics.observe_rtf(time.perf_counter() - t0, len(audio) / SAMPLE_RATE)

        text = self.sp.decode(emitted_tokens(stream.search, CONTEXT_SIZE))
        if end:
            del self.streams[corrid]

        out = np.array([[text.encode("utf-8")]], dtype=object)
        return pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("TRANSCRIPT", out)])

    def execute(self, requests):
        responses = []
        self._sweep()
        for request in requests:
            try:
                responses.append(self._handle(request))
            except Exception as e:
                # lỗi của một sequence không được lây sang các stream khác trong cùng batch (spec §10)
                pb_utils.Logger.log_error(f"{MODEL_NAME}: corrid lỗi: {e}")
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))
                )
        # Sau vòng lặp: chunk có END đã xoá state của nó, số này mới đúng.
        self.metrics.set_ccu(len(self.streams))
        return responses
