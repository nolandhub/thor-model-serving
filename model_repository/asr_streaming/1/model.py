# ABOUTME: Triton Python backend - streaming ASR: fbank dần + encoder streaming + greedy theo chunk
# ABOUTME: State per-stream khoá theo CORRID; logic thuần nằm ở streaming_search.py

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

_ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


class _Stream:
    """State của một stream đang sống - mọi thứ phải nhớ giữa hai chunk."""

    def __init__(self, model):
        self.fbank = StreamingFbank()
        self.feat = np.zeros((0, NUM_MEL_BINS), dtype=np.float32)
        self.enc_states = [
            np.zeros(shape, dtype=dtype) for shape, dtype in model.init_state_specs
        ]
        self.search = init_search_state(model.run_decoder, BLANK_ID, CONTEXT_SIZE)
        self.last_seen = time.monotonic()


class TritonPythonModel:
    def initialize(self, args):
        d = os.path.join(args["model_repository"], args["model_version"])
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.encoder = ort.InferenceSession(os.path.join(d, "encoder.onnx"), providers=providers)
        self.decoder = ort.InferenceSession(os.path.join(d, "decoder.onnx"), providers=providers)
        self.joiner = ort.InferenceSession(os.path.join(d, "joiner.onnx"), providers=providers)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(d, "bpe.model"))

        # Nhịp tiêu thụ khung nằm trong metadata của chính file ONNX (kiểm ở plan Task 1)
        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.decode_chunk_len = int(meta["decode_chunk_len"])

        enc_inputs = self.encoder.get_inputs()
        self.x_name = enc_inputs[0].name
        self.x_dtype = _ORT_TO_NP[enc_inputs[0].type]
        self.T = int(enc_inputs[0].shape[1])   # khung đưa vào mỗi bước; T - decode_chunk_len là lookahead

        # State của encoder: mọi input trừ x, khớp THEO VỊ TRÍ với mọi output trừ encoder_out.
        # Không viết cứng tên/số lượng - export đổi layer thì code vẫn đúng.
        self.state_in_names = [i.name for i in enc_inputs[1:]]
        self.state_out_count = len(self.encoder.get_outputs()) - 1
        assert len(self.state_in_names) == self.state_out_count
        self.init_state_specs = [
            (tuple(s if isinstance(s, int) else 1 for s in i.shape), _ORT_TO_NP[i.type])
            for i in enc_inputs[1:]
        ]

        self.decoder_in = self.decoder.get_inputs()[0].name
        self.joiner_in = [i.name for i in self.joiner.get_inputs()]
        self.joiner_dtype = _ORT_TO_NP[self.joiner.get_inputs()[0].type]

        self.streams = {}   # corrid -> _Stream
        self.metrics = ModelMetrics(
            pb_utils, "asr_streaming", args["model_instance_name"], ASR_RTF_BUCKETS
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

    def _encoder_step(self, stream, feat_chunk):
        """Một bước encoder streaming: (T, 80) + cache cũ -> (T', C) + cache mới."""
        feeds = {self.x_name: feat_chunk[None].astype(self.x_dtype)}
        feeds.update(zip(self.state_in_names, stream.enc_states))
        outs = self.encoder.run(None, feeds)
        stream.enc_states = outs[1:]
        return outs[0][0].astype(np.float32)

    def _advance(self, stream, new_feat, flush):
        """Nạp khung mới, chạy encoder đủ số bước, đi tiếp greedy search."""
        if len(new_feat):
            stream.feat = np.concatenate([stream.feat, new_feat])
        while stream.feat.shape[0] >= self.T:
            enc_out = self._encoder_step(stream, stream.feat[: self.T])
            stream.feat = stream.feat[self.decode_chunk_len :]
            greedy_search_step(
                enc_out, stream.search, self.run_decoder, self.run_joiner, BLANK_ID, CONTEXT_SIZE
            )
        if flush and stream.feat.shape[0] > 0:
            # khung cuối không đủ T - đệm LOG_EPS cho đủ một bước encoder chót
            pad = np.full(
                (self.T - stream.feat.shape[0], NUM_MEL_BINS), LOG_EPS, dtype=np.float32
            )
            enc_out = self._encoder_step(stream, np.concatenate([stream.feat, pad]))
            stream.feat = stream.feat[:0]
            greedy_search_step(
                enc_out, stream.search, self.run_decoder, self.run_joiner, BLANK_ID, CONTEXT_SIZE
            )

    def _sweep(self):
        """Xoá state của stream chết không gửi END - nếu không dict rò rỉ vĩnh viễn."""
        now = time.monotonic()
        for k in [k for k, s in self.streams.items() if now - s.last_seen > STATE_TTL_S]:
            pb_utils.Logger.log_warn(f"asr_streaming: xoá state mồ côi corrid={k}")
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
                    f"asr_streaming: chunk không có state (corrid={corrid}), khởi tạo lại"
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
        self._advance(stream, new_feat, flush=end)
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
                pb_utils.Logger.log_error(f"asr_streaming: corrid lỗi: {e}")
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))
                )
        # Sau vòng lặp: chunk có END đã xoá state của nó, số này mới đúng.
        self.metrics.set_ccu(len(self.streams))
        return responses
