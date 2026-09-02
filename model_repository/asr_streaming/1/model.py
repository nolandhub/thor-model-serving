# ABOUTME: Triton Python backend - streaming ASR, encoder VÀ greedy đều chạy theo batch
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
    greedy_search_batch,
    init_search_state,
)

sys.path.insert(0, "/opt/serving")
from serving.metrics import ASR_RTF_BUCKETS, ModelMetrics  # noqa: E402

BLANK_ID = 0
CONTEXT_SIZE = 2
LOG_EPS = math.log(1e-10)   # giá trị đệm khung cuối, theo quy ước icefall/sherpa
STATE_TTL_S = 60.0          # soi gương max_sequence_idle_microseconds trong config.pbtxt
SAMPLE_RATE = 16000
DEC_CACHE_MAX = 4096

# Số luồng cho MỖI instance. Thor có 14 nhân; để ORT tự chọn thì mỗi process lấy
# cả 14, N instance thành 14N luồng tranh nhau. Biểu hiện: CPU 100%, GPU util
# thấp, latency phình mà không rõ nguyên nhân.
ORT_INTRA_THREADS = 2

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
        # Giữ NGUYÊN chiều batch cỡ 1 trong mỗi state, ở đúng trục của nó. Nhờ
        # vậy gom batch chỉ là concatenate, tách chỉ là split - không phải chèn
        # hay bỏ chiều, chỗ rất dễ sai vì 74 tensor có hai trục batch khác nhau.
        self.enc_states = [
            np.zeros(shape, dtype=dtype) for shape, dtype in model.init_state_specs
        ]
        self.search = init_search_state(model.run_decoder_one, BLANK_ID, CONTEXT_SIZE)
        self.last_seen = time.monotonic()
        self.text = ""        # transcript đã dựng, chỉ dựng lại khi có token mới
        self.n_emitted = 0


class _Ctx:
    """Một request trong batch hiện tại, sống đúng một lần gọi execute()."""

    __slots__ = ("idx", "corrid", "stream", "end", "audio_s", "t0", "pending")

    def __init__(self, idx, corrid, stream, end, audio_s, t0, pending):
        self.idx = idx
        self.corrid = corrid
        self.stream = stream
        self.end = end
        self.audio_s = audio_s
        self.t0 = t0
        self.pending = pending    # list chunk (T, 80) chờ qua encoder


class TritonPythonModel:
    def initialize(self, args):
        d = os.path.join(args["model_repository"], args["model_version"])

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ORT_INTRA_THREADS
        opts.inter_op_num_threads = 1
        # Batch size đổi giữa các lần execute (1..max_batch_size). Mem pattern
        # chỉ đúng khi shape cố định; bật với shape động thì ORT phải huỷ và
        # dựng lại plan mỗi lần B đổi.
        opts.enable_mem_pattern = False

        gpu = [("CUDAExecutionProvider", {
            "device_id": 0,
            "cudnn_conv_algo_search": "HEURISTIC",   # mặc định EXHAUSTIVE, warmup rất lâu
            "do_copy_in_default_stream": True,
        }), "CPUExecutionProvider"]

        self.encoder = ort.InferenceSession(
            os.path.join(d, "encoder.onnx"), opts, providers=gpu)
        if self.encoder.get_providers()[0] != "CUDAExecutionProvider":
            # Rơi về CPU âm thầm là kịch bản tệ nhất: chạy đúng, chậm gấp bội,
            # và không có gì trong log nói ra. Nổ ngay lúc load thay vì để bench
            # cho ra số vô nghĩa.
            raise RuntimeError(
                f"encoder không chạy CUDA EP: {self.encoder.get_providers()}"
            )

        # decoder (B,2)->(B,512) và joiner (B,512)+(B,512)->(B,2000) đều quá bé
        # để bù chi phí H2D + launch + sync của CUDA. Trên 14 nhân ARM, CPU EP
        # thường nhanh hơn. Đây là giả thuyết cần đo - đặt ASR_SMALL_ON_GPU=1 để
        # đối chiếu.
        small = gpu if os.environ.get("ASR_SMALL_ON_GPU") else ["CPUExecutionProvider"]
        self.decoder = ort.InferenceSession(
            os.path.join(d, "decoder.onnx"), opts, providers=small)
        self.joiner = ort.InferenceSession(
            os.path.join(d, "joiner.onnx"), opts, providers=small)

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(d, "bpe.model"))

        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.decode_chunk_len = int(meta["decode_chunk_len"])

        enc_inputs = self.encoder.get_inputs()
        self.x_name = enc_inputs[0].name
        self.x_dtype = _ORT_TO_NP[enc_inputs[0].type]
        self.T = int(enc_inputs[0].shape[1])

        # State: mọi input trừ x, khớp THEO VỊ TRÍ với mọi output trừ encoder_out.
        self.state_in_names = [i.name for i in enc_inputs[1:]]
        assert len(self.state_in_names) == len(self.encoder.get_outputs()) - 1

        # Trục batch KHÔNG đồng nhất trong zipformer2: cache attention là
        # (left_context, N, dim) còn cache conv là (N, dim, kernel), và
        # processed_lens là (N,) rank-1. Suy trục từ vị trí chiều symbolic thay
        # vì viết cứng theo tên: export đổi thì code vẫn đúng, và shape mơ hồ
        # thì nổ lúc load chứ không lặng lẽ trộn state giữa các stream lúc chạy.
        self.state_batch_axis = []
        for i in enc_inputs[1:]:
            dyn = [k for k, s in enumerate(i.shape) if not isinstance(s, int)]
            if len(dyn) != 1:
                raise RuntimeError(
                    f"{i.name}: shape {i.shape} có {len(dyn)} chiều động, "
                    "không xác định được trục batch"
                )
            self.state_batch_axis.append(dyn[0])

        self.init_state_specs = [
            (tuple(s if isinstance(s, int) else 1 for s in i.shape), _ORT_TO_NP[i.type])
            for i in enc_inputs[1:]
        ]

        self.decoder_in = self.decoder.get_inputs()[0].name
        self.dec_dim = int(self.decoder.get_outputs()[0].shape[-1])
        self.joiner_in = [i.name for i in self.joiner.get_inputs()]
        self.joiner_dtype = _ORT_TO_NP[self.joiner.get_inputs()[0].type]

        self._dec_cache = {}
        self.streams = {}   # corrid -> _Stream
        self.metrics = ModelMetrics(
            pb_utils, "asr_streaming", args["model_instance_name"], ASR_RTF_BUCKETS
        )

    # ---------- decoder / joiner ----------

    def run_decoder_batch(self, ctx):
        """(B, context_size) int64 -> (B, C) float32, có cache và khử trùng lặp.

        decoder stateless với context_size=2 nên output chỉ phụ thuộc 2 token
        cuối. Trong một batch các stream rất hay trùng context, và giữa các
        chunk thì tiếng Việt lặp cặp BPE nhiều - cache ăn phần lớn lượt gọi.
        """
        keys = [(int(r[0]), int(r[1])) for r in ctx]
        todo = []
        for k in keys:
            if k not in self._dec_cache and k not in todo:
                todo.append(k)
        if todo:
            out = self.decoder.run(
                None, {self.decoder_in: np.asarray(todo, dtype=np.int64)}
            )[0]
            # Một số bản export trả (N, 1, C), bỏ chiều giữa cho khớp joiner
            out = out[:, 0, :] if out.ndim == 3 else out
            out = out.astype(np.float32)
            if len(self._dec_cache) + len(todo) > DEC_CACHE_MAX:
                self._dec_cache.clear()
            for k, v in zip(todo, out):
                self._dec_cache[k] = v
        return np.stack([self._dec_cache[k] for k in keys])

    def run_decoder_one(self, context):
        """Đường batch-1 cho init_search_state; trả (1, C) đúng shape SearchState."""
        return self.run_decoder_batch(np.asarray([context], dtype=np.int64))

    def run_joiner_batch(self, enc, dec):
        """(B, C) + (B, C) -> (B, V)."""
        return self.joiner.run(None, {
            self.joiner_in[0]: enc.astype(self.joiner_dtype),
            self.joiner_in[1]: dec.astype(self.joiner_dtype),
        })[0].astype(np.float32)

    # ---------- encoder ----------

    def _encoder_batch(self, streams, chunks):
        """Một bước encoder cho B stream cùng lúc -> (B, T', C).

        B=1 đi đường tắt: concatenate/split trên 74 tensor tốn hơn phần tiết
        kiệm được khi chỉ có một stream.
        """
        b = len(streams)
        feeds = {self.x_name: np.stack(chunks).astype(self.x_dtype)}
        if b == 1:
            st = streams[0].enc_states
            for j, name in enumerate(self.state_in_names):
                feeds[name] = st[j]
        else:
            for j, name in enumerate(self.state_in_names):
                feeds[name] = np.concatenate(
                    [s.enc_states[j] for s in streams], axis=self.state_batch_axis[j]
                )

        outs = self.encoder.run(None, feeds)

        if b == 1:
            streams[0].enc_states = list(outs[1:])
        else:
            for j in range(len(self.state_in_names)):
                ax = self.state_batch_axis[j]
                for k, part in enumerate(np.split(outs[j + 1], b, axis=ax)):
                    # ascontiguousarray: split trả về view; giữ view thì cả
                    # tensor batch B bị neo trong RAM suốt vòng đời stream
                    streams[k].enc_states[j] = np.ascontiguousarray(part)
        return outs[0].astype(np.float32)

    def _run_rounds(self, ctxs):
        """Chạy encoder rồi greedy theo vòng, mỗi vòng gom mọi stream còn việc.

        Các stream trong một batch không nhất thiết cần cùng số bước encoder:
        chunk client gửi có thể lệch nhịp decode_chunk_len, hoặc chunk END mang
        thêm phần đuôi. Vòng nào cũng chỉ gom stream còn chunk chờ, nên stream
        ngắn không kéo dài stream khác.

        Với --chunk-ms = decode_chunk_len*10 (320ms) thì hầu như luôn một vòng.
        """
        rnd = 0
        while True:
            batch = [c for c in ctxs if len(c.pending) > rnd]
            if not batch:
                return
            streams = [c.stream for c in batch]
            enc = self._encoder_batch(streams, [c.pending[rnd] for c in batch])
            greedy_search_batch(
                enc, [s.search for s in streams],
                self.run_decoder_batch, self.run_joiner_batch,
                BLANK_ID, CONTEXT_SIZE,
            )
            rnd += 1

    # ---------- vòng đời stream ----------

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

    def _prepare(self, idx, request):
        """Phần CPU thuần của một request: fbank, cắt thành các chunk chờ encoder."""
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
        if len(new_feat):
            stream.feat = np.concatenate([stream.feat, new_feat])

        pending = []
        while stream.feat.shape[0] >= self.T:
            pending.append(stream.feat[: self.T])
            stream.feat = stream.feat[self.decode_chunk_len :]
        if end and stream.feat.shape[0] > 0:
            # khung cuối không đủ T - đệm LOG_EPS cho đủ một bước encoder chót
            pad = np.full(
                (self.T - stream.feat.shape[0], NUM_MEL_BINS), LOG_EPS, dtype=np.float32
            )
            pending.append(np.concatenate([stream.feat, pad]))
            stream.feat = stream.feat[:0]

        return _Ctx(idx, corrid, stream, end, len(audio) / SAMPLE_RATE, t0, pending)

    def _finish(self, ctx):
        """Dựng transcript và trả response. Greedy đã chạy xong ở _run_rounds."""
        stream = ctx.stream

        # sp.decode cả hyp mỗi chunk là O(n^2) theo độ dài stream; chunk im lặng
        # (phần lớn) không sinh token nên khỏi dựng lại.
        toks = emitted_tokens(stream.search, CONTEXT_SIZE)
        if len(toks) != stream.n_emitted:
            stream.text = self.sp.decode(toks)
            stream.n_emitted = len(toks)
        text = stream.text

        # t0 đặt ở _prepare nên RTF này gồm cả thời gian chờ batch encoder. Bản
        # cũ chỉ đo compute riêng lẻ nên RTF đẹp trong khi latency thật xấu.
        if ctx.audio_s > 0:
            self.metrics.observe_rtf(time.perf_counter() - ctx.t0, ctx.audio_s)

        if ctx.end:
            self.streams.pop(ctx.corrid, None)

        out = np.array([[text.encode("utf-8")]], dtype=object)
        return pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("TRANSCRIPT", out)])

    # ---------- entry point ----------

    def execute(self, requests):
        self._sweep()
        responses = [None] * len(requests)
        ctxs = []

        for i, request in enumerate(requests):
            try:
                ctxs.append(self._prepare(i, request))
            except Exception as e:   # noqa: BLE001
                # lỗi của một sequence không được lây sang stream khác trong cùng batch (spec §10)
                pb_utils.Logger.log_error(f"asr_streaming: corrid lỗi (prepare): {e}")
                responses[i] = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))

        try:
            self._run_rounds(ctxs)
        except Exception as e:   # noqa: BLE001
            # Encoder hoặc greedy hỏng là hỏng cả batch: không biết state của
            # stream nào đã cập nhật đến đâu, giữ lại là rủi ro sai âm thầm.
            # Xoá sạch để lượt sau khởi tạo lại từ đầu.
            pb_utils.Logger.log_error(f"asr_streaming: lỗi cả batch: {e}")
            for c in ctxs:
                self.streams.pop(c.corrid, None)
                responses[c.idx] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                )
            self.metrics.set_ccu(len(self.streams))
            return responses

        for c in ctxs:
            try:
                responses[c.idx] = self._finish(c)
            except Exception as e:   # noqa: BLE001
                pb_utils.Logger.log_error(f"asr_streaming: corrid lỗi (finish): {e}")
                self.streams.pop(c.corrid, None)
                responses[c.idx] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                )

        # Sau vòng lặp: chunk có END đã xoá state của nó, số này mới đúng.
        self.metrics.set_ccu(len(self.streams))
        return responses
