# ABOUTME: Triton BLS - encoder gọi qua model riêng, cache do Triton giữ theo correlation_id
# ABOUTME: execute() async: encoder chạy song song, greedy gom batch trong process này

import asyncio
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
LOG_EPS = math.log(1e-10)
STATE_TTL_S = 60.0          # soi gương max_sequence_idle_microseconds ở CẢ HAI config
SAMPLE_RATE = 16000
DEC_CACHE_MAX = 4096
MODEL_NAME = "asr_bls"
ENCODER_MODEL = "encoder"
ORT_INTRA_THREADS = 2

_ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


class _Stream:
    """State của một stream đang sống.

    Cache encoder KHÔNG nằm ở đây - Triton giữ nó theo correlation_id (khối
    `state` trong encoder/config.pbtxt). Mỗi bước chỉ truyền x và nhận
    encoder_out: 2 tensor thay vì 148.
    """

    def __init__(self, model):
        self.fbank = StreamingFbank()
        self.feat = np.zeros((0, NUM_MEL_BINS), dtype=np.float32)
        self.search = init_search_state(model.run_decoder_one, BLANK_ID, CONTEXT_SIZE)
        self.last_seen = time.monotonic()
        self.started = False    # đã gửi SEQUENCE_START xuống encoder chưa
        self.text = ""
        self.n_emitted = 0


class _Ctx:
    __slots__ = ("idx", "corrid", "stream", "end", "audio_s", "t0", "pending")

    def __init__(self, idx, corrid, stream, end, audio_s, t0, pending):
        self.idx = idx
        self.corrid = corrid
        self.stream = stream
        self.end = end
        self.audio_s = audio_s
        self.t0 = t0
        self.pending = pending


class TritonPythonModel:
    def initialize(self, args):
        # Trọng số nằm ngay trong version dir của chính model này - asr_streaming
        # (bản monolith cũ) đã bị xoá sau khi BLS trở thành kiến trúc chốt.
        d = os.path.dirname(os.path.abspath(__file__))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ORT_INTRA_THREADS
        opts.inter_op_num_threads = 1
        opts.enable_mem_pattern = False

        # decoder/joiner chạy CPU EP: tensor (B,512) quá bé để bù H2D + launch +
        # sync. Quan trọng hơn, nhờ vậy process BLS KHÔNG cần CUDA context nào -
        # instance_group để KIND_CPU được, và mỗi instance tiết kiệm ~500 MiB.
        # Đặt ASR_SMALL_ON_GPU=1 để đối chiếu.
        if os.environ.get("ASR_SMALL_ON_GPU"):
            small = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        else:
            small = ["CPUExecutionProvider"]
        self.decoder = ort.InferenceSession(
            os.path.join(d, "decoder.onnx"), opts, providers=small)
        self.joiner = ort.InferenceSession(
            os.path.join(d, "joiner.onnx"), opts, providers=small)

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(d, "bpe.model"))

        # Chỉ cần metadata và shape của encoder, KHÔNG chạy nó ở đây. Mở bằng
        # CPU provider rồi bỏ session ngay: giữ session sống là tốn thêm một
        # CUDA context cho mỗi instance mà không dùng đến.
        probe_opts = ort.SessionOptions()
        probe_opts.log_severity_level = 3
        probe = ort.InferenceSession(
            os.path.join(d, "encoder.onnx"), probe_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = probe.get_modelmeta().custom_metadata_map
        self.decode_chunk_len = int(meta["decode_chunk_len"])
        enc_in = probe.get_inputs()[0]
        self.x_name = enc_in.name
        self.x_dtype = _ORT_TO_NP[enc_in.type]
        self.T = int(enc_in.shape[1])   # T - decode_chunk_len là lookahead
        del probe

        self.decoder_in = self.decoder.get_inputs()[0].name
        self.joiner_in = [i.name for i in self.joiner.get_inputs()]
        self.joiner_dtype = _ORT_TO_NP[self.joiner.get_inputs()[0].type]

        self._dec_cache = {}
        self.streams = {}
        self.metrics = ModelMetrics(
            pb_utils, MODEL_NAME, args["model_instance_name"], ASR_RTF_BUCKETS
        )

    # ---------- decoder / joiner ----------

    def run_decoder_batch(self, ctx):
        keys = [(int(r[0]), int(r[1])) for r in ctx]
        todo = []
        for k in keys:
            if k not in self._dec_cache and k not in todo:
                todo.append(k)
        if todo:
            out = self.decoder.run(
                None, {self.decoder_in: np.asarray(todo, dtype=np.int64)}
            )[0]
            out = out[:, 0, :] if out.ndim == 3 else out
            out = out.astype(np.float32)
            if len(self._dec_cache) + len(todo) > DEC_CACHE_MAX:
                self._dec_cache.clear()
            for k, v in zip(todo, out):
                self._dec_cache[k] = v
        return np.stack([self._dec_cache[k] for k in keys])

    def run_decoder_one(self, context):
        return self.run_decoder_batch(np.asarray([context], dtype=np.int64))

    def run_joiner_batch(self, enc, dec):
        return self.joiner.run(None, {
            self.joiner_in[0]: enc.astype(self.joiner_dtype),
            self.joiner_in[1]: dec.astype(self.joiner_dtype),
        })[0].astype(np.float32)

    # ---------- encoder qua BLS ----------

    async def _encoder_step(self, stream, corrid, feat_chunk, end=False):
        """Một bước encoder. State do Triton giữ theo corrid.

        Cờ START phải đi cùng lần gọi ĐẦU TIÊN của stream, END cùng lần cuối.
        Quên END thì slot state trong Triton không bao giờ được giải phóng: số
        sequence đồng thời bò dần tới max_candidate_sequences rồi stream mới
        treo, và không có gì trong log tố cáo nguyên nhân.
        """
        flags = 0
        if not stream.started:
            flags |= pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_START
            stream.started = True
        if end:
            flags |= pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_END

        resp = await pb_utils.InferenceRequest(
            model_name=ENCODER_MODEL,
            requested_output_names=["encoder_out"],
            inputs=[pb_utils.Tensor(self.x_name, feat_chunk[None].astype(self.x_dtype))],
            correlation_id=corrid,
            flags=flags,
        ).async_exec()
        if resp.has_error():
            raise RuntimeError(f"encoder: {resp.error().message()}")
        # encoder khai max_batch_size: 0 nên output giữ nguyên chiều 1 ở đầu
        return pb_utils.get_output_tensor_by_name(resp, "encoder_out").as_numpy()[0]

    async def _close(self, stream, corrid):
        """Gửi một bước END rỗng để Triton giải phóng slot state."""
        if not stream.started:
            return
        try:
            await self._encoder_step(
                stream, corrid,
                np.full((self.T, NUM_MEL_BINS), LOG_EPS, dtype=np.float32),
                end=True,
            )
        except Exception as e:   # noqa: BLE001
            pb_utils.Logger.log_warn(f"{MODEL_NAME}: failed to close corrid={corrid}: {e}")

    async def _run_rounds(self, ctxs):
        """Mỗi vòng: gọi encoder song song cho mọi stream còn việc, rồi greedy batch.

        Hai pha tách bạch là có chủ ý. Nếu để mỗi request tự await encoder rồi
        tự chạy greedy (kiểu asyncio.gather trên cả _handle) thì greedy thành
        batch-1 và mất đúng phần tiết kiệm lớn nhất. Gom encoder trước, greedy
        sau, thì vừa có concurrency cho encoder vừa có batch cho joiner.
        """
        rnd = 0
        while True:
            batch = [c for c in ctxs if len(c.pending) > rnd]
            if not batch:
                return
            encs = await asyncio.gather(*[
                self._encoder_step(
                    c.stream, c.corrid, c.pending[rnd],
                    end=c.end and rnd == len(c.pending) - 1,
                )
                for c in batch
            ])
            greedy_search_batch(
                np.stack(encs), [c.stream.search for c in batch],
                self.run_decoder_batch, self.run_joiner_batch,
                BLANK_ID, CONTEXT_SIZE,
            )
            rnd += 1

    # ---------- vòng đời stream ----------

    def _sweep(self):
        """Dọn stream chết không gửi END.

        Khác bản monolith ở chỗ: ngoài dict python còn phải BÁO TRITON giải
        phóng slot state của encoder. Bỏ bước đó thì python sạch mà Triton giữ
        cache vĩnh viễn - rò rỉ chỉ lộ ra khi chạm max_candidate_sequences, tức
        đúng ở mức tải mà ta đang cố nới lên.

        Đóng slot chạy nền: việc dọn stream chết không được chặn đường stream
        đang sống.
        """
        now = time.monotonic()
        for k in [k for k, s in self.streams.items() if now - s.last_seen > STATE_TTL_S]:
            pb_utils.Logger.log_warn(f"{MODEL_NAME}: evicting orphaned state corrid={k}")
            stream = self.streams.pop(k)
            asyncio.create_task(self._close(stream, k))

    @staticmethod
    def _flag(request, name):
        t = pb_utils.get_input_tensor_by_name(request, name)
        return t is not None and bool(t.as_numpy().reshape(-1)[0])

    def _prepare(self, idx, request):
        corrid = int(
            pb_utils.get_input_tensor_by_name(request, "CORRID").as_numpy().reshape(-1)[0]
        )
        start = self._flag(request, "START")
        end = self._flag(request, "END")

        if start or corrid not in self.streams:
            if not start:
                pb_utils.Logger.log_warn(
                    f"{MODEL_NAME}: chunk has no state (corrid={corrid}), reinitializing"
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
        if end:
            # LUÔN phát một bước cuối khi END, kể cả feat rỗng - đây là chỗ bản
            # BLS cũ rò slot: nó bọc khối này trong `if feat.shape[0] > 0` nên
            # chunk END không kèm audio, hoặc audio chia hết cho decode_chunk_len,
            # thì encoder không bao giờ nhận cờ END. Một bước inference thừa rẻ
            # hơn nhiều so với một slot rò vĩnh viễn.
            pad = np.full(
                (self.T - stream.feat.shape[0], NUM_MEL_BINS), LOG_EPS, dtype=np.float32
            )
            pending.append(np.concatenate([stream.feat, pad]))
            stream.feat = stream.feat[:0]

        return _Ctx(idx, corrid, stream, end, len(audio) / SAMPLE_RATE, t0, pending)

    def _finish(self, ctx):
        stream = ctx.stream
        toks = emitted_tokens(stream.search, CONTEXT_SIZE)
        if len(toks) != stream.n_emitted:
            stream.text = self.sp.decode(toks)
            stream.n_emitted = len(toks)
        text = stream.text

        if ctx.audio_s > 0:
            self.metrics.observe_rtf(time.perf_counter() - ctx.t0, ctx.audio_s)
        if ctx.end:
            # _run_rounds đã gửi cờ END xuống encoder ở bước cuối
            self.streams.pop(ctx.corrid, None)

        out = np.array([[text.encode("utf-8")]], dtype=object)
        return pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("TRANSCRIPT", out)])

    # ---------- entry point ----------

    async def execute(self, requests):
        self._sweep()
        responses = [None] * len(requests)
        ctxs = []

        for i, request in enumerate(requests):
            try:
                ctxs.append(self._prepare(i, request))
            except Exception as e:   # noqa: BLE001
                pb_utils.Logger.log_error(f"{MODEL_NAME}: corrid failed (prepare): {e}")
                responses[i] = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))

        try:
            await self._run_rounds(ctxs)
        except Exception as e:   # noqa: BLE001
            # Hỏng giữa chừng thì không biết stream nào đã đi tới đâu. Đóng slot
            # encoder cho TẤT CẢ rồi xoá state - giữ lại là rủi ro cache trôi.
            pb_utils.Logger.log_error(f"{MODEL_NAME}: whole batch failed: {e}")
            for c in ctxs:
                s = self.streams.pop(c.corrid, None)
                if s is not None:
                    asyncio.create_task(self._close(s, c.corrid))
                responses[c.idx] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                )
            self.metrics.set_ccu(len(self.streams))
            return responses

        for c in ctxs:
            try:
                responses[c.idx] = self._finish(c)
            except Exception as e:   # noqa: BLE001
                pb_utils.Logger.log_error(f"{MODEL_NAME}: corrid failed (finish): {e}")
                s = self.streams.pop(c.corrid, None)
                if s is not None:
                    asyncio.create_task(self._close(s, c.corrid))
                responses[c.idx] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                )

        self.metrics.set_ccu(len(self.streams))
        return responses
