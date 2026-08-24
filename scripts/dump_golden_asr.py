# ABOUTME: Dump chuẩn đối chiếu của ASR streaming từ ONNX Runtime ra tests/assets/golden_asr.npz
# ABOUTME: Cần ORT nên phải chạy trong image triton-voice - lệnh cụ thể ở cuối file

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_repository" / "asr_streaming" / "1"))

from client.common import load_wav_16k  # noqa: E402
from streaming_search import (  # noqa: E402
    NUM_MEL_BINS,
    StreamingFbank,
    emitted_tokens,
    greedy_search_step,
    init_search_state,
)

ASSETS = ROOT / "tests" / "assets"
MODEL_DIR = ROOT / "model_repository" / "asr_streaming" / "1"

# Hai mẫu, hai vai. sample_vi 3 giây đủ để bắt lỗi bind sai hay shape lệch.
# sample_vi_long 18 giây mới bắt được sai số fp16 tích luỹ qua state - loại lỗi
# chỉ lộ ra sau vài chục bước encoder, và cũng là loại TensorRT hay gây nhất.
STEMS = ("sample_vi", "sample_vi_long")


def wav_path(stem: str) -> Path:
    return ASSETS / f"{stem}.wav"


def out_path(stem: str) -> Path:
    return ASSETS / f"golden_asr_{stem}.npz"

# Soi gương model_repository/asr_streaming/1/model.py. Lệch một hằng ở đây là
# fixture trông vẫn hợp lệ nhưng mô tả sai pipeline - loại sai khó thấy nhất.
BLANK_ID = 0
CONTEXT_SIZE = 2
LOG_EPS = math.log(1e-10)
SAMPLE_RATE = 16000
CHUNK_MS = 200      # mặc định của client/asr_streaming_client.py
JOINER_HEAD = 16    # giữ nguyên logits của ngần này lần gọi đầu để soi lệch số học

_ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


class Engines:
    """Ba session ORT kèm metadata - đúng những gì initialize() của model.py dựng."""

    def __init__(self, model_dir: Path):
        # Đúng thứ tự provider của model.py. Dump bằng CPU thì fixture mô tả một
        # pipeline không ai chạy: model là fp16, CPU EP không có kernel fp16 nên
        # đi đường khác hẳn CUDA EP. Chuẩn đối chiếu phải là cái production chạy.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        d = Path(model_dir)
        self.encoder = ort.InferenceSession(str(d / "encoder.onnx"), providers=providers)
        self.decoder = ort.InferenceSession(str(d / "decoder.onnx"), providers=providers)
        self.joiner = ort.InferenceSession(str(d / "joiner.onnx"), providers=providers)

        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.decode_chunk_len = int(meta["decode_chunk_len"])

        enc_inputs = self.encoder.get_inputs()
        self.x_name = enc_inputs[0].name
        self.x_dtype = _ORT_TO_NP[enc_inputs[0].type]
        self.T = int(enc_inputs[0].shape[1])

        self.state_in_names = [i.name for i in enc_inputs[1:]]
        assert len(self.state_in_names) == len(self.encoder.get_outputs()) - 1
        self.init_state_specs = [
            (tuple(s if isinstance(s, int) else 1 for s in i.shape), _ORT_TO_NP[i.type])
            for i in enc_inputs[1:]
        ]

        self.decoder_in = self.decoder.get_inputs()[0].name
        self.joiner_in = [i.name for i in self.joiner.get_inputs()]
        self.joiner_dtype = _ORT_TO_NP[self.joiner.get_inputs()[0].type]
        # Provider thực tế ORT chọn, không phải cái ta xin - ghi vào fixture để
        # test biết nó đang so với chuẩn dựng trên nền nào.
        self.provider = self.encoder.get_providers()[0]


def load_engines(model_dir: Path = MODEL_DIR) -> Engines:
    return Engines(model_dir)


def _chunks(wav: np.ndarray, chunk_ms: int = CHUNK_MS):
    """Cắt như client streaming: chunk cuối ngắn hơn vẫn gửi.

    Khác client.common.chunk_wav vốn bỏ phần dư - ở đó là để perf_analyzer khai
    một --shape duy nhất. Fixture thì phải phủ hết audio, kể cả cái đuôi.
    """
    n = SAMPLE_RATE * chunk_ms // 1000
    return [wav[i : i + n] for i in range(0, len(wav), n)]


def _stats(arrays):
    """mean/std/min/max từng state tensor - đủ bắt lỗi bind sai thứ tự mà không phình file.

    Lưu nguyên state là ~1.3MB mỗi bước; với ngần ấy bước thì fixture thành vài
    chục MB trong git. Bốn con số này bắt được đúng loại lỗi cần bắt.
    """
    return np.array(
        [[a.astype(np.float64).mean(), a.astype(np.float64).std(), a.min(), a.max()] for a in arrays],
        dtype=np.float64,
    )


def run_pipeline(engines: Engines, audio: Path) -> dict:
    """Chạy trọn một stream và ghi lại mọi tensor trung gian.

    Logic soi gương _advance/_encoder_step của model.py. Không import model.py
    được vì nó cần triton_python_backend_utils - thứ chỉ tồn tại trong Triton.
    """
    wav = load_wav_16k(audio)
    fbank = StreamingFbank()
    feat = np.zeros((0, NUM_MEL_BINS), dtype=np.float32)
    states = [np.zeros(shape, dtype=dtype) for shape, dtype in engines.init_state_specs]

    enc_chunks, step_index = [], [0]
    dec_outs, joiner_argmax, joiner_head = [], [], []
    first_state_stats = None

    def run_decoder(context):
        y = np.array([context], dtype=np.int64)
        out = engines.decoder.run(None, {engines.decoder_in: y})[0]
        out = out[:, 0, :] if out.ndim == 3 else out
        out = out.astype(np.float32)
        dec_outs.append(out[0])
        return out

    def run_joiner(enc_frame, dec_out):
        feeds = {
            engines.joiner_in[0]: enc_frame.astype(engines.joiner_dtype),
            engines.joiner_in[1]: dec_out.astype(engines.joiner_dtype),
        }
        logits = engines.joiner.run(None, feeds)[0].astype(np.float32)
        flat = logits.reshape(-1)
        joiner_argmax.append(int(np.argmax(flat)))
        if len(joiner_head) < JOINER_HEAD:
            joiner_head.append(flat.copy())
        return logits

    def encoder_step(feat_chunk):
        nonlocal states, first_state_stats
        feeds = {engines.x_name: feat_chunk[None].astype(engines.x_dtype)}
        feeds.update(zip(engines.state_in_names, states))
        outs = engines.encoder.run(None, feeds)
        states = outs[1:]
        if first_state_stats is None:
            first_state_stats = _stats(states)
        enc_out = outs[0][0].astype(np.float32)
        enc_chunks.append(enc_out)
        step_index.append(step_index[-1] + enc_out.shape[0])
        return enc_out

    search = init_search_state(run_decoder, BLANK_ID, CONTEXT_SIZE)

    def advance(new_feat, flush):
        nonlocal feat
        if len(new_feat):
            feat = np.concatenate([feat, new_feat])
        while feat.shape[0] >= engines.T:
            enc_out = encoder_step(feat[: engines.T])
            feat = feat[engines.decode_chunk_len :]
            greedy_search_step(enc_out, search, run_decoder, run_joiner, BLANK_ID, CONTEXT_SIZE)
        if flush and feat.shape[0] > 0:
            pad = np.full((engines.T - feat.shape[0], NUM_MEL_BINS), LOG_EPS, dtype=np.float32)
            enc_out = encoder_step(np.concatenate([feat, pad]))
            feat = feat[:0]
            greedy_search_step(enc_out, search, run_decoder, run_joiner, BLANK_ID, CONTEXT_SIZE)

    parts = _chunks(wav)
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        new_feat = fbank.accept_waveform(part)
        if last:
            tail = fbank.flush()
            if len(tail):
                new_feat = np.concatenate([new_feat, tail]) if len(new_feat) else tail
        advance(new_feat, flush=last)

    return {
        "encoder_out": np.concatenate(enc_chunks, axis=0),
        "encoder_step_index": np.array(step_index, dtype=np.int64),
        "decoder_out": np.stack(dec_outs, axis=0),
        "joiner_argmax": np.array(joiner_argmax, dtype=np.int64),
        "joiner_logits_head": np.stack(joiner_head, axis=0),
        "tokens": np.array(emitted_tokens(search, CONTEXT_SIZE), dtype=np.int64),
        "decode_chunk_len": np.int64(engines.decode_chunk_len),
        "encoder_frames_per_step": np.int64(engines.T),
        "state_stats": first_state_stats,
        "provider": np.array(engines.provider, dtype="<U64"),
        "state_shapes_json": np.array(
            json.dumps([[list(s.shape), str(s.dtype)] for s in states]), dtype="<U8192"
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Không dùng choices=: với nargs="*" argparse đem cả cái default đi so với
    # choices rồi trượt, kể cả khi default là list rỗng. Tự kiểm cho xong.
    ap.add_argument("stems", nargs="*", default=[], help=f"mặc định: {' '.join(STEMS)}")
    args = ap.parse_args()
    stems = args.stems or list(STEMS)
    unknown = [x for x in stems if x not in STEMS]
    if unknown:
        ap.error(f"không biết mẫu {unknown} - có: {list(STEMS)}")

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_DIR / "bpe.model"))

    engines = load_engines()
    print(f"provider        {engines.provider}")
    print(f"T / chunk_len   {engines.T} / {engines.decode_chunk_len}")

    for stem in stems:
        audio = wav_path(stem)
        result = run_pipeline(engines, audio)
        text = sp.decode([int(t) for t in result["tokens"]])

        out = out_path(stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **result)

        print(f"\n{audio.name}")
        print(f"  encoder steps {len(result['encoder_step_index']) - 1}")
        print(f"  encoder_out   {result['encoder_out'].shape}")
        print(f"  decoder calls {result['decoder_out'].shape[0]}")
        print(f"  joiner calls  {result['joiner_argmax'].shape[0]}")
        print(f"  tokens        {result['tokens'].shape[0]}")
        print(f"  transcript    {text}")
        print(f"  đã ghi        {out.name}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

# Chạy:
#   docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD:/w" -w /w triton-voice \
#       python3 scripts/dump_golden_asr.py
