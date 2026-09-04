# ABOUTME: Hằng số và hàm dùng chung cho client, bench và test
# ABOUTME: Mọi nơi phải dùng cùng SAMPLE_RATE - lệch một chỗ là ASR nghe sai tốc độ

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16000


def load_wav_16k(path) -> np.ndarray:
    """Đọc file wav bất kỳ về mono 16kHz float32 - tần số duy nhất ASR nhận.

    TTS xuất ra 24kHz, nên muốn đưa audio vừa sinh quay lại ASR để kiểm tra thì
    bắt buộc phải hạ tần số. resample_poly có lọc chống aliasing sẵn.
    """
    wav, sample_rate = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)   # trộn stereo về mono
    if sample_rate != SAMPLE_RATE:
        wav = resample_poly(wav, SAMPLE_RATE, sample_rate)
    return np.asarray(wav, dtype=np.float32)


def chunk_wav(wav: np.ndarray, chunk_ms: int = 200) -> list[np.ndarray]:
    """Cắt waveform thành các chunk CÙNG độ dài, bỏ phần dư cuối.

    Khác cách client streaming cắt: ở đó chunk cuối ngắn hơn cũng gửi được, còn
    perf_analyzer khai `--shape` một lần cho cả sequence nên mọi request phải
    đúng một cỡ. Thà bỏ phần dư (<chunk_ms) còn hơn đệm im lặng vào - đệm sẽ
    thêm khung blank giả và làm lệch chính thứ đang đo là vòng greedy search.
    """
    chunk = SAMPLE_RATE * chunk_ms // 1000
    if chunk <= 0:
        raise ValueError(f"chunk_ms={chunk_ms} yields a {chunk}-sample chunk")
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return [wav[i : i + chunk] for i in range(0, len(wav) - chunk + 1, chunk)]
