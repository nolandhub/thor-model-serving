# ABOUTME: Image ASR streaming cho Jetson AGX Thor - dựng trên base Triton nội bộ đã chạy thật trên máy này
# ABOUTME: Công thức torch/onnxruntime-gpu và LD_LIBRARY_PATH lấy từ voice-agent-deployment/asr-triton (healthy 12+ ngày, cùng phần cứng)

# Image này KHÔNG có trên Docker Hub / NGC - nó đã nằm sẵn trong `docker images`
# trên Thor (build cục bộ theo layer: cuda→python→numpy→cmake→onnx→pytorch→
# pybind11→triton→tritonserver, khớp L4T R38.4.0 + CUDA 13.0 của máy). Nếu build
# trên một con Thor khác chưa có image này, phải tự dựng lại chuỗi layer đó trước.
ARG BASE_IMAGE=tritonserver:r38.4.arm64-sbsa-cu130-24.04
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Không dùng PyPI thường: wheel torch/onnxruntime-gpu ở đó không có bản aarch64
# CUDA 13. Index này là nơi duy nhất đã kiểm chứng có cả hai trên chính Thor.
RUN python3 -m pip install --no-cache-dir \
        torch==2.9.1 torchaudio==2.9.1 onnxruntime-gpu \
        --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130

# bpe.model của repo mình decode bằng sentencepiece - không có trong base image.
RUN python3 -m pip install --no-cache-dir sentencepiece==0.2.0

# Stub Python backend link nhầm libpython hệ thống (3.12.3) thay vì bản standalone
# (3.12.13) đang chứa site-packages vừa cài ở trên - hai bên bất đồng module nào
# là builtin (_contextvars), nên MỌI Python backend model gãy ngay lúc import
# numpy. Không tự suy ra được từ lỗi; phát hiện từ voice-agent-deployment.
ENV LD_LIBRARY_PATH=/opt/python/cpython-3.12-linux-aarch64-gnu/lib:${LD_LIBRARY_PATH}

# Chốt lại: mọi import phải sạch, và CUDAExecutionProvider phải THẬT SỰ có mặt -
# thiếu nó ORT âm thầm rơi về CPU, ASR vẫn chạy nhưng chậm gấp nhiều lần, không
# ai biết cho đến khi đo latency.
RUN python3 -c "\
import numpy, torch, torchaudio, onnxruntime, sentencepiece; \
providers = onnxruntime.get_available_providers(); \
assert 'CUDAExecutionProvider' in providers, f'CUDA EP thiếu: {providers}'; \
print('deps ok, providers =', providers)"
