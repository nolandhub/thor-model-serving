# ABOUTME: Base asr-triton:latest - image của đồng nghiệp, đã có torch/onnxruntime-gpu/libsndfile1
# ABOUTME: KHÔNG cài thêm gì (proxy công ty chặn apt/pip), chỉ kiểm CUDA EP có mặt lúc build

ARG BASE_IMAGE=asr-triton:latest
FROM ${BASE_IMAGE}

# Các dòng nối \ KHÔNG được thụt đầu dòng: khoảng trắng đó nằm nguyên trong
# chuỗi truyền cho python3 -c, và Python báo IndentationError ngay dòng đầu.
RUN python3 -c "import numpy, torch, torchaudio, onnxruntime, sentencepiece; \
providers = onnxruntime.get_available_providers(); \
assert 'CUDAExecutionProvider' in providers, f'CUDA EP missing: {providers}'; \
print('deps ok, providers =', providers)"
