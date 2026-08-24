# ABOUTME: Image ASR streaming cho Jetson AGX Thor - base thẳng trên asr-triton:latest (đồng nghiệp, sản xuất 12+ ngày)
# ABOUTME: Không apt-get/pip gì thêm - mọi dependency cần đã có sẵn trong base image, né hẳn proxy công ty chặn ports.ubuntu.com/nvidia

# asr-triton:latest KHÔNG có trên Docker Hub / NGC - đã nằm sẵn trong `docker
# images` trên Thor, do đồng nghiệp build từ voice-agent-deployment/asr-triton.
# Nó đã có torch==2.9.1, torchaudio==2.9.1, onnxruntime-gpu, libsndfile1,
# sentencepiece==0.2.0, và fix LD_LIBRARY_PATH cho Python backend stub - đúng
# hệt những gì model.py/streaming_search.py của mình cần, không thiếu không thừa.
# Dùng thẳng thay vì tự cài lại: tự cài từng đi qua apt-get/pip và bị proxy nội
# bộ công ty chặn (ports.ubuntu.com 401, developer.download.nvidia.com cert lỗi).
#
# Rủi ro cần biết: image này do team khác kiểm soát. Họ rebuild/xoá là ASR của
# mình build lại ra khác đi mà không ai báo. Nếu build trên Thor khác chưa có
# image này, phải tự dựng lại theo docker/Dockerfile.thor.fallback (TODO nếu
# cần) hoặc quay về base tritonserver:r38.4.arm64-sbsa-cu130-24.04 + tự cài.
ARG BASE_IMAGE=asr-triton:latest
FROM ${BASE_IMAGE}

# Chốt lại: mọi import phải sạch, và CUDAExecutionProvider phải THẬT SỰ có mặt -
# thiếu nó ORT âm thầm rơi về CPU, ASR vẫn chạy nhưng chậm gấp nhiều lần, không
# ai biết cho đến khi đo latency. LD_LIBRARY_PATH đã được base image set sẵn.
RUN python3 -c "\
        import numpy, torch, torchaudio, onnxruntime, sentencepiece; \
        providers = onnxruntime.get_available_providers(); \
        assert 'CUDAExecutionProvider' in providers, f'CUDA EP thiếu: {providers}'; \
        print('deps ok, providers =', providers)"
