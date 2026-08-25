
ARG BASE_IMAGE=asr-triton:latest
FROM ${BASE_IMAGE}


RUN python3 -c "\
        import numpy, torch, torchaudio, onnxruntime, sentencepiece; \
        providers = onnxruntime.get_available_providers(); \
        assert 'CUDAExecutionProvider' in providers, f'CUDA EP thiếu: {providers}'; \
        print('deps ok, providers =', providers)"
