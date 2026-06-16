# ── gpu target: pytorch base with CUDA ────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS gpu

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY prompts/ prompts/
COPY static/ static/
COPY analysis/crypto_5m_equity_payload.json analysis/crypto_5m_equity_payload.json

RUN pip install --no-cache-dir -e ".[serve,tracking,pipeline,trading]"

# The RAG embedding model is NOT baked into the image — it's mounted at runtime
# from a GCS volume at HF_HOME (Cloud Run --add-volume). This keeps the image
# small (fast deploys), and the build never touches HuggingFace (no 429).
# Offline mode forces a local load from the mounted cache.
ENV HF_HOME=/app/.hf_cache

ENV MODEL_DEVICE=cuda
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["analyze-llm-rationale", "serve", "--model", "gpt-oss-120b", "--variant", "variant0_neutral_baseline", "--host", "0.0.0.0", "--port", "8000"]

# ── cpu target: slim Python, CPU-only torch (default) ─────────────────────────
FROM python:3.11-slim AS cpu

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY prompts/ prompts/
COPY static/ static/
COPY analysis/crypto_5m_equity_payload.json analysis/crypto_5m_equity_payload.json

# Install CPU-only torch first so pip doesn't pull CUDA wheels when resolving the package deps
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e ".[serve,tracking,pipeline,trading]"

# The RAG embedding model is NOT baked into the image — it's mounted at runtime
# from a GCS volume at HF_HOME (Cloud Run --add-volume). This keeps the image
# small (fast deploys), and the build never touches HuggingFace (no 429).
# Offline mode forces a local load from the mounted cache.
ENV HF_HOME=/app/.hf_cache

ENV MODEL_DEVICE=cpu
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["analyze-llm-rationale", "serve", "--model", "gpt-oss-120b", "--variant", "variant0_neutral_baseline", "--host", "0.0.0.0", "--port", "8000"]
