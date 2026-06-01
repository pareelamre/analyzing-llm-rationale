# ── gpu target: pytorch base with CUDA ────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS gpu

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY prompts/ prompts/
COPY static/ static/

RUN pip install --no-cache-dir -e ".[serve,tracking,pipeline]"

# Bake the RAG embedding model into the image so it loads locally on every
# instance (no flaky runtime download from HuggingFace).
ENV HF_HOME=/app/.hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

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

# Install CPU-only torch first so pip doesn't pull CUDA wheels when resolving the package deps
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e ".[serve,tracking,pipeline]"

# Bake the RAG embedding model into the image so it loads locally on every
# instance (no flaky runtime download from HuggingFace).
ENV HF_HOME=/app/.hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENV MODEL_DEVICE=cpu
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["analyze-llm-rationale", "serve", "--model", "gpt-oss-120b", "--variant", "variant0_neutral_baseline", "--host", "0.0.0.0", "--port", "8000"]
