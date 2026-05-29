# ── gpu target: pytorch base with CUDA ────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS gpu

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY prompts/ prompts/

RUN pip install --no-cache-dir -e ".[serve,tracking,pipeline]"

ENV MODEL_DEVICE=cuda
EXPOSE 8000
CMD ["analyze-llm-rationale", "serve", "--model", "gpt-oss-120b", "--variant", "variant0_neutral_baseline", "--host", "0.0.0.0", "--port", "8000"]

# ── cpu target: slim Python, CPU-only torch (default) ─────────────────────────
FROM python:3.11-slim AS cpu

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY prompts/ prompts/

# Install CPU-only torch first so pip doesn't pull CUDA wheels when resolving the package deps
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e ".[serve,tracking,pipeline]"

ENV MODEL_DEVICE=cpu
EXPOSE 8000
CMD ["analyze-llm-rationale", "serve", "--model", "gpt-oss-120b", "--variant", "variant0_neutral_baseline", "--host", "0.0.0.0", "--port", "8000"]
