# ── Stage 1: install deps ──────────────────────────────────────────────────
FROM public.ecr.aws/lambda/python:3.10 AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target /build/deps

# ── Stage 2: final Lambda image ────────────────────────────────────────────
FROM public.ecr.aws/lambda/python:3.10

WORKDIR ${LAMBDA_TASK_ROOT}

# Copy installed packages
COPY --from=builder /build/deps ${LAMBDA_TASK_ROOT}

# Copy source
COPY src/ ${LAMBDA_TASK_ROOT}/src/

# No API keys baked in — all supplied via Lambda env vars at deploy time
CMD ["src.api.main.handler"]
