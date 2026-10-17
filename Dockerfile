# Built and pushed to ghcr.io/v1adych/meta-rl by .github/workflows/docker.yml on every push to main

# Minimal CUDA image: jax's pip wheels bring their own CUDA libraries, so we only
# need the GPU driver hookup this image sets up (~100 MB instead of runpod/pytorch's 10 GB)
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

# sshd for logging into the pod, curl for fetching runpodctl
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# runpodctl lets the pod stop itself after training
RUN curl -fsSL -o /usr/local/bin/runpodctl https://github.com/runpod/runpodctl/releases/download/v2.14.0/runpodctl-linux-amd64 \
    && chmod +x /usr/local/bin/runpodctl

# same uv as on the Mac
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

WORKDIR /app

# uv also installs python 3.12 itself, as pinned in pyproject.toml
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
ENV PATH=/opt/venv/bin:$PATH

COPY start.sh /start.sh
COPY *.py ./

CMD ["/start.sh"]
