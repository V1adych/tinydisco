# Build and push from the Mac (RunPod hosts are x86_64):
#   docker buildx build --platform linux/amd64 -t <user>/meta-rl:v1 --push .

# RunPod caches this image on its hosts, and its startup script sets up SSH
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# same uv as on the Mac, since the base image's uv may be too old to read uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

# the base image's startup script runs /post_start.sh once SSH is up
COPY post_start.sh /post_start.sh

WORKDIR /app

# own venv, so jax's CUDA libraries don't clash with torch's in the system python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
ENV PATH=/opt/venv/bin:$PATH

COPY *.py ./
