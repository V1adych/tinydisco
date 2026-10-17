FROM nvidia/cuda:12.8.1-base-ubuntu24.04

# sshd for logging into the pod, git for cloning the code, curl for fetching runpodctl
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# runpodctl lets the pod stop itself after training
RUN curl -fsSL -o /usr/local/bin/runpodctl https://github.com/runpod/runpodctl/releases/download/v2.14.0/runpodctl-linux-amd64 \
    && chmod +x /usr/local/bin/runpodctl

# same uv as on the Mac
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

# The image is only the environment; start.sh clones the code when the pod starts.
# uv also installs python 3.12 itself, as pinned in pyproject.toml
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock /tmp/env/
RUN cd /tmp/env && uv sync --frozen
ENV PATH=/opt/venv/bin:$PATH

COPY start.sh /start.sh

CMD ["/start.sh"]
