# AdAnvil MCP server over stdio.
#   docker build -t adanvil .
#   docker run -i --rm -v "$PWD:/projects" adanvil
# Optional Python extras: docker build --build-arg EXTRAS=transcribe -t adanvil:transcribe .
# Ubuntu 24.04 ships ffmpeg 6.1 (with libzimg).
FROM ubuntu:24.04

ARG EXTRAS=""
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

COPY pyproject.toml README.md LICENSE /src/
COPY src /src/src
RUN uv venv /opt/adanvil --python /usr/bin/python3 \
    && if [ -n "$EXTRAS" ]; then SPEC="/src[$EXTRAS]"; else SPEC="/src"; fi \
    && uv pip install --python /opt/adanvil/bin/python "$SPEC" \
    && rm -rf /src /root/.cache

# Non-root user with an id that is unlikely to clash; override with --user on Linux hosts
# so files written to the mounted folder belong to you.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin adanvil \
    && mkdir /projects && chown adanvil /projects
ENV PATH="/opt/adanvil/bin:$PATH" \
    HOME=/tmp \
    PYTHONUNBUFFERED=1
USER adanvil
WORKDIR /projects
VOLUME /projects
ENTRYPOINT ["adanvil-mcp"]
