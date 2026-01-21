FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

# Set working directory
WORKDIR /clinibench

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml ./
COPY src src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync
