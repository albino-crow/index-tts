# Use NVIDIA CUDA base image for GPU support
# Using CUDA 12.8 to match system CUDA version and PyTorch requirements
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04


# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set UV environment variables for better performance and timeout handling
ENV UV_HTTP_TIMEOUT=300
ENV UV_CONCURRENT_DOWNLOADS=4
ENV UV_CACHE_DIR=/root/.cache/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY . .

# Install dependencies with uv (with increased timeout and retry logic)
RUN uv sync --all-extras

# Install huggingface-hub CLI tool
RUN uv tool install "huggingface-hub[cli,hf_xet]"
ENV PATH="/root/.local/share/uv/tools/huggingface-hub/bin:$PATH"

# Download model checkpoints
RUN uv run hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints

# Expose port for FastAPI
EXPOSE 8000

# Run the application with uvicorn
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
