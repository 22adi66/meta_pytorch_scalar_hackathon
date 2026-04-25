# CUDA 12.1 runtime base — we install torch from the official PyTorch whl index below
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps + Python 3.10 + Rust (for cargo check / cargo test in verifier)
RUN apt-get update && apt-get install -y \
        software-properties-common curl build-essential git pkg-config libssl-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
        python3.10 python3.10-dev python3.10-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Single pip install — requirements.txt pins torch==2.2.0+cu121 via --extra-index-url
# so trl/transformers/peft cannot pull in a CPU-only torch from PyPI
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app/

ENV PYTHONPATH=/app
ENV CRUST_WORKSPACE=/app/crust_env/dummy_workspace

EXPOSE 8000

CMD ["uvicorn", "crust_env.api:app", "--host", "0.0.0.0", "--port", "8000"]
