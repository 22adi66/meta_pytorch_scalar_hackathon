FROM python:3.10-slim

# Install system dependencies and Rust toolchain
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code into module directory to resolve relative imports
COPY . /app/crust_env

# Set PYTHONPATH so absolute and relative imports work correctly
ENV PYTHONPATH=/app

# Expose port for FastAPI
EXPOSE 8000

# Set workspace environment variable
ENV CRUST_WORKSPACE=/app/crust_env/dummy_workspace

# Run uvicorn server
CMD ["uvicorn", "crust_env.api:app", "--host", "0.0.0.0", "--port", "8000"]
