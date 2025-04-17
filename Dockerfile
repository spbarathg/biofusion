FROM python:3.9-slim as python-base

# Install Rust and dependencies
FROM python-base as builder
RUN apt-get update && apt-get install -y \
    curl build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy source
WORKDIR /app
COPY requirements.txt ./
COPY rust_core rust_core/
COPY setup.py ./

# Build Rust core
WORKDIR /app/rust_core
RUN cargo build --release

# Final stage
FROM python-base as runtime

WORKDIR /app

# Copy Python requirements
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Rust binary - use the actual compiled library name
COPY --from=builder /app/rust_core/target/release/libant_bot_core.so /usr/local/lib/
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY config/ config/
COPY .env .env

# Create necessary directories
RUN mkdir -p data/wallets data/backups logs config/secrets
RUN chmod -R 777 logs

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_DIR=/app/logs
ENV ENCRYPTION_KEY_PATH=/app/config/secrets/.encryption_key

# Run as non-root user
RUN groupadd -r antbot && useradd -r -g antbot antbot
RUN chown -R antbot:antbot /app
USER antbot

# Default command runs the queen agent
CMD ["python", "-m", "src.models.queen"]

# Alternative commands:
# Dashboard: python -m src.dashboard.run_dashboard
# Worker: python -m src.models.worker
# CLI: python scripts/wallet_cli.py 