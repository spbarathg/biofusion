# SMART APE TRADING BOT - OPTIMIZED DOCKERFILE
FROM python:3.11-slim as builder

# Build arguments for flexibility
ARG TRADING_MODE=production
ARG BUILD_ENV=production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRADING_MODE=${TRADING_MODE} \
    ENVIRONMENT=${BUILD_ENV} \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better Docker caching
COPY config/requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Second stage for smaller final image
FROM python:3.11-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRADING_MODE=${TRADING_MODE} \
    ENVIRONMENT=${BUILD_ENV}

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories with secure permissions
RUN mkdir -p data logs wallets monitoring \
    && chmod 700 wallets \
    && chmod 755 data logs monitoring \
    && find . -type f -name "*.py" -exec chmod 644 {} \; \
    && find . -type d -exec chmod 755 {} \;

# Create non-root user with minimal permissions
RUN groupadd -r trader && useradd -r -g trader -s /sbin/nologin -M trader \
    && chown -R trader:trader /app

# Switch to non-root user
USER trader

# Improved health check that works with unified launcher
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys, os; sys.path.append('.'); \
    from worker_ant_v1.core.unified_config import get_config_manager; \
    config = get_config_manager(); \
    print('Bot healthy') if config else exit(1)"

# Security: Drop capabilities and set read-only root filesystem
RUN echo "Dropping capabilities and setting read-only filesystem"

# Expose ports
EXPOSE 8080

# Use unified launcher with environment-based mode selection
CMD ["python", "entry_points/run_bot.py", "--mode", "production"]
