#!/bin/bash
# ENHANCED PRODUCTION DEPLOYMENT SCRIPT
# ====================================
# Comprehensive deployment with configuration validation and health checks

set -e

echo "🚀 Starting Smart Ape Mode Production Deployment..."
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if .env.production exists
if [ ! -f ".env.production" ]; then
    error ".env.production file not found!"
    echo "Please copy .env.example to .env.production and configure it."
    echo ""
    echo "Quick setup commands:"
    echo "  cp .env.example .env.production"
    echo "  nano .env.production  # Edit configuration"
    echo ""
    exit 1
fi

# Load environment variables
log "Loading production environment variables..."
set -a
source .env.production
set +a

# 1. PREREQUISITE CHECKS
log "Checking prerequisites..."

# Check Docker
if ! command -v docker >/dev/null 2>&1; then
    error "Docker is required but not installed. Please install Docker and try again."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose >/dev/null 2>&1; then
    error "Docker Compose is required but not installed. Please install Docker Compose and try again."
    exit 1
fi

# Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    error "Docker daemon is not running. Please start Docker and try again."
    exit 1
fi

success "Prerequisites check passed"

# 2. CONFIGURATION VALIDATION
log "Validating production configuration..."

# Run Python configuration validator
if python3 -c "
import sys
sys.path.append('.')
from worker_ant_v1.core.config_validator import validate_production_config
exit(0 if validate_production_config() else 1)
" 2>/dev/null; then
    success "Configuration validation passed"
else
    error "Configuration validation failed!"
    echo ""
    echo "Please check your .env.production file and ensure all required variables are set."
    echo "Run this command to see detailed validation errors:"
    echo "  python3 worker_ant_v1/core/config_validator.py"
    echo ""
    exit 1
fi

# 3. SECURITY CHECKS
log "Performing security checks..."

# Check for default passwords
if grep -q "CHANGE_ME" .env.production; then
    error "Default passwords detected in .env.production!"
    echo "Please change all CHANGE_ME placeholders to secure values."
    exit 1
fi

# Check trading mode
if [ "${TRADING_MODE:-}" = "LIVE" ]; then
    warning "Trading mode is set to LIVE - real money will be used!"
    echo -n "Are you sure you want to deploy in LIVE mode? (yes/no): "
    read -r confirmation
    if [ "$confirmation" != "yes" ]; then
        echo "Deployment cancelled."
        exit 1
    fi
fi

success "Security checks passed"

# 4. DIRECTORY SETUP
log "Setting up directories..."

# Create necessary directories
mkdir -p data logs wallets monitoring

# Set proper permissions
chmod 700 wallets
chmod 755 data logs monitoring

# Create backup directory
mkdir -p data/backups

success "Directory setup completed"

# 5. BACKUP EXISTING DEPLOYMENT
if [ -d "data" ] && [ "$(ls -A data)" ]; then
    log "Creating backup of existing data..."
    backup_name="backup_$(date +%Y%m%d_%H%M%S)"
    ./backup.sh || warning "Backup creation failed, continuing..."
fi

# 6. BUILD AND DEPLOY
log "Building Docker image..."
docker-compose build --no-cache

log "Starting services..."
docker-compose up -d

# 7. WAIT FOR SERVICES
log "Waiting for services to initialize..."
sleep 30

# 8. HEALTH CHECKS
log "Performing health checks..."

# Check if services are running
if ! docker-compose ps | grep -q "Up"; then
    error "Some services failed to start!"
    echo "Service status:"
    docker-compose ps
    exit 1
fi

# Check trading bot health endpoint
for i in {1..10}; do
    if curl -f -s http://localhost:8080/health >/dev/null 2>&1; then
        break
    fi
    if [ $i -eq 10 ]; then
        error "Trading bot health check failed after 10 attempts!"
        echo "Check logs with: docker-compose logs trading-bot"
        exit 1
    fi
    log "Waiting for trading bot to be ready... (attempt $i/10)"
    sleep 10
done

# Check Prometheus
if ! curl -f -s http://localhost:9090/-/ready >/dev/null 2>&1; then
    warning "Prometheus health check failed, but continuing..."
fi

# Check Grafana
if ! curl -f -s http://localhost:3000/api/health >/dev/null 2>&1; then
    warning "Grafana health check failed, but continuing..."
fi

success "Health checks completed"

# 9. DEPLOYMENT SUMMARY
echo ""
echo "====================================="
echo "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
echo "====================================="
echo ""
echo "📊 Monitoring Dashboard:"
echo "  • Grafana: http://localhost:3000 (admin/admin)"
echo "  • Prometheus: http://localhost:9090"
echo ""
echo "🤖 Trading Bot:"
echo "  • Status: http://localhost:8080/health"
echo "  • Mode: ${TRADING_MODE:-UNKNOWN}"
echo "  • Capital: ${INITIAL_CAPITAL:-Unknown} SOL"
echo ""
echo "📋 Useful Commands:"
echo "  • View logs: docker-compose logs -f trading-bot"
echo "  • Stop system: docker-compose down"
echo "  • Restart: docker-compose restart"
echo "  • Update config: edit .env.production && docker-compose restart"
echo ""
echo "🔒 Security Reminders:"
echo "  • Monitor your alerts (Discord/Telegram/Email)"
echo "  • Check logs regularly for any issues"
echo "  • Keep your .env.production file secure"
echo "  • Never share your private keys or passwords"
echo ""

if [ "${TRADING_MODE:-}" = "LIVE" ]; then
    echo "⚠️  LIVE TRADING ACTIVE - Monitor carefully!"
else
    echo "🛡️  Running in ${TRADING_MODE:-SIMULATION} mode"
fi

echo ""
success "Smart Ape Mode is now running!"
