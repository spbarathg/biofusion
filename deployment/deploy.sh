#!/bin/bash

# SMART APE TRADING BOT - DEPLOYMENT SCRIPT
# ========================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
error() {
    echo -e "${RED}❌ ERROR: $1${NC}" >&2
    exit 1
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  WARNING: $1${NC}"
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root"
fi

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        error "Python 3.8+ is required but not installed"
    fi
    
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
        error "Python 3.8+ is required, found $python_version"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is required but not installed"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is required but not installed"
    fi
    
    success "Prerequisites check passed"
}

# Validate configuration
validate_configuration() {
    info "Validating configuration..."
    
    # Check if .env.production exists
    if [[ ! -f ".env.production" ]]; then
        error "Configuration file .env.production not found. Please copy config/env.template to .env.production and configure it."
    fi
    
    # Check for required API keys
    required_keys=(
        "HELIUS_API_KEY"
        "SOLANA_TRACKER_API_KEY"
        "JUPITER_API_KEY"
        "RAYDIUM_API_KEY"
    )
    
    missing_keys=()
    for key in "${required_keys[@]}"; do
        if ! grep -q "^${key}=" .env.production || grep -q "^${key}=your_" .env.production || grep -q "^${key}=REPLACE_WITH" .env.production; then
            missing_keys+=("$key")
        fi
    done
    
    if [[ ${#missing_keys[@]} -gt 0 ]]; then
        error "Missing or invalid API keys: ${missing_keys[*]}"
    fi
    
    # Check trading parameters
    trading_params=(
        "TRADING_MODE"
        "INITIAL_CAPITAL"
        "MAX_TRADE_SIZE_SOL"
        "MIN_TRADE_SIZE_SOL"
        "MAX_SLIPPAGE_PERCENT"
        "PROFIT_TARGET_PERCENT"
        "STOP_LOSS_PERCENT"
    )
    
    missing_params=()
    for param in "${trading_params[@]}"; do
        if ! grep -q "^${param}=" .env.production; then
            missing_params+=("$param")
        fi
    done
    
    if [[ ${#missing_params[@]} -gt 0 ]]; then
        error "Missing trading parameters: ${missing_params[*]}"
    fi
    
    success "Configuration validation passed"
}

# Security checks
security_checks() {
    info "Performing security checks..."
    
    # Check for default passwords
    if grep -q "admin123" .env.production; then
        error "Default password detected in .env.production!"
    fi
    
    if grep -q "password123" .env.production; then
        error "Default password detected in .env.production!"
    fi
    
    # Check file permissions
    if [[ $(stat -c %a .env.production) != "600" ]]; then
        warning "Setting secure permissions on .env.production"
        chmod 600 .env.production
    fi
    
    # Check for sensitive data in logs
    if [[ -d "logs" ]]; then
        if grep -r "private_key\|secret\|password" logs/ 2>/dev/null; then
            warning "Sensitive data found in logs directory"
        fi
    fi
    
    success "Security checks passed"
}

# Install dependencies
install_dependencies() {
    info "Installing Python dependencies..."
    
    if [[ ! -f "config/requirements.txt" ]]; then
        error "requirements.txt not found"
    fi
    
    pip3 install -r config/requirements.txt
    
    success "Dependencies installed successfully"
}

# Build Docker images
build_docker() {
    info "Building Docker images..."
    
    # Build main application
    docker build -t smart-ape-bot:latest .
    
    success "Docker images built successfully"
}

# Start services
start_services() {
    info "Starting services..."
    
    # Start with docker-compose
    docker-compose -f deployment/docker-compose.yml up -d
    
    success "Services started successfully"
}

# Health check
health_check() {
    info "Performing health check..."
    
    # Wait for services to start
    sleep 10
    
    # Check if containers are running
    if ! docker-compose -f deployment/docker-compose.yml ps | grep -q "Up"; then
        error "Services failed to start properly"
    fi
    
    # Check application health
    if ! curl -f http://localhost:8080/health 2>/dev/null; then
        warning "Application health check failed, but continuing..."
    fi
    
    success "Health check completed"
}

# Main deployment function
main() {
    echo "🚀 SMART APE TRADING BOT - DEPLOYMENT"
    echo "====================================="
    
    check_prerequisites
    validate_configuration
    security_checks
    install_dependencies
    build_docker
    start_services
    health_check
    
    echo ""
    echo "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo "====================================="
    echo ""
    echo "📊 Monitoring Dashboard: http://localhost:3000"
    echo "📈 Prometheus Metrics: http://localhost:9090"
    echo "🔧 Application Status: http://localhost:8080"
    echo ""
    echo "📋 Next Steps:"
    echo "  • Monitor the application logs: docker-compose logs -f"
    echo "  • Check trading performance in the dashboard"
    echo "  • Review system metrics in Prometheus"
    echo "  • Set up alerts for critical events"
    echo ""
    echo "🔒 Security Reminders:"
    echo "  • Keep your API keys secure"
    echo "  • Regularly update dependencies"
    echo "  • Monitor for unusual activity"
    echo "  • Never share your private keys or passwords"
    echo ""
}

# Run main function
main "$@"
