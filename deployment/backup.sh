#!/bin/bash
# SMART APE TRADING BOT - BACKUP SCRIPT
# ====================================
# Comprehensive backup of all critical data and configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Create timestamped backup directory
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

log "Starting comprehensive backup..."

# Backup data directories
for dir in data wallets logs monitoring; do
    if [ -d "$dir" ]; then
        log "Backing up $dir/"
        cp -r "$dir/" "$BACKUP_DIR/" || warning "Could not backup $dir/"
    else
        warning "$dir/ not found"
    fi
done

# Backup configuration files
log "Backing up configuration files..."
for config in .env.production config/env.template config/requirements.txt; do
    if [ -f "$config" ]; then
        cp "$config" "$BACKUP_DIR/" || warning "Could not backup $config"
    else
        warning "$config not found"
    fi
done

# Backup Docker configuration
log "Backing up Docker configuration..."
for docker_file in deployment/docker-compose.yml deployment/docker-compose.dev.yml deployment/Dockerfile; do
    if [ -f "$docker_file" ]; then
        mkdir -p "$BACKUP_DIR/deployment"
        cp "$docker_file" "$BACKUP_DIR/deployment/" || warning "Could not backup $docker_file"
    else
        warning "$docker_file not found"
    fi
done

# Create archive with compression
log "Creating compressed archive..."
tar -czf "${BACKUP_DIR}.tar.gz" "$BACKUP_DIR" || {
    error "Failed to create archive"
    exit 1
}

# Cleanup temporary directory
rm -rf "$BACKUP_DIR"

# Verify backup
if [ -f "${BACKUP_DIR}.tar.gz" ]; then
    size=$(du -h "${BACKUP_DIR}.tar.gz" | cut -f1)
    log "✅ Backup successful: ${BACKUP_DIR}.tar.gz (${size})"
    
    # Keep only last 5 backups
    cd backups
    ls -t | tail -n +6 | xargs -I {} rm -f {} 2>/dev/null || true
    cd ..
    
    log "Cleaned old backups (keeping last 5)"
else
    error "Backup failed: Archive not found"
    exit 1
fi
