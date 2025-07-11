#!/bin/bash
# SMART APE NEURAL SWARM - BACKUP SCRIPT
# ====================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Configuration
BACKUP_ROOT="/var/backups/trading-bot"
BACKUP_DIR="${BACKUP_ROOT}/$(date +%Y%m%d_%H%M%S)"
RETENTION_DAYS=30
MAX_BACKUPS=10
S3_BUCKET="s3://your-backup-bucket"
ENCRYPTION_KEY="/etc/trading-bot/backup.key"

# Ensure backup directory exists
mkdir -p "${BACKUP_DIR}"

# Backup function with error handling
backup_component() {
    local component=$1
    local source=$2
    local dest="${BACKUP_DIR}/${component}"
    
    log "Backing up ${component}..."
    
    if [ ! -e "${source}" ]; then
        warning "Source ${source} does not exist, skipping..."
        return 0
    fi
    
    mkdir -p "${dest}"
    
    case ${component} in
        "redis")
            redis-cli save
            cp /var/lib/redis/dump.rdb "${dest}/"
            ;;
        "postgres")
            PGPASSWORD=${DB_PASSWORD} pg_dump -h localhost -U ${DB_USER} ${DB_NAME} > "${dest}/database.sql"
            ;;
        "config")
            cp -r /etc/trading-bot/* "${dest}/"
            ;;
        "logs")
            find /var/log/trading-bot -type f -name "*.log*" -exec cp {} "${dest}/" \;
            ;;
        "data")
            cp -r "${source}"/* "${dest}/"
            ;;
        *)
            error "Unknown component: ${component}"
            return 1
            ;;
    esac
    
    if [ $? -ne 0 ]; then
        error "Failed to backup ${component}"
        return 1
    fi
    
    log "Successfully backed up ${component}"
    return 0
}

# Encrypt backup
encrypt_backup() {
    log "Encrypting backup..."
    tar czf - "${BACKUP_DIR}" | gpg --encrypt --recipient-file "${ENCRYPTION_KEY}" > "${BACKUP_DIR}.tar.gz.gpg"
    if [ $? -eq 0 ]; then
        log "Backup encrypted successfully"
        rm -rf "${BACKUP_DIR}"
    else
        error "Failed to encrypt backup"
        return 1
    fi
}

# Upload to S3
upload_to_s3() {
    log "Uploading to S3..."
    aws s3 cp "${BACKUP_DIR}.tar.gz.gpg" "${S3_BUCKET}/$(basename ${BACKUP_DIR}).tar.gz.gpg"
    if [ $? -eq 0 ]; then
        log "Upload successful"
        rm -f "${BACKUP_DIR}.tar.gz.gpg"
    else
        error "Failed to upload to S3"
        return 1
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    # Local cleanup
    find "${BACKUP_ROOT}" -type f -name "*.gpg" -mtime +${RETENTION_DAYS} -delete
    
    # S3 cleanup
    aws s3 ls "${S3_BUCKET}" | while read -r line; do
        createDate=$(echo $line | awk {'print $1" "$2'})
        createDate=$(date -d "$createDate" +%s)
        olderThan=$(date -d "-${RETENTION_DAYS} days" +%s)
        if [[ $createDate -lt $olderThan ]]; then
            fileName=$(echo $line | awk {'print $4'})
            if [[ $fileName != "" ]]; then
                aws s3 rm "${S3_BUCKET}/${fileName}"
            fi
        fi
    done
}

# Main backup process
main() {
    log "Starting backup process..."
    
    # Backup each component
    backup_component "redis" "/var/lib/redis" || exit 1
    backup_component "postgres" "/var/lib/postgresql" || exit 1
    backup_component "config" "/etc/trading-bot" || exit 1
    backup_component "logs" "/var/log/trading-bot" || exit 1
    backup_component "data" "/var/lib/trading-bot" || exit 1
    
    # Encrypt backup
    encrypt_backup || exit 1
    
    # Upload to S3
    upload_to_s3 || exit 1
    
    # Cleanup old backups
    cleanup_old_backups
    
    log "Backup process completed successfully"
}

# Run main function
main 