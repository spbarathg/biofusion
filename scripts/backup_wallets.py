#!/usr/bin/env python3
"""
Automated Wallet Backup Script for AntBot

This script provides a secure way to backup wallet data with the following features:
- Automated daily backups
- Compression to save space
- Encryption for security
- Remote backup storage options
- Backup integrity verification
- Automatic cleanup of old backups

Usage:
    python backup_wallets.py [--remote] [--verify] [--clean-older-than DAYS]
"""

import os
import sys
import time
import json
import shutil
import argparse
import logging
import tarfile
import hashlib
import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from src.utils.secrets import get_encryption_key
from src.core.paths import WALLETS_DIR, BACKUPS_DIR
from src.utils.logging.logger import setup_logging

# Optional dependencies for remote storage
try:
    import boto3
    import paramiko
    REMOTE_AVAILABLE = True
except ImportError:
    REMOTE_AVAILABLE = False

# Set up logging
logger = setup_logging("wallet_backup", "wallet_backup.log")

def create_backup(verify=False, cleanup_days=30):
    """
    Create an encrypted backup of wallet files
    
    Args:
        verify: Whether to verify the backup after creation
        cleanup_days: Delete backups older than this many days
        
    Returns:
        Path to the created backup file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"wallet_backup_{timestamp}.tar.gz.enc"
    backup_path = BACKUPS_DIR / backup_filename
    
    # Ensure backup directory exists
    BACKUPS_DIR.mkdir(exist_ok=True, parents=True)
    
    try:
        logger.info(f"Starting wallet backup to {backup_path}")
        
        # First create a tar.gz archive
        temp_archive = BACKUPS_DIR / f"temp_backup_{timestamp}.tar.gz"
        
        with tarfile.open(temp_archive, "w:gz") as tar:
            # Add all wallet files
            for wallet_file in WALLETS_DIR.glob("**/*"):
                if wallet_file.is_file():
                    arcname = wallet_file.relative_to(WALLETS_DIR)
                    tar.add(wallet_file, arcname=arcname)
            
            # Add metadata file
            metadata = {
                "timestamp": timestamp,
                "total_wallets": len(list(WALLETS_DIR.glob("*.json"))),
                "backup_version": "1.0"
            }
            
            metadata_file = BACKUPS_DIR / f"metadata_{timestamp}.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            tar.add(metadata_file, arcname="metadata.json")
            
            # Clean up temp metadata file
            metadata_file.unlink()
        
        # Now encrypt the archive
        encryption_key = get_encryption_key()
        
        # Simple encryption for demonstration (in production use stronger encryption)
        from cryptography.fernet import Fernet
        f = Fernet(encryption_key)
        
        with open(temp_archive, "rb") as input_file:
            encrypted_data = f.encrypt(input_file.read())
            
            with open(backup_path, "wb") as output_file:
                output_file.write(encrypted_data)
        
        # Calculate checksum
        checksum = hashlib.sha256()
        with open(backup_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                checksum.update(chunk)
        
        checksum_file = backup_path.with_suffix('.sha256')
        with open(checksum_file, "w") as f:
            f.write(checksum.hexdigest())
        
        # Clean up temp file
        temp_archive.unlink()
        
        logger.info(f"Backup completed successfully: {backup_path}")
        logger.info(f"Checksum saved to: {checksum_file}")
        
        # Verify the backup if requested
        if verify:
            if verify_backup(backup_path):
                logger.info("Backup verification successful")
            else:
                logger.error("Backup verification failed!")
                return None
        
        # Cleanup old backups
        if cleanup_days > 0:
            cleanup_old_backups(cleanup_days)
        
        return backup_path
        
    except Exception as e:
        logger.error(f"Backup failed: {str(e)}")
        # Clean up any partial backups
        if backup_path.exists():
            backup_path.unlink()
        return None

def verify_backup(backup_path):
    """
    Verify that a backup is valid and can be restored
    
    Args:
        backup_path: Path to the backup file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Verify checksum
        checksum_file = backup_path.with_suffix('.sha256')
        if not checksum_file.exists():
            logger.error("Checksum file not found")
            return False
        
        with open(checksum_file, "r") as f:
            expected_checksum = f.read().strip()
        
        actual_checksum = hashlib.sha256()
        with open(backup_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                actual_checksum.update(chunk)
        
        if actual_checksum.hexdigest() != expected_checksum:
            logger.error("Checksum verification failed")
            return False
        
        # Try to decrypt a small portion
        encryption_key = get_encryption_key()
        from cryptography.fernet import Fernet
        f = Fernet(encryption_key)
        
        with open(backup_path, "rb") as backup_file:
            # Just try to decrypt the first chunk to verify the key works
            encrypted_data = backup_file.read(1024)
            try:
                f.decrypt(encrypted_data)
                logger.info("Decryption test successful")
                return True
            except Exception as e:
                logger.error(f"Decryption test failed: {str(e)}")
                return False
    
    except Exception as e:
        logger.error(f"Backup verification failed: {str(e)}")
        return False

def upload_to_remote(backup_path, method="s3"):
    """
    Upload backup to a remote storage location
    
    Args:
        backup_path: Path to the backup file
        method: Remote storage method (s3, sftp)
        
    Returns:
        True if successful, False otherwise
    """
    if not REMOTE_AVAILABLE:
        logger.error("Remote storage packages not installed. Install boto3 and paramiko.")
        return False
    
    try:
        if method == "s3":
            # Upload to S3
            bucket_name = os.getenv("BACKUP_S3_BUCKET")
            if not bucket_name:
                logger.error("BACKUP_S3_BUCKET environment variable not set")
                return False
            
            s3_client = boto3.client('s3')
            s3_key = f"antbot/backups/{backup_path.name}"
            
            s3_client.upload_file(str(backup_path), bucket_name, s3_key)
            logger.info(f"Backup uploaded to S3: {bucket_name}/{s3_key}")
            
            # Also upload checksum file
            checksum_file = backup_path.with_suffix('.sha256')
            s3_checksum_key = f"antbot/backups/{checksum_file.name}"
            s3_client.upload_file(str(checksum_file), bucket_name, s3_checksum_key)
            
            return True
            
        elif method == "sftp":
            # Upload via SFTP
            sftp_host = os.getenv("BACKUP_SFTP_HOST")
            sftp_user = os.getenv("BACKUP_SFTP_USER")
            sftp_key = os.getenv("BACKUP_SFTP_KEY")
            sftp_dir = os.getenv("BACKUP_SFTP_DIR", "/backups")
            
            if not all([sftp_host, sftp_user, sftp_key]):
                logger.error("SFTP environment variables not set")
                return False
            
            key = paramiko.RSAKey.from_private_key_file(sftp_key)
            transport = paramiko.Transport((sftp_host, 22))
            transport.connect(username=sftp_user, pkey=key)
            sftp = paramiko.SFTPClient.from_transport(transport)
            
            remote_path = f"{sftp_dir}/{backup_path.name}"
            sftp.put(str(backup_path), remote_path)
            
            # Also upload checksum file
            checksum_file = backup_path.with_suffix('.sha256')
            remote_checksum_path = f"{sftp_dir}/{checksum_file.name}"
            sftp.put(str(checksum_file), remote_checksum_path)
            
            sftp.close()
            transport.close()
            
            logger.info(f"Backup uploaded via SFTP to {sftp_host}:{remote_path}")
            return True
            
        else:
            logger.error(f"Unsupported remote storage method: {method}")
            return False
            
    except Exception as e:
        logger.error(f"Remote upload failed: {str(e)}")
        return False

def cleanup_old_backups(days):
    """
    Delete backups older than the specified number of days
    
    Args:
        days: Delete backups older than this many days
    """
    try:
        cutoff_time = time.time() - (days * 86400)  # 86400 seconds in a day
        
        for backup_file in BACKUPS_DIR.glob("wallet_backup_*.tar.gz.enc"):
            file_time = backup_file.stat().st_mtime
            if file_time < cutoff_time:
                # Delete the backup file
                backup_file.unlink()
                
                # Also delete the checksum file if it exists
                checksum_file = backup_file.with_suffix('.sha256')
                if checksum_file.exists():
                    checksum_file.unlink()
                
                logger.info(f"Deleted old backup: {backup_file.name}")
        
        logger.info(f"Cleanup complete. Removed backups older than {days} days.")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="AntBot Wallet Backup Tool")
    parser.add_argument("--remote", choices=["s3", "sftp"], help="Upload to remote storage")
    parser.add_argument("--verify", action="store_true", help="Verify backup after creation")
    parser.add_argument("--clean-older-than", type=int, default=30,
                        help="Delete backups older than this many days (default: 30)")
    
    args = parser.parse_args()
    
    # Create the backup
    backup_path = create_backup(verify=args.verify, cleanup_days=args.clean_older_than)
    
    if backup_path and args.remote:
        upload_to_remote(backup_path, method=args.remote)

if __name__ == "__main__":
    main() 