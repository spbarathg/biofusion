# PowerShell script to test the key rotation process on Windows
# This script simulates a key rotation using a test environment

# Exit on error
$ErrorActionPreference = "Stop"

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$TestDir = "$env:TEMP\antbot_key_rotation_test"
$VenvDir = "$ProjectRoot\venv"
$Python = "python"

# Create directory function
function Ensure-Dir {
    param (
        [string]$Path
    )
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Host "Created directory: $Path"
    }
}

# Ensure we have a virtual environment
if (-not (Test-Path $VenvDir)) {
    Write-Host "Error: Python virtual environment not found at $VenvDir" -ForegroundColor Red
    Write-Host "Please create a virtual environment before running this test." -ForegroundColor Red
    exit 1
}

# Set up test environment
Write-Host "Setting up test environment at $TestDir..." -ForegroundColor Cyan
Ensure-Dir "$TestDir"
Ensure-Dir "$TestDir\wallets"
Ensure-Dir "$TestDir\data"
Ensure-Dir "$TestDir\backups"

# Create a test wallet file
Write-Host "Creating test wallet file..." -ForegroundColor Cyan
$TestWalletScript = @"
import os
import json
import base64
import sys
from pathlib import Path
from cryptography.fernet import Fernet

# Create test directory structure
test_dir = Path(r"$TestDir")
wallets_dir = test_dir / "wallets"
data_dir = test_dir / "data"

# Generate a test encryption key
key = Fernet.generate_key()
with open(data_dir / ".encryption_key", "wb") as f:
    f.write(key)

# Create a sample wallet
wallet_data = {
    "name": "test_wallet",
    "public_key": "test_public_key",
    "private_key": "test_private_key_encrypted",
    "created_at": "2023-01-01T00:00:00Z",
    "test_value": "this_is_a_test"
}

# Encrypt the wallet data
f = Fernet(key)
wallet_json = json.dumps(wallet_data).encode()
encrypted_data = f.encrypt(wallet_json)

# Save the encrypted wallet
with open(wallets_dir / "test_wallet.json", "wb") as f:
    f.write(encrypted_data)

print("Test wallet created successfully")
print(f"Original key: {base64.b64encode(key).decode()}")
"@

$TestWalletScript | Out-File -FilePath "$TestDir\generate_test_wallet.py" -Encoding utf8

# Run the test wallet generator
& $Python "$TestDir\generate_test_wallet.py"

# Set environment variables for the rotation script
$env:DATA_DIR = "$TestDir\data"
$env:WALLETS_DIR = "$TestDir\wallets"
$env:BACKUPS_DIR = "$TestDir\backups"

# Create a verification script
Write-Host "Creating verification script..." -ForegroundColor Cyan
$VerificationScript = @"
import os
import json
import sys
import glob
from pathlib import Path
from cryptography.fernet import Fernet

# Get environment variables
data_dir = Path(os.environ.get("DATA_DIR", r"$TestDir\data"))
wallets_dir = Path(os.environ.get("WALLETS_DIR", r"$TestDir\wallets"))
backups_dir = Path(os.environ.get("BACKUPS_DIR", r"$TestDir\backups"))

# Get the current key
with open(data_dir / ".encryption_key", "rb") as f:
    current_key = f.read()

print(f"Current key: {current_key}")

# Try to decrypt the wallet
try:
    wallet_path = next(wallets_dir.glob("*.json"))
    with open(wallet_path, "rb") as f:
        encrypted_data = f.read()
    
    f = Fernet(current_key)
    decrypted_data = f.decrypt(encrypted_data)
    wallet_data = json.loads(decrypted_data.decode())
    
    print("Wallet decryption successful!")
    print(f"Wallet data: {wallet_data}")
    
    if wallet_data.get("test_value") == "this_is_a_test":
        print("Verification PASSED: Test value matches expected value")
        sys.exit(0)
    else:
        print("Verification FAILED: Test value does not match expected value")
        sys.exit(1)
except Exception as e:
    print(f"Verification FAILED: {str(e)}")
    sys.exit(1)
"@

$VerificationScript | Out-File -FilePath "$TestDir\verify_rotation.py" -Encoding utf8

# Run the key rotation script in dry-run mode first
Write-Host "Testing key rotation in dry-run mode..." -ForegroundColor Cyan
& $Python "$ProjectRoot\scripts\rotate_keys.py" --force --dry-run

# Run the verification script to confirm the original key still works
Write-Host "Verifying original key still works after dry-run..." -ForegroundColor Cyan
& $Python "$TestDir\verify_rotation.py"

# Now run the real key rotation
Write-Host "Running actual key rotation..." -ForegroundColor Green
& $Python "$ProjectRoot\scripts\rotate_keys.py" --force --verification

# Verify that the rotation worked
Write-Host "Verifying key rotation was successful..." -ForegroundColor Cyan
& $Python "$TestDir\verify_rotation.py"

Write-Host "Key rotation test completed successfully!" -ForegroundColor Green
Write-Host "Test environment is at $TestDir if you want to inspect it." -ForegroundColor Yellow
Write-Host "To clean up the test environment, run: Remove-Item -Recurse -Force $TestDir" -ForegroundColor Yellow 