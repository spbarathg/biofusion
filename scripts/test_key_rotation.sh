#!/bin/bash
# Script to test the key rotation process
# This script simulates a key rotation using a test environment

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_DIR="/tmp/antbot_key_rotation_test"
VENV_DIR="$PROJECT_ROOT/venv"
PYTHON="$VENV_DIR/bin/python"

# Ensure we have a virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Python virtual environment not found at $VENV_DIR"
    echo "Please create a virtual environment before running this test."
    exit 1
fi

# Set up test environment
echo "Setting up test environment at $TEST_DIR..."
mkdir -p "$TEST_DIR/wallets"
mkdir -p "$TEST_DIR/data"
mkdir -p "$TEST_DIR/backups"

# Create a test wallet file
echo "Creating test wallet file..."
cat > "$TEST_DIR/generate_test_wallet.py" <<EOL
import os
import json
import base64
import sys
from pathlib import Path
from cryptography.fernet import Fernet

# Create test directory structure
test_dir = Path("$TEST_DIR")
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
EOL

# Run the test wallet generator
$PYTHON "$TEST_DIR/generate_test_wallet.py"

# Set environment variables for the rotation script
export DATA_DIR="$TEST_DIR/data"
export WALLETS_DIR="$TEST_DIR/wallets"
export BACKUPS_DIR="$TEST_DIR/backups"

# Create a verification script
echo "Creating verification script..."
cat > "$TEST_DIR/verify_rotation.py" <<EOL
import os
import json
import sys
import glob
from pathlib import Path
from cryptography.fernet import Fernet

# Get environment variables
data_dir = Path(os.environ.get("DATA_DIR", "$TEST_DIR/data"))
wallets_dir = Path(os.environ.get("WALLETS_DIR", "$TEST_DIR/wallets"))
backups_dir = Path(os.environ.get("BACKUPS_DIR", "$TEST_DIR/backups"))

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
EOL

# Run the key rotation script in dry-run mode first
echo "Testing key rotation in dry-run mode..."
$PYTHON "$PROJECT_ROOT/scripts/rotate_keys.py" --force --dry-run

# Run the verification script to confirm the original key still works
echo "Verifying original key still works after dry-run..."
$PYTHON "$TEST_DIR/verify_rotation.py"

# Now run the real key rotation
echo "Running actual key rotation..."
$PYTHON "$PROJECT_ROOT/scripts/rotate_keys.py" --force --verification

# Verify that the rotation worked
echo "Verifying key rotation was successful..."
$PYTHON "$TEST_DIR/verify_rotation.py"

echo "Key rotation test completed successfully!"
echo "Test environment is at $TEST_DIR if you want to inspect it."
echo "To clean up the test environment, run: rm -rf $TEST_DIR" 