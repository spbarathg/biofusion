#!/bin/bash
# Script to install AntBot systemd services
# This script must be run as root

set -e

# Configuration
INSTALL_DIR="/opt/antbot"
SYSTEMD_DIR="/etc/systemd/system"
SERVICE_FILES=("antbot-key-rotation.service" "antbot-key-rotation.timer")

# Ensure we're running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root. Please use sudo."
    exit 1
fi

# Check if source directory exists
if [ ! -d "systemd" ]; then
    echo "Error: systemd directory not found. Please run this script from the project root."
    exit 1
fi

# Copy service files to systemd directory
echo "Installing systemd service files..."
for file in "${SERVICE_FILES[@]}"; do
    cp "systemd/$file" "$SYSTEMD_DIR/$file"
    echo " - Installed $file"
done

# Reload systemd to recognize new services
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable and start the timer
echo "Enabling and starting antbot-key-rotation.timer..."
systemctl enable antbot-key-rotation.timer
systemctl start antbot-key-rotation.timer

# Verify timer status
echo "Checking timer status..."
systemctl status antbot-key-rotation.timer

echo "Installation complete. The key rotation will run every 90 days."
echo "To test the service immediately, run: sudo systemctl start antbot-key-rotation.service"
echo "To check the timer schedule, run: systemctl list-timers | grep antbot" 