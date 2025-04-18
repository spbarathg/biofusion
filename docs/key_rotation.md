# AntBot Key Rotation

This document describes the key rotation setup for AntBot's encryption system.

## Overview

AntBot uses encryption keys to protect sensitive wallet information. To maintain security, we automatically rotate these keys every 90 days using a systemd timer.

## Components

The key rotation system consists of:

1. **Key Rotation Script** (`scripts/rotate_keys.py`): Python script that performs the key rotation process
2. **Systemd Service** (`systemd/antbot-key-rotation.service`): Defines how to run the key rotation script
3. **Systemd Timer** (`systemd/antbot-key-rotation.timer`): Schedules when to run the key rotation

## Installation

To install the key rotation service, run the installation script as root:

```bash
sudo ./scripts/install_systemd_services.sh
```

This will:
- Copy the service and timer files to `/etc/systemd/system/`
- Enable and start the timer
- Verify the timer is active

## Schedule

The key rotation runs:
- Every 90 days
- At midnight (with a randomized delay of up to 1 hour)
- Persistent (missed rotations will run on system boot)

## Testing

To test the key rotation process without affecting production data:

```bash
./scripts/test_key_rotation.sh
```

This creates a test environment with sample data to verify the key rotation works correctly.

## Manual Rotation

To manually trigger a key rotation:

```bash
sudo systemctl start antbot-key-rotation.service
```

## Verification

To verify the timer is set up correctly:

```bash
systemctl status antbot-key-rotation.timer
systemctl list-timers | grep antbot
```

## Logs

Key rotation logs can be viewed with:

```bash
journalctl -u antbot-key-rotation.service
```

## Troubleshooting

### Failed Rotation

If a key rotation fails:
1. Check the logs: `journalctl -u antbot-key-rotation.service`
2. The backup is stored in `/opt/antbot/backups/key_rotation_[timestamp]`
3. Restore from backup if needed

### Timer Not Running

If the timer is not running:
1. Check status: `systemctl status antbot-key-rotation.timer`
2. Try restarting: `sudo systemctl restart antbot-key-rotation.timer`
3. Check for errors: `journalctl -u antbot-key-rotation.timer`

## Security Considerations

- The key rotation script requires root access to read and write encrypted files
- All previous data is backed up before rotation
- The verification step ensures all files can be decrypted with the new key
- Automated monitoring alerts are triggered if key rotation fails 