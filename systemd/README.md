# AntBot Systemd Service Files

This directory contains systemd service files for running the AntBot services on a Linux server.

## Installation

1. Copy these service files to `/etc/systemd/system/`:

```bash
sudo cp antbot.service /etc/systemd/system/
sudo cp antbot-dashboard.service /etc/systemd/system/
```

2. Reload systemd daemon:

```bash
sudo systemctl daemon-reload
```

3. Enable and start the services:

```bash
sudo systemctl enable antbot
sudo systemctl enable antbot-dashboard
sudo systemctl start antbot
sudo systemctl start antbot-dashboard
```

## Service Configuration

The services are configured to:

* Run with root permissions
* Set the working directory to `/opt/antbot`
* Use the Python virtual environment at `/opt/antbot/venv`
* Set the PYTHONPATH environment variable to resolve import issues
* Automatically restart on failure

## Checking Service Status

```bash
sudo systemctl status antbot
sudo systemctl status antbot-dashboard
```

## Viewing Logs

```bash
# View service logs
sudo journalctl -u antbot -n 100
sudo journalctl -u antbot-dashboard -n 100

# View application logs
cat /opt/antbot/logs/debug.log
cat /opt/antbot/logs/error.log
``` 