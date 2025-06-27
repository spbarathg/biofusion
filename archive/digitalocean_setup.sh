#!/bin/bash
# ============================================================================
# Enhanced Crypto Trading Bot - DigitalOcean Production Setup Script
# ============================================================================
# Run this script on a fresh Ubuntu 22.04 LTS droplet

set -e  # Exit on any error

echo "🚀 Starting Enhanced Crypto Trading Bot Production Setup..."

# ============================================================================
# 1. SYSTEM UPDATES & ESSENTIAL PACKAGES
# ============================================================================
echo "📦 Updating system packages..."
apt update && apt upgrade -y

echo "📦 Installing essential packages..."
apt install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    htop \
    wget \
    curl \
    unzip \
    build-essential \
    software-properties-common \
    nginx \
    supervisor \
    ufw \
    fail2ban \
    sqlite3 \
    cron \
    logrotate

# ============================================================================
# 2. PYTHON ENVIRONMENT SETUP
# ============================================================================
echo "🐍 Setting up Python environment..."

# Create dedicated user for the bot
useradd -m -s /bin/bash tradebot || true
usermod -aG sudo tradebot

# Switch to tradebot user for application setup
su - tradebot << 'EOF'
cd /home/tradebot

# Clone the repository (replace with your repo)
git clone https://github.com/your-username/antbotNew.git
cd antbotNew

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install additional production dependencies
pip install \
    gunicorn \
    supervisor \
    psutil \
    prometheus-client \
    sentry-sdk

# Set up environment variables
cp worker_ant_v1/.env.example worker_ant_v1/.env
echo "⚠️  EDIT /home/tradebot/antbotNew/worker_ant_v1/.env WITH YOUR ACTUAL VALUES"
EOF

# ============================================================================
# 3. SYSTEM OPTIMIZATION FOR HIGH-FREQUENCY TRADING
# ============================================================================
echo "⚡ Optimizing system for high-frequency trading..."

# Increase file descriptor limits
cat >> /etc/security/limits.conf << 'EOF'
tradebot soft nofile 65536
tradebot hard nofile 65536
* soft nofile 65536
* hard nofile 65536
EOF

# Optimize network settings for low latency
cat >> /etc/sysctl.conf << 'EOF'
# Network optimizations for trading
net.core.rmem_max = 268435456
net.core.wmem_max = 268435456
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_rmem = 4096 87380 268435456
net.ipv4.tcp_wmem = 4096 65536 268435456
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
EOF

sysctl -p

# ============================================================================
# 4. FIREWALL CONFIGURATION
# ============================================================================
echo "🔥 Configuring firewall..."

# Enable UFW
ufw --force enable

# Allow SSH (change 22 to your custom port if using one)
ufw allow 22/tcp

# Allow HTTP/HTTPS for monitoring dashboards
ufw allow 80/tcp
ufw allow 443/tcp

# Allow outbound connections for trading
ufw allow out 443/tcp
ufw allow out 80/tcp

# Deny all other incoming by default
ufw default deny incoming
ufw default allow outgoing

# ============================================================================
# 5. SUPERVISOR CONFIGURATION FOR PROCESS MANAGEMENT
# ============================================================================
echo "👑 Setting up Supervisor for process management..."

cat > /etc/supervisor/conf.d/trading-bot.conf << 'EOF'
[program:enhanced-trading-bot]
command=/home/tradebot/antbotNew/venv/bin/python -m worker_ant_v1.run_enhanced_bot --mode live
directory=/home/tradebot/antbotNew
user=tradebot
group=tradebot
autostart=true
autorestart=true
startretries=3
stderr_logfile=/var/log/supervisor/trading-bot-error.log
stdout_logfile=/var/log/supervisor/trading-bot.log
environment=PATH="/home/tradebot/antbotNew/venv/bin"

[program:trading-bot-swarm]
command=/home/tradebot/antbotNew/venv/bin/python -m worker_ant_v1.start_swarm --mode genesis
directory=/home/tradebot/antbotNew
user=tradebot
group=tradebot
autostart=false
autorestart=true
startretries=3
stderr_logfile=/var/log/supervisor/swarm-error.log
stdout_logfile=/var/log/supervisor/swarm.log
environment=PATH="/home/tradebot/antbotNew/venv/bin"

[program:kill-switch-monitor]
command=/home/tradebot/antbotNew/venv/bin/python -c "
from worker_ant_v1.swarm_kill_switch import kill_switch_monitor
import asyncio
asyncio.run(kill_switch_monitor.run_continuous_monitoring())
"
directory=/home/tradebot/antbotNew
user=tradebot
group=tradebot
autostart=true
autorestart=true
startretries=999
stderr_logfile=/var/log/supervisor/kill-switch-error.log
stdout_logfile=/var/log/supervisor/kill-switch.log
environment=PATH="/home/tradebot/antbotNew/venv/bin"
EOF

# ============================================================================
# 6. NGINX CONFIGURATION FOR MONITORING DASHBOARD
# ============================================================================
echo "🌐 Setting up Nginx for monitoring dashboard..."

cat > /etc/nginx/sites-available/trading-bot << 'EOF'
server {
    listen 80;
    server_name _;
    
    # Basic auth for security
    auth_basic "Trading Bot Monitor";
    auth_basic_user_file /etc/nginx/.htpasswd;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    location /logs {
        alias /home/tradebot/antbotNew/worker_ant_v1/logs;
        autoindex on;
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8080/health;
    }
}
EOF

# Enable the site
ln -sf /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Create basic auth (change username/password)
echo "admin:$(openssl passwd -apr1 'TradingBot2024!')" > /etc/nginx/.htpasswd

# ============================================================================
# 7. LOG ROTATION SETUP
# ============================================================================
echo "📝 Setting up log rotation..."

cat > /etc/logrotate.d/trading-bot << 'EOF'
/home/tradebot/antbotNew/worker_ant_v1/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 644 tradebot tradebot
    postrotate
        supervisorctl restart enhanced-trading-bot
    endscript
}

/var/log/supervisor/trading-bot*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 644 root root
}
EOF

# ============================================================================
# 8. MONITORING & HEALTH CHECK SCRIPTS
# ============================================================================
echo "🏥 Setting up health monitoring..."

# Create health check script
cat > /home/tradebot/antbotNew/health_check.py << 'EOF'
#!/usr/bin/env python3
"""
Health Check Script for Trading Bot
Monitors system resources, bot performance, and trading metrics
"""

import psutil
import sqlite3
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add the project root to Python path
sys.path.append('/home/tradebot/antbotNew')

def check_system_resources():
    """Check system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'disk_percent': disk.percent,
        'memory_available_gb': memory.available / (1024**3),
        'disk_free_gb': disk.free / (1024**3)
    }

def check_bot_status():
    """Check if trading bot processes are running"""
    bot_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if 'worker_ant_v1' in cmdline and 'python' in proc.info['name']:
                bot_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cmdline': cmdline
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return bot_processes

def check_recent_trades():
    """Check recent trading activity"""
    db_path = '/home/tradebot/antbotNew/worker_ant_v1/trades.db'
    if not os.path.exists(db_path):
        return {'error': 'Database not found'}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get trades from last hour
        one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
        cursor.execute(
            "SELECT COUNT(*), AVG(profit_loss_sol) FROM trades WHERE timestamp > ?",
            (one_hour_ago,)
        )
        result = cursor.fetchone()
        
        conn.close()
        
        return {
            'trades_last_hour': result[0] or 0,
            'avg_profit_last_hour': result[1] or 0
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    """Main health check"""
    health_data = {
        'timestamp': datetime.now().isoformat(),
        'system': check_system_resources(),
        'processes': check_bot_status(),
        'trading': check_recent_trades()
    }
    
    # Print JSON for parsing
    print(json.dumps(health_data, indent=2))
    
    # Check for critical issues
    critical_issues = []
    
    if health_data['system']['cpu_percent'] > 90:
        critical_issues.append('High CPU usage')
    
    if health_data['system']['memory_percent'] > 90:
        critical_issues.append('High memory usage')
    
    if health_data['system']['disk_percent'] > 90:
        critical_issues.append('Low disk space')
    
    if len(health_data['processes']) == 0:
        critical_issues.append('No bot processes running')
    
    if critical_issues:
        print(f"\n🚨 CRITICAL ISSUES: {', '.join(critical_issues)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
EOF

chmod +x /home/tradebot/antbotNew/health_check.py
chown tradebot:tradebot /home/tradebot/antbotNew/health_check.py

# ============================================================================
# 9. CRON JOBS FOR MONITORING
# ============================================================================
echo "⏰ Setting up monitoring cron jobs..."

# Add cron jobs for tradebot user
su - tradebot << 'EOF'
# Add health check every 5 minutes
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/tradebot/antbotNew/health_check.py >> /home/tradebot/antbotNew/logs/health.log 2>&1") | crontab -

# Add daily database backup
(crontab -l 2>/dev/null; echo "0 2 * * * cp /home/tradebot/antbotNew/worker_ant_v1/trades.db /home/tradebot/antbotNew/data/backups/trades_$(date +\%Y\%m\%d).db") | crontab -

# Add system restart check (weekly)
(crontab -l 2>/dev/null; echo "0 3 * * 0 supervisorctl restart enhanced-trading-bot") | crontab -
EOF

# ============================================================================
# 10. START SERVICES
# ============================================================================
echo "🚀 Starting services..."

# Reload systemd and start services
systemctl daemon-reload
systemctl enable supervisor
systemctl start supervisor
systemctl enable nginx
systemctl start nginx

# Start supervisor programs
supervisorctl reread
supervisorctl update

echo "✅ Setup complete!"
echo ""
echo "🔧 NEXT STEPS:"
echo "1. Edit /home/tradebot/antbotNew/worker_ant_v1/.env with your actual values"
echo "2. Test the setup: supervisorctl start enhanced-trading-bot"
echo "3. Monitor logs: tail -f /var/log/supervisor/trading-bot.log"
echo "4. Check health: /home/tradebot/antbotNew/health_check.py"
echo "5. Access dashboard: http://YOUR_DROPLET_IP (admin/TradingBot2024!)"
echo ""
echo "⚠️  SECURITY REMINDERS:"
echo "- Change default passwords"
echo "- Configure your .env file"
echo "- Test with simulation mode first"
echo "- Monitor system resources"
echo ""
echo "🎯 Ready for deployment!" 