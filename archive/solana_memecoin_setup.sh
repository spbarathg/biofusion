#!/bin/bash
# ============================================================================
# SOLANA MEMECOIN TRADING BOT - DigitalOcean Production Setup
# ============================================================================
# Optimized specifically for Solana memecoin trading (25+ trades/hour)
# Run this script on a fresh Ubuntu 22.04 LTS droplet in Singapore region

set -e

echo "🟪 Starting Solana Memecoin Trading Bot Setup..."
echo "⚡ Optimizing for sub-500ms execution and 400+ trades/hour"

# ============================================================================
# 1. SOLANA-OPTIMIZED SYSTEM PACKAGES
# ============================================================================
echo "📦 Installing Solana-optimized packages..."
apt update && apt upgrade -y

apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    htop \
    iotop \
    nethogs \
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
    redis-server \
    cron \
    logrotate \
    jq \
    bc \
    ncdu

# Install Rust (required for Solana SDK performance)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# ============================================================================
# 2. EXTREME PERFORMANCE OPTIMIZATIONS FOR SOLANA TRADING
# ============================================================================
echo "⚡ Applying extreme performance optimizations for memecoin trading..."

# CPU Governor for maximum performance
echo 'performance' | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable unnecessary services for trading performance
systemctl disable bluetooth
systemctl disable cups
systemctl disable avahi-daemon
systemctl stop bluetooth cups avahi-daemon

# Memory optimizations for high-frequency trading
cat >> /etc/sysctl.conf << 'EOF'
# === SOLANA MEMECOIN TRADING OPTIMIZATIONS ===

# Network optimizations for sub-100ms latency
net.core.rmem_max = 536870912
net.core.wmem_max = 536870912
net.core.netdev_max_backlog = 10000
net.ipv4.tcp_rmem = 4096 131072 536870912
net.ipv4.tcp_wmem = 4096 131072 536870912
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq

# Reduce network latency
net.ipv4.tcp_low_latency = 1
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_timestamps = 0
net.ipv4.tcp_sack = 1
net.ipv4.tcp_window_scaling = 1

# Memory management for high-frequency operations
vm.swappiness = 1
vm.dirty_ratio = 10
vm.dirty_background_ratio = 5
vm.vfs_cache_pressure = 50

# File system optimizations
fs.file-max = 2097152
fs.inotify.max_user_watches = 1048576
EOF

sysctl -p

# ============================================================================
# 3. SOLANA RPC ENDPOINT OPTIMIZATION
# ============================================================================
echo "🟪 Setting up Solana RPC endpoint optimizations..."

# Create Solana RPC monitoring script
cat > /opt/check_solana_latency.sh << 'EOF'
#!/bin/bash
# Monitor Solana RPC latency and switch to best endpoint

ENDPOINTS=(
    "https://api.mainnet-beta.solana.com"
    "https://solana-api.projectserum.com"
    "https://rpc.ankr.com/solana"
    "https://solana.rpcpool.com"
)

BEST_ENDPOINT=""
BEST_LATENCY=9999

for endpoint in "${ENDPOINTS[@]}"; do
    # Test latency with getVersion call
    start_time=$(date +%s%3N)
    response=$(curl -s -X POST -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","id":1,"method":"getVersion"}' \
        "$endpoint" --max-time 5)
    end_time=$(date +%s%3N)
    
    if [[ $? -eq 0 ]] && [[ "$response" == *"solana-core"* ]]; then
        latency=$((end_time - start_time))
        echo "✅ $endpoint: ${latency}ms"
        
        if [[ $latency -lt $BEST_LATENCY ]]; then
            BEST_LATENCY=$latency
            BEST_ENDPOINT=$endpoint
        fi
    else
        echo "❌ $endpoint: FAILED"
    fi
done

echo "🚀 Best RPC: $BEST_ENDPOINT (${BEST_LATENCY}ms)"
echo "$BEST_ENDPOINT" > /tmp/best_solana_rpc
EOF

chmod +x /opt/check_solana_latency.sh

# ============================================================================
# 4. DEDICATED USER AND ENVIRONMENT SETUP
# ============================================================================
echo "👤 Setting up dedicated memecoin trading user..."

useradd -m -s /bin/bash memetrader || true
usermod -aG sudo memetrader

# Setup trading environment
su - memetrader << 'EOF'
cd /home/memetrader

# Clone repository (replace with your actual repo)
git clone https://github.com/your-username/antbotNew.git
cd antbotNew

# Create Python virtual environment with Solana optimizations
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Solana-specific packages for maximum performance
pip install \
    solana \
    solders \
    base58 \
    httpx[http2] \
    aioredis \
    psutil \
    prometheus-client \
    sentry-sdk \
    uvloop

# Create Solana memecoin specific environment
cp worker_ant_v1/.env.example worker_ant_v1/.env

cat >> worker_ant_v1/.env << 'SOLANA_ENV'

# === SOLANA MEMECOIN SPECIFIC SETTINGS ===
SOLANA_NETWORK=mainnet-beta
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
PRIVATE_RPC_URL=https://your-private-rpc-endpoint
BACKUP_RPC_URLS=https://solana-api.projectserum.com,https://rpc.ankr.com/solana

# Jupiter aggregator for Solana DEX routing
JUPITER_API_URL=https://quote-api.jup.ag/v6
JUPITER_PRICE_API_URL=https://price.jup.ag/v4

# Raydium DEX for memecoin pools
RAYDIUM_API_URL=https://api.raydium.io/v2

# Trading configuration for memecoins
TRADING_MODE=live
TRADE_AMOUNT_SOL=0.2
MAX_SLIPPAGE_PERCENT=3.0
ENTRY_TIMEOUT_MS=300
MAX_TRADES_PER_HOUR=25

# Memecoin specific filters
MIN_LIQUIDITY_SOL=10.0
MIN_LIQUIDITY_USD=2000
MAX_MARKET_CAP_USD=50000000

# Kill switch for memecoin volatility
SWARM_KILL_SWITCH_THRESHOLD=3.0
MAX_DAILY_LOSS_SOL=5.0

# Performance monitoring
ENABLE_PERFORMANCE_MONITORING=true
REDIS_URL=redis://localhost:6379
SOLANA_ENV
EOF

# ============================================================================
# 5. REDIS SETUP FOR HIGH-SPEED CACHING
# ============================================================================
echo "📦 Configuring Redis for memecoin price caching..."

# Optimize Redis for high-frequency trading
cat > /etc/redis/redis.conf << 'EOF'
# Redis configuration for Solana memecoin trading
bind 127.0.0.1
port 6379
timeout 0
tcp-keepalive 300

# Memory optimizations
maxmemory 2gb
maxmemory-policy allkeys-lru

# Performance optimizations for trading
save ""
appendonly no
tcp-nodelay yes
always-show-logo no

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log
EOF

systemctl enable redis-server
systemctl start redis-server

# ============================================================================
# 6. SOLANA MEMECOIN SPECIFIC PROCESS MANAGEMENT
# ============================================================================
echo "👑 Setting up Supervisor for Solana memecoin trading..."

cat > /etc/supervisor/conf.d/solana-memecoin-bot.conf << 'EOF'
[program:memecoin-scanner]
command=/home/memetrader/antbotNew/venv/bin/python -c "
import asyncio
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from worker_ant_v1.scanner import main_scanner_loop
asyncio.run(main_scanner_loop())
"
directory=/home/memetrader/antbotNew
user=memetrader
group=memetrader
autostart=true
autorestart=true
startretries=999
priority=100
stderr_logfile=/var/log/supervisor/memecoin-scanner-error.log
stdout_logfile=/var/log/supervisor/memecoin-scanner.log
environment=PATH="/home/memetrader/antbotNew/venv/bin",PYTHONPATH="/home/memetrader/antbotNew"

[program:memecoin-trader]
command=/home/memetrader/antbotNew/venv/bin/python -c "
import asyncio
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from worker_ant_v1.run_enhanced_bot import main
asyncio.run(main())
"
directory=/home/memetrader/antbotNew
user=memetrader
group=memetrader
autostart=true
autorestart=true
startretries=999
priority=200
stderr_logfile=/var/log/supervisor/memecoin-trader-error.log
stdout_logfile=/var/log/supervisor/memecoin-trader.log
environment=PATH="/home/memetrader/antbotNew/venv/bin",PYTHONPATH="/home/memetrader/antbotNew"

[program:kill-switch-monitor]
command=/home/memetrader/antbotNew/venv/bin/python -c "
import asyncio
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from worker_ant_v1.swarm_kill_switch import kill_switch_monitor
asyncio.run(kill_switch_monitor.run_continuous_monitoring())
"
directory=/home/memetrader/antbotNew
user=memetrader
group=memetrader
autostart=true
autorestart=true
startretries=999
priority=300
stderr_logfile=/var/log/supervisor/kill-switch-error.log
stdout_logfile=/var/log/supervisor/kill-switch.log
environment=PATH="/home/memetrader/antbotNew/venv/bin",PYTHONPATH="/home/memetrader/antbotNew"

[program:rpc-latency-monitor]
command=/opt/check_solana_latency.sh
autostart=true
autorestart=true
startretries=999
priority=50
stderr_logfile=/var/log/supervisor/rpc-monitor-error.log
stdout_logfile=/var/log/supervisor/rpc-monitor.log
EOF

# ============================================================================
# 7. SOLANA MEMECOIN MONITORING DASHBOARD
# ============================================================================
echo "📊 Setting up Solana memecoin monitoring dashboard..."

cat > /etc/nginx/sites-available/memecoin-dashboard << 'EOF'
server {
    listen 80;
    server_name _;
    
    # Basic auth for security
    auth_basic "Memecoin Trading Dashboard";
    auth_basic_user_file /etc/nginx/.htpasswd;
    
    # Main dashboard
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Real-time logs
    location /logs {
        alias /home/memetrader/antbotNew/worker_ant_v1/logs;
        autoindex on;
        autoindex_exact_size off;
    }
    
    # Health endpoint
    location /health {
        proxy_pass http://127.0.0.1:8080/health;
    }
    
    # Solana RPC status
    location /rpc-status {
        proxy_pass http://127.0.0.1:8080/rpc-status;
    }
    
    # Trading metrics
    location /metrics {
        proxy_pass http://127.0.0.1:8080/metrics;
    }
}
EOF

ln -sf /etc/nginx/sites-available/memecoin-dashboard /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Create secure authentication
echo "admin:$(openssl passwd -apr1 'SolanaMemecoin2024!')" > /etc/nginx/.htpasswd

# ============================================================================
# 8. SOLANA SPECIFIC MONITORING AND ALERTS
# ============================================================================
echo "🚨 Setting up Solana-specific monitoring..."

cat > /home/memetrader/antbotNew/solana_health_monitor.py << 'EOF'
#!/usr/bin/env python3
"""
Solana Memecoin Trading Health Monitor
Monitors Solana network, RPC endpoints, and trading performance
"""

import asyncio
import aiohttp
import json
import time
import psutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

sys.path.append('/home/memetrader/antbotNew')

async def check_solana_rpc_health():
    """Check Solana RPC endpoint health and latency"""
    
    endpoints = [
        "https://api.mainnet-beta.solana.com",
        "https://solana-api.projectserum.com", 
        "https://rpc.ankr.com/solana"
    ]
    
    results = {}
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            try:
                start_time = time.time()
                
                async with session.post(
                    endpoint,
                    json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    
                    latency_ms = int((time.time() - start_time) * 1000)
                    
                    if response.status == 200:
                        data = await response.json()
                        results[endpoint] = {
                            'status': 'healthy' if data.get('result') == 'ok' else 'unhealthy',
                            'latency_ms': latency_ms
                        }
                    else:
                        results[endpoint] = {
                            'status': 'error',
                            'latency_ms': latency_ms
                        }
                        
            except Exception as e:
                results[endpoint] = {
                    'status': 'failed',
                    'error': str(e),
                    'latency_ms': 9999
                }
    
    return results

async def check_memecoin_trading_performance():
    """Check recent memecoin trading performance"""
    
    db_path = '/home/memetrader/antbotNew/worker_ant_v1/trades.db'
    
    if not os.path.exists(db_path):
        return {'error': 'Trading database not found'}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get last hour's trading stats
        one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN success = 1 THEN 1 END) as successful_trades,
                AVG(latency_ms) as avg_latency_ms,
                AVG(profit_loss_sol) as avg_profit_sol,
                MAX(profit_loss_sol) as max_profit_sol,
                MIN(profit_loss_sol) as min_loss_sol
            FROM trades 
            WHERE timestamp > ?
        """, (one_hour_ago,))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'total_trades': result[0] or 0,
            'successful_trades': result[1] or 0,
            'win_rate': (result[1] / max(result[0], 1)) * 100,
            'avg_latency_ms': result[2] or 0,
            'avg_profit_sol': result[3] or 0,
            'max_profit_sol': result[4] or 0,
            'min_loss_sol': result[5] or 0
        }
        
    except Exception as e:
        return {'error': str(e)}

def check_system_resources():
    """Check system resource usage for trading optimization"""
    
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Check network connections
    net_connections = len(psutil.net_connections())
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'disk_percent': disk.percent,
        'memory_available_gb': memory.available / (1024**3),
        'disk_free_gb': disk.free / (1024**3),
        'network_connections': net_connections
    }

async def main():
    """Main health check routine"""
    
    print(f"🟪 Solana Memecoin Health Check - {datetime.now()}")
    print("=" * 60)
    
    # Check Solana RPC endpoints
    print("🔗 Checking Solana RPC endpoints...")
    rpc_health = await check_solana_rpc_health()
    
    for endpoint, health in rpc_health.items():
        status_icon = "✅" if health['status'] == 'healthy' else "❌"
        print(f"  {status_icon} {endpoint}: {health['status']} ({health.get('latency_ms', 'N/A')}ms)")
    
    # Check trading performance
    print("\n📈 Checking trading performance...")
    trading_stats = await check_memecoin_trading_performance()
    
    if 'error' not in trading_stats:
        print(f"  • Trades last hour: {trading_stats['total_trades']}")
        print(f"  • Win rate: {trading_stats['win_rate']:.1f}%")
        print(f"  • Avg latency: {trading_stats['avg_latency_ms']:.0f}ms")
        print(f"  • Avg profit: {trading_stats['avg_profit_sol']:.4f} SOL")
    else:
        print(f"  ❌ Error: {trading_stats['error']}")
    
    # Check system resources
    print("\n🖥️  Checking system resources...")
    system = check_system_resources()
    
    print(f"  • CPU usage: {system['cpu_percent']:.1f}%")
    print(f"  • Memory usage: {system['memory_percent']:.1f}%")
    print(f"  • Disk usage: {system['disk_percent']:.1f}%")
    print(f"  • Network connections: {system['network_connections']}")
    
    # Performance warnings
    print("\n⚠️  Performance Analysis:")
    
    issues = []
    if system['cpu_percent'] > 80:
        issues.append("High CPU usage - may affect trading speed")
    if system['memory_percent'] > 85:
        issues.append("High memory usage - risk of slowdowns")
    if any(health.get('latency_ms', 0) > 100 for health in rpc_health.values()):
        issues.append("High RPC latency - may miss fast trades")
    if trading_stats.get('avg_latency_ms', 0) > 500:
        issues.append("Slow trade execution - missing optimal entries")
    
    if issues:
        for issue in issues:
            print(f"  🚨 {issue}")
    else:
        print("  ✅ All systems optimal for memecoin trading")
    
    # Save health data for trending
    health_data = {
        'timestamp': datetime.now().isoformat(),
        'solana_rpc': rpc_health,
        'trading': trading_stats,
        'system': system
    }
    
    with open('/home/memetrader/antbotNew/logs/health_history.jsonl', 'a') as f:
        f.write(json.dumps(health_data) + '\n')

if __name__ == '__main__':
    asyncio.run(main())
EOF

chmod +x /home/memetrader/antbotNew/solana_health_monitor.py
chown memetrader:memetrader /home/memetrader/antbotNew/solana_health_monitor.py

# ============================================================================
# 9. AUTOMATED MEMECOIN TRADING CRON JOBS
# ============================================================================
echo "⏰ Setting up automated monitoring for memecoin trading..."

su - memetrader << 'EOF'
# Health check every 2 minutes (critical for memecoin trading)
(crontab -l 2>/dev/null; echo "*/2 * * * * /home/memetrader/antbotNew/solana_health_monitor.py >> /home/memetrader/antbotNew/logs/health.log 2>&1") | crontab -

# RPC latency check every minute
(crontab -l 2>/dev/null; echo "* * * * * /opt/check_solana_latency.sh >> /home/memetrader/antbotNew/logs/rpc_latency.log 2>&1") | crontab -

# Daily trading database backup
(crontab -l 2>/dev/null; echo "0 1 * * * cp /home/memetrader/antbotNew/worker_ant_v1/trades.db /home/memetrader/antbotNew/data/backups/trades_$(date +\%Y\%m\%d).db") | crontab -

# Weekly performance analysis
(crontab -l 2>/dev/null; echo "0 2 * * 1 /home/memetrader/antbotNew/venv/bin/python /home/memetrader/antbotNew/worker_ant_v1/system_overview.py > /home/memetrader/antbotNew/logs/weekly_report.log") | crontab -
EOF

# ============================================================================
# 10. FIREWALL OPTIMIZATION FOR SOLANA TRADING
# ============================================================================
echo "🔥 Setting up firewall optimized for Solana trading..."

ufw --force enable

# SSH access
ufw allow 22/tcp

# HTTP/HTTPS for monitoring
ufw allow 80/tcp
ufw allow 443/tcp

# Solana RPC endpoints (443/80 outbound)
ufw allow out 443/tcp
ufw allow out 80/tcp

# Allow high-frequency outbound connections for trading
ufw allow out 8899/tcp  # Solana RPC
ufw allow out 8900/tcp  # Solana RPC
ufw allow out 443/tcp   # HTTPS APIs

# Deny all other incoming
ufw default deny incoming
ufw default allow outgoing

# ============================================================================
# 11. START ALL SERVICES
# ============================================================================
echo "🚀 Starting Solana memecoin trading services..."

# Start and enable services
systemctl daemon-reload
systemctl enable supervisor nginx redis-server
systemctl start supervisor nginx redis-server

# Load supervisor configurations
supervisorctl reread
supervisorctl update

echo "✅ Solana Memecoin Trading Bot Setup Complete!"
echo ""
echo "🎯 SOLANA MEMECOIN TRADING CONFIGURATION:"
echo "  • Optimized for: 25+ trades/hour, <300ms execution"
echo "  • Target: Singapore region for optimal Solana latency"
echo "  • Performance: Sub-100ms RPC latency, BBR congestion control"
echo "  • Monitoring: Real-time health checks every 2 minutes"
echo ""
echo "🔧 NEXT STEPS:"
echo "1. Edit /home/memetrader/antbotNew/worker_ant_v1/.env with your values"
echo "2. Add your Solana wallet private keys (SECURE!)"
echo "3. Configure Jupiter API and private RPC endpoints"
echo "4. Test with: supervisorctl start memecoin-scanner"
echo "5. Monitor: http://YOUR_IP (admin/SolanaMemecoin2024!)"
echo "6. Health check: /home/memetrader/antbotNew/solana_health_monitor.py"
echo ""
echo "⚠️  CRITICAL REMINDERS:"
echo "  • Use Singapore region for lowest Solana latency"
echo "  • Configure private RPC endpoints for best performance"
echo "  • Start with small amounts until proven profitable"
echo "  • Monitor RPC latency - should be <50ms"
echo ""
echo "🟪 Ready for Solana memecoin trading!" 