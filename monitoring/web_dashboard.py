#!/usr/bin/env python3
"""
SMART APE MODE - WEB MONITORING DASHBOARD
==========================================

Simple web dashboard for remote monitoring of the trading bot.
Provides real-time status, trades, and system health via web interface.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import asyncio

try:
    from flask import Flask, render_template_string, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

class WebDashboard:
    """Simple web dashboard for bot monitoring"""
    
    def __init__(self, port: int = 5000):
        self.port = port
        
        if not FLASK_AVAILABLE:
            print("⚠️ Flask not available. Install with: pip install flask")
            return
            
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Setup web routes"""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self.get_dashboard_html())
        
        @self.app.route('/api/status')
        def api_status():
            return jsonify(self.get_bot_status())
        
        @self.app.route('/api/trades')
        def api_trades():
            return jsonify(self.get_recent_trades())
        
        @self.app.route('/api/logs')
        def api_logs():
            return jsonify(self.get_recent_logs())
    
    def get_dashboard_html(self) -> str:
        """Get dashboard HTML template"""
        
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Smart Ape Mode Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }
        .header { text-align: center; margin-bottom: 30px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .status-card { background: #2a2a2a; padding: 20px; border-radius: 8px; border-left: 4px solid #00ff00; }
        .status-card.warning { border-left-color: #ffaa00; }
        .status-card.error { border-left-color: #ff4444; }
        .status-value { font-size: 2em; font-weight: bold; margin: 10px 0; }
        .trades-section { background: #2a2a2a; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .trade-item { padding: 10px; border-bottom: 1px solid #444; display: flex; justify-content: space-between; }
        .logs-section { background: #2a2a2a; padding: 20px; border-radius: 8px; max-height: 400px; overflow-y: auto; }
        .log-item { font-family: monospace; font-size: 12px; padding: 5px; border-bottom: 1px solid #333; }
        .success { color: #00ff00; }
        .error { color: #ff4444; }
        .warning { color: #ffaa00; }
        .refresh-btn { background: #0066cc; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Smart Ape Mode Dashboard</h1>
        <button class="refresh-btn" onclick="refreshData()">🔄 Refresh</button>
        <span id="last-update">Last updated: --</span>
    </div>
    
    <div class="status-grid" id="status-grid">
        <div class="status-card">
            <h3>Bot Status</h3>
            <div class="status-value" id="bot-status">--</div>
        </div>
        
        <div class="status-card">
            <h3>Current Balance</h3>
            <div class="status-value" id="current-balance">$--</div>
        </div>
        
        <div class="status-card">
            <h3>Total P&L</h3>
            <div class="status-value" id="total-pnl">$--</div>
        </div>
        
        <div class="status-card">
            <h3>Trades Today</h3>
            <div class="status-value" id="trades-today">--</div>
        </div>
        
        <div class="status-card">
            <h3>Win Rate</h3>
            <div class="status-value" id="win-rate">--%</div>
        </div>
        
        <div class="status-card">
            <h3>Uptime</h3>
            <div class="status-value" id="uptime">--h</div>
        </div>
    </div>
    
    <div class="trades-section">
        <h3>📊 Recent Trades</h3>
        <div id="trades-list">
            <div>No trades data available</div>
        </div>
    </div>
    
    <div class="logs-section">
        <h3>📝 Recent Logs</h3>
        <div id="logs-list">
            <div>No logs available</div>
        </div>
    </div>
    
    <script>
        async function refreshData() {
            try {
                // Update status
                const statusResponse = await fetch('/api/status');
                const status = await statusResponse.json();
                updateStatus(status);
                
                // Update trades
                const tradesResponse = await fetch('/api/trades');
                const trades = await tradesResponse.json();
                updateTrades(trades);
                
                // Update logs
                const logsResponse = await fetch('/api/logs');
                const logs = await logsResponse.json();
                updateLogs(logs);
                
                document.getElementById('last-update').textContent = 
                    'Last updated: ' + new Date().toLocaleTimeString();
                    
            } catch (error) {
                console.error('Error refreshing data:', error);
            }
        }
        
        function updateStatus(status) {
            document.getElementById('bot-status').textContent = status.running ? '🟢 RUNNING' : '🔴 STOPPED';
            document.getElementById('current-balance').textContent = '$' + status.current_balance.toFixed(2);
            document.getElementById('total-pnl').textContent = '$' + status.profit_loss.toFixed(2);
            document.getElementById('trades-today').textContent = status.trades_today;
            document.getElementById('win-rate').textContent = status.win_rate.toFixed(1) + '%';
            document.getElementById('uptime').textContent = status.uptime_hours.toFixed(1) + 'h';
        }
        
        function updateTrades(trades) {
            const tradesList = document.getElementById('trades-list');
            if (trades.length === 0) {
                tradesList.innerHTML = '<div>No recent trades</div>';
                return;
            }
            
            tradesList.innerHTML = trades.map(trade => `
                <div class="trade-item">
                    <span>${trade.timestamp} | ${trade.type} | ${trade.token}</span>
                    <span class="${trade.success ? 'success' : 'error'}">${trade.success ? '✅' : '❌'}</span>
                </div>
            `).join('');
        }
        
        function updateLogs(logs) {
            const logsList = document.getElementById('logs-list');
            if (logs.length === 0) {
                logsList.innerHTML = '<div>No recent logs</div>';
                return;
            }
            
            logsList.innerHTML = logs.map(log => `
                <div class="log-item ${log.level.toLowerCase()}">${log.timestamp} | ${log.message}</div>
            `).join('');
        }
        
        // Auto-refresh every 30 seconds
        setInterval(refreshData, 30000);
        
        // Initial load
        refreshData();
    </script>
</body>
</html>
        '''
    
    def get_bot_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        
        try:
            # Parse logs to get status
            return {
                'running': self.check_bot_running(),
                'current_balance': self.get_current_balance(),
                'profit_loss': self.get_current_balance() - 300.0,
                'trades_today': self.count_trades_today(),
                'win_rate': self.calculate_win_rate(),
                'uptime_hours': self.get_uptime_hours(),
                'last_trade': self.get_last_trade_time(),
                'alerts_count': self.count_alerts()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def check_bot_running(self) -> bool:
        """Check if bot is currently running"""
        try:
            # Check recent log activity
            log_files = ['logs/smartapelauncher.log', 'logs/tradinglogger.log']
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    mod_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                    if datetime.now() - mod_time < timedelta(minutes=5):
                        return True
            return False
        except:
            return False
    
    def get_current_balance(self) -> float:
        """Get current balance from logs"""
        try:
            balance = 300.0
            
            # Read logs for balance updates
            if os.path.exists('logs/smartapelauncher.log'):
                with open('logs/smartapelauncher.log', 'r') as f:
                    lines = f.readlines()
                    for line in reversed(lines[-50:]):
                        if 'Balance: $' in line:
                            balance_str = line.split('Balance: $')[1].split(',')[0]
                            balance = float(balance_str)
                            break
            
            return balance
        except:
            return 300.0
    
    def count_trades_today(self) -> int:
        """Count trades executed today"""
        try:
            count = 0
            today = datetime.now().strftime('%Y-%m-%d')
            
            log_files = ['logs/tradinglogger.log', 'logs/smartapelauncher.log']
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        for line in f:
                            if today in line and 'TRADE #' in line:
                                count += 1
            
            return count
        except:
            return 0
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate from recent trades"""
        try:
            total_trades = 0
            successful_trades = 0
            
            log_files = ['logs/tradinglogger.log', 'logs/smartapelauncher.log']
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        for line in f:
                            if 'TRADE #' in line:
                                total_trades += 1
                                if 'SUCCESS' in line:
                                    successful_trades += 1
            
            return (successful_trades / total_trades * 100) if total_trades > 0 else 0.0
        except:
            return 0.0
    
    def get_uptime_hours(self) -> float:
        """Get uptime in hours"""
        try:
            # Find first log entry today
            today = datetime.now().strftime('%Y-%m-%d')
            
            if os.path.exists('logs/smartapelauncher.log'):
                with open('logs/smartapelauncher.log', 'r') as f:
                    for line in f:
                        if today in line:
                            # Parse timestamp
                            timestamp_str = line.split(' |')[0].strip()
                            start_time = datetime.strptime(f"{today} {timestamp_str.split()[-1]}", '%Y-%m-%d %H:%M:%S')
                            uptime = datetime.now() - start_time
                            return uptime.total_seconds() / 3600
            
            return 0.0
        except:
            return 0.0
    
    def get_last_trade_time(self) -> str:
        """Get last trade timestamp"""
        try:
            log_files = ['logs/tradinglogger.log', 'logs/smartapelauncher.log']
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        for line in reversed(lines[-50:]):
                            if 'TRADE #' in line:
                                return line.split(' |')[0].strip()
            
            return "No trades yet"
        except:
            return "Unknown"
    
    def count_alerts(self) -> int:
        """Count recent alerts"""
        try:
            count = 0
            if os.path.exists('logs/error.log'):
                with open('logs/error.log', 'r') as f:
                    for line in f:
                        if 'ERROR' in line or 'WARNING' in line:
                            count += 1
            return count
        except:
            return 0
    
    def get_recent_trades(self) -> List[Dict[str, Any]]:
        """Get recent trades data"""
        try:
            trades = []
            log_files = ['logs/tradinglogger.log', 'logs/smartapelauncher.log']
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        for line in reversed(lines[-20:]):
                            if 'TRADE #' in line:
                                parts = line.split(' | ')
                                trades.append({
                                    'timestamp': parts[0].strip() if len(parts) > 0 else 'Unknown',
                                    'type': parts[1].strip() if len(parts) > 1 else 'Unknown',
                                    'token': parts[2].split(': ')[1][:8] + '...' if len(parts) > 2 and ': ' in parts[2] else 'Unknown',
                                    'success': 'SUCCESS' in line
                                })
                                
                                if len(trades) >= 10:
                                    break
            
            return trades[:10]
        except:
            return []
    
    def get_recent_logs(self) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        try:
            logs = []
            
            if os.path.exists('logs/error.log'):
                with open('logs/error.log', 'r') as f:
                    lines = f.readlines()
                    for line in reversed(lines[-20:]):
                        if line.strip():
                            level = 'ERROR' if 'ERROR' in line else 'WARNING' if 'WARNING' in line else 'INFO'
                            logs.append({
                                'timestamp': line.split(' -')[0].strip() if ' -' in line else 'Unknown',
                                'level': level,
                                'message': line.strip()
                            })
            
            return logs[:20]
        except:
            return []
    
    def run(self):
        """Run the web dashboard"""
        if not FLASK_AVAILABLE:
            print("❌ Flask not available. Cannot start web dashboard.")
            return
        
        print(f"🌐 Starting web dashboard on http://localhost:{self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    dashboard = WebDashboard(port=5000)
    dashboard.run() 