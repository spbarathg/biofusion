#!/usr/bin/env python3
"""
SMART APE MODE - REAL-TIME MONITORING DASHBOARD
==============================================

Complete monitoring system for 24/7 tracking of bot activities.
Provides real-time insights into trades, performance, system health, and alerts.
"""

import asyncio
import json
import os
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

class SmartApeMonitor:
    """Real-time monitoring dashboard for Smart Ape Mode bot"""
    
    def __init__(self):
        self.setup_logging()
        self.start_time = datetime.now()
        self.monitoring_active = True
        
        # Tracking data
        self.trades_data = []
        self.performance_data = []
        self.system_health = {}
        self.alerts = []
        
        # Statistics
        self.session_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_profit_sol': 0.0,
            'current_balance': 300.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'uptime_hours': 0.0
        }
        
    def setup_logging(self):
        """Setup monitoring logger"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("SmartApeMonitor")
    
    async def start_monitoring(self):
        """Start the monitoring dashboard"""
        
        self.logger.info("🖥️ SMART APE MODE MONITORING DASHBOARD STARTING")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self.monitor_logs()),
            asyncio.create_task(self.monitor_system_health()),
            asyncio.create_task(self.monitor_performance()),
            asyncio.create_task(self.generate_reports()),
            asyncio.create_task(self.display_dashboard())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("📊 Monitoring dashboard stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Monitoring error: {e}")
    
    async def monitor_logs(self):
        """Monitor log files for trading activity"""
        
        log_files = [
            'logs/smartapelauncher.log',
            'logs/tradinglogger.log',
            'logs/worker_ant_v1.trading.market_scanner.log',
            'logs/worker_ant_v1.trading.order_buyer.log',
            'logs/worker_ant_v1.trading.order_seller.log',
            'logs/error.log'
        ]
        
        last_positions = {file: 0 for file in log_files}
        
        while self.monitoring_active:
            try:
                for log_file in log_files:
                    if os.path.exists(log_file):
                        await self.parse_log_file(log_file, last_positions[log_file])
                        
                        # Update position
                        try:
                            last_positions[log_file] = os.path.getsize(log_file)
                        except:
                            pass
                
                await asyncio.sleep(2)  # Check logs every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Log monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def parse_log_file(self, log_file: str, last_position: int):
        """Parse log file for new entries"""
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                
                for line in new_lines:
                    await self.process_log_line(line.strip(), log_file)
                    
        except Exception as e:
            # File might be locked or rotated
            pass
    
    async def process_log_line(self, line: str, source_file: str):
        """Process a single log line"""
        
        if not line:
            return
            
        # Extract trade information
        if "TRADE #" in line and ("SUCCESS" in line or "FAILED" in line):
            await self.extract_trade_data(line)
        
        # Extract performance updates
        elif "Performance Update:" in line:
            await self.extract_performance_data(line)
        
        # Extract errors and warnings
        elif "ERROR" in line or "WARNING" in line or "CRITICAL" in line:
            await self.extract_alert_data(line, source_file)
        
        # Extract buy/sell activities
        elif ("BUY SUCCESS:" in line or "BUY FAILED:" in line or 
              "SELL SUCCESS:" in line or "SELL FAILED:" in line):
            await self.extract_execution_data(line)
    
    async def extract_trade_data(self, line: str):
        """Extract trade data from log line"""
        
        try:
            # Parse trade line format: "TRADE #X | TYPE | Token: XXX... | Amount: X | Price: $X | STATUS"
            parts = line.split(" | ")
            
            trade_data = {
                'timestamp': datetime.now().isoformat(),
                'trade_number': int(parts[0].split("#")[1]) if "#" in parts[0] else 0,
                'trade_type': parts[1].strip(),
                'token': parts[2].split(": ")[1] if "Token:" in parts[2] else "Unknown",
                'amount': float(parts[3].split(": ")[1]) if "Amount:" in parts[3] else 0.0,
                'price': float(parts[4].split(": $")[1]) if "Price: $" in parts[4] else 0.0,
                'success': "SUCCESS" in parts[-1],
                'source': 'log_parser'
            }
            
            self.trades_data.append(trade_data)
            
            # Update session stats
            self.session_stats['total_trades'] += 1
            if trade_data['success']:
                self.session_stats['successful_trades'] += 1
            
            # Calculate win rate
            if self.session_stats['total_trades'] > 0:
                self.session_stats['win_rate'] = (
                    self.session_stats['successful_trades'] / 
                    self.session_stats['total_trades'] * 100
                )
            
            self.logger.info(f"📊 Trade logged: {trade_data['trade_type']} - {'✅' if trade_data['success'] else '❌'}")
            
        except Exception as e:
            self.logger.error(f"Error parsing trade data: {e}")
    
    async def extract_performance_data(self, line: str):
        """Extract performance data from log line"""
        
        try:
            # Parse performance line format
            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'runtime_hours': 0.0,
                'trades_count': 0,
                'balance': 0.0,
                'pnl_percent': 0.0,
                'source': 'performance_tracker'
            }
            
            # Extract runtime
            if "Runtime:" in line:
                runtime_str = line.split("Runtime: ")[1].split("h,")[0]
                performance_data['runtime_hours'] = float(runtime_str)
            
            # Extract trades
            if "Trades:" in line:
                trades_str = line.split("Trades: ")[1].split(",")[0]
                performance_data['trades_count'] = int(trades_str)
            
            # Extract balance
            if "Balance: $" in line:
                balance_str = line.split("Balance: $")[1].split(",")[0]
                performance_data['balance'] = float(balance_str)
                self.session_stats['current_balance'] = performance_data['balance']
            
            # Extract P&L
            if "P&L:" in line:
                pnl_str = line.split("P&L: ")[1].split("%")[0]
                performance_data['pnl_percent'] = float(pnl_str)
            
            self.performance_data.append(performance_data)
            
            # Update session stats
            profit = self.session_stats['current_balance'] - 300.0
            self.session_stats['total_profit_sol'] = profit
            
            self.logger.info(f"📈 Performance updated: Balance ${performance_data['balance']:.2f}, P&L {performance_data['pnl_percent']:+.2f}%")
            
        except Exception as e:
            self.logger.error(f"Error parsing performance data: {e}")
    
    async def extract_alert_data(self, line: str, source_file: str):
        """Extract alert/error data from log line"""
        
        try:
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'level': 'ERROR' if 'ERROR' in line else 'WARNING' if 'WARNING' in line else 'CRITICAL',
                'message': line,
                'source_file': source_file,
                'resolved': False
            }
            
            self.alerts.append(alert_data)
            
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
            
            self.logger.warning(f"🚨 Alert: {alert_data['level']} - {source_file}")
            
        except Exception as e:
            self.logger.error(f"Error parsing alert data: {e}")
    
    async def extract_execution_data(self, line: str):
        """Extract buy/sell execution data"""
        
        try:
            execution_data = {
                'timestamp': datetime.now().isoformat(),
                'type': 'BUY' if 'BUY' in line else 'SELL',
                'success': 'SUCCESS' in line,
                'token': 'Unknown',
                'amount': 0.0,
                'latency_ms': 0,
                'slippage': 0.0
            }
            
            # Extract token symbol
            if ":" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    execution_data['token'] = parts[1].split(" |")[0].strip()
            
            # Extract amounts and metrics
            if "SOL" in line:
                sol_parts = line.split("SOL")
                for part in sol_parts:
                    try:
                        # Look for amount before SOL
                        amount_str = part.split()[-1]
                        execution_data['amount'] = float(amount_str)
                        break
                    except:
                        continue
            
            # Extract latency
            if "ms" in line:
                ms_parts = line.split("ms")
                for part in ms_parts:
                    try:
                        latency_str = part.split()[-1]
                        execution_data['latency_ms'] = int(latency_str)
                        break
                    except:
                        continue
            
            # Extract slippage
            if "Slippage:" in line:
                slippage_str = line.split("Slippage: ")[1].split("%")[0]
                execution_data['slippage'] = float(slippage_str)
            
            self.logger.info(f"⚡ Execution: {execution_data['type']} - {'✅' if execution_data['success'] else '❌'} - {execution_data['latency_ms']}ms")
            
        except Exception as e:
            self.logger.error(f"Error parsing execution data: {e}")
    
    async def monitor_system_health(self):
        """Monitor system health metrics"""
        
        while self.monitoring_active:
            try:
                # Get system metrics
                self.system_health = await self.get_system_metrics()
                
                # Check for concerning metrics
                await self.check_health_alerts()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"System health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        
        try:
            # Try to use psutil if available
            try:
                import psutil
                return {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                    'disk_usage_percent': psutil.disk_usage('.').percent,
                    'network_connections': len(psutil.net_connections()),
                    'process_count': len(psutil.pids()),
                    'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                    'uptime_hours': (time.time() - psutil.boot_time()) / 3600
                }
            except ImportError:
                # Fallback metrics
                return {
                    'cpu_percent': 0.0,
                    'memory_percent': 0.0,
                    'memory_available_gb': 4.0,
                    'disk_usage_percent': 50.0,
                    'network_connections': 50,
                    'process_count': 200,
                    'boot_time': datetime.now().isoformat(),
                    'uptime_hours': 24.0,
                    'status': 'metrics_unavailable'
                }
                
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {'error': str(e)}
    
    async def check_health_alerts(self):
        """Check system health and generate alerts if needed"""
        
        health = self.system_health
        
        # CPU alert
        if health.get('cpu_percent', 0) > 85:
            await self.add_health_alert("HIGH_CPU", f"CPU usage: {health['cpu_percent']:.1f}%")
        
        # Memory alert
        if health.get('memory_percent', 0) > 90:
            await self.add_health_alert("HIGH_MEMORY", f"Memory usage: {health['memory_percent']:.1f}%")
        
        # Disk alert
        if health.get('disk_usage_percent', 0) > 85:
            await self.add_health_alert("HIGH_DISK", f"Disk usage: {health['disk_usage_percent']:.1f}%")
    
    async def add_health_alert(self, alert_type: str, message: str):
        """Add a health alert"""
        
        # Check if we already have this alert recently
        recent_alerts = [a for a in self.alerts[-10:] if alert_type in a.get('message', '')]
        if recent_alerts:
            return  # Don't spam same alert
            
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': 'WARNING',
            'type': alert_type,
            'message': f"SYSTEM HEALTH: {message}",
            'source_file': 'system_monitor',
            'resolved': False
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"🚨 Health Alert: {message}")
    
    async def monitor_performance(self):
        """Monitor bot performance and calculate metrics"""
        
        while self.monitoring_active:
            try:
                # Update session uptime
                uptime = datetime.now() - self.start_time
                self.session_stats['uptime_hours'] = uptime.total_seconds() / 3600
                
                # Calculate performance metrics
                await self.calculate_advanced_metrics()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def calculate_advanced_metrics(self):
        """Calculate advanced performance metrics"""
        
        try:
            # Calculate max drawdown
            if self.performance_data:
                balances = [p.get('balance', 300.0) for p in self.performance_data]
                peak_balance = max(balances) if balances else 300.0
                current_balance = self.session_stats['current_balance']
                
                if peak_balance > 0:
                    drawdown = ((peak_balance - current_balance) / peak_balance) * 100
                    self.session_stats['max_drawdown'] = max(self.session_stats['max_drawdown'], drawdown)
            
            # Calculate other metrics as needed
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
    
    async def generate_reports(self):
        """Generate periodic reports"""
        
        while self.monitoring_active:
            try:
                # Generate hourly report
                await self.generate_hourly_report()
                
                # Generate daily summary (if uptime > 24h)
                if self.session_stats['uptime_hours'] >= 24:
                    await self.generate_daily_summary()
                
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                self.logger.error(f"Report generation error: {e}")
                await asyncio.sleep(3600)
    
    async def generate_hourly_report(self):
        """Generate hourly performance report"""
        
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'period': 'hourly',
                'session_stats': self.session_stats.copy(),
                'recent_trades': len([t for t in self.trades_data if 
                                    datetime.fromisoformat(t['timestamp']) > 
                                    datetime.now() - timedelta(hours=1)]),
                'system_health': self.system_health.copy(),
                'active_alerts': len([a for a in self.alerts if not a.get('resolved', False)])
            }
            
            # Save report
            report_file = f"logs/hourly_report_{datetime.now().strftime('%Y%m%d_%H')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"📋 Hourly report generated: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating hourly report: {e}")
    
    async def generate_daily_summary(self):
        """Generate daily summary report"""
        
        try:
            # Calculate daily metrics
            daily_trades = [t for t in self.trades_data if 
                          datetime.fromisoformat(t['timestamp']) > 
                          datetime.now() - timedelta(days=1)]
            
            daily_summary = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'total_trades_24h': len(daily_trades),
                'successful_trades_24h': len([t for t in daily_trades if t['success']]),
                'win_rate_24h': (len([t for t in daily_trades if t['success']]) / 
                               max(len(daily_trades), 1)) * 100,
                'session_stats': self.session_stats.copy(),
                'system_health_summary': self.system_health.copy()
            }
            
            # Save daily summary
            summary_file = f"logs/daily_summary_{datetime.now().strftime('%Y%m%d')}.json"
            with open(summary_file, 'w') as f:
                json.dump(daily_summary, f, indent=2)
            
            self.logger.info(f"📊 Daily summary generated: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating daily summary: {e}")
    
    async def display_dashboard(self):
        """Display real-time dashboard"""
        
        while self.monitoring_active:
            try:
                # Clear screen (cross-platform)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Display dashboard
                self.print_dashboard()
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Dashboard display error: {e}")
                await asyncio.sleep(10)
    
    def print_dashboard(self):
        """Print the monitoring dashboard"""
        
        print("=" * 100)
        print("🧠 SMART APE MODE - REAL-TIME MONITORING DASHBOARD")
        print("=" * 100)
        
        # Session Overview
        print(f"\n📊 SESSION OVERVIEW")
        print(f"├─ Uptime: {self.session_stats['uptime_hours']:.1f} hours")
        print(f"├─ Total Trades: {self.session_stats['total_trades']}")
        print(f"├─ Successful: {self.session_stats['successful_trades']} ({self.session_stats['win_rate']:.1f}%)")
        print(f"├─ Current Balance: ${self.session_stats['current_balance']:.2f}")
        print(f"├─ Total Profit: ${self.session_stats['total_profit_sol']:.2f}")
        print(f"└─ Max Drawdown: {self.session_stats['max_drawdown']:.2f}%")
        
        # Footer
        print(f"\n{'=' * 100}")
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Press Ctrl+C to stop monitoring")
        print("=" * 100)


async def main():
    """Main monitoring function"""
    
    monitor = SmartApeMonitor()
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n📊 Monitoring dashboard stopped")
    except Exception as e:
        print(f"❌ Monitoring error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 