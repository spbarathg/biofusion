#!/usr/bin/env python3
"""
SMART APE MODE - TELEGRAM MONITORING BOT
========================================

Remote monitoring and control via Telegram for 24/7 operation.
Get real-time updates, alerts, and control commands remotely.
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

class TelegramMonitor:
    """Telegram bot for remote monitoring"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logging.getLogger("TelegramMonitor")
        
        # Monitoring state
        self.monitoring_active = True
        self.last_update_time = datetime.now()
        
        # Cache for bot status
        self.bot_status = {
            'running': False,
            'trades_today': 0,
            'current_balance': 300.0,
            'profit_loss': 0.0,
            'last_trade_time': None,
            'alerts_count': 0
        }
    
    async def start_monitoring(self):
        """Start Telegram monitoring"""
        
        self.logger.info("📱 Starting Telegram monitoring...")
        
        # Send startup notification
        await self.send_message("🚀 Smart Ape Mode Telegram Monitor Started!")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self.monitor_bot_status()),
            asyncio.create_task(self.send_periodic_updates()),
            asyncio.create_task(self.monitor_alerts())
        ]
        
        await asyncio.gather(*tasks)
    
    async def send_message(self, message: str, parse_mode: str = None):
        """Send message to Telegram"""
        
        try:
            # This would use the Telegram Bot API
            # For now, we'll log the message
            self.logger.info(f"📱 TELEGRAM: {message}")
            
            # In production, you'd use:
            # import aiohttp
            # url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            # data = {'chat_id': self.chat_id, 'text': message}
            # if parse_mode:
            #     data['parse_mode'] = parse_mode
            # async with aiohttp.ClientSession() as session:
            #     await session.post(url, data=data)
            
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
    
    async def monitor_bot_status(self):
        """Monitor bot status and update cache"""
        
        while self.monitoring_active:
            try:
                # Read bot status from logs/files
                status = await self.get_bot_status()
                
                # Check for significant changes
                if status != self.bot_status:
                    await self.handle_status_change(status)
                    self.bot_status = status
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring bot status: {e}")
                await asyncio.sleep(60)
    
    async def get_bot_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        
        try:
            # Check if bot process is running
            running = await self.check_bot_process()
            
            # Read recent log data
            trades_today = await self.count_trades_today()
            current_balance = await self.get_current_balance()
            alerts = await self.count_active_alerts()
            
            return {
                'running': running,
                'trades_today': trades_today,
                'current_balance': current_balance,
                'profit_loss': current_balance - 300.0,
                'last_trade_time': await self.get_last_trade_time(),
                'alerts_count': alerts,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting bot status: {e}")
            return self.bot_status
    
    async def check_bot_process(self) -> bool:
        """Check if bot process is running"""
        
        try:
            # Check for Python processes running the bot
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'launch_smart_ape.py' in cmdline or 'main_launcher.py' in cmdline:
                        return True
                except:
                    continue
            return False
        except ImportError:
            # Fallback: check for log activity
            return await self.check_recent_log_activity()
    
    async def check_recent_log_activity(self) -> bool:
        """Check for recent log activity"""
        
        try:
            log_files = ['logs/smartapelauncher.log', 'logs/tradinglogger.log']
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    # Check modification time
                    mod_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                    if datetime.now() - mod_time < timedelta(minutes=5):
                        return True
            return False
        except:
            return False
    
    async def count_trades_today(self) -> int:
        """Count trades executed today"""
        
        try:
            count = 0
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Read trading logs
            for log_file in ['logs/tradinglogger.log', 'logs/smartapelauncher.log']:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        for line in f:
                            if today in line and 'TRADE #' in line:
                                count += 1
            
            return count
        except:
            return 0
    
    async def get_current_balance(self) -> float:
        """Get current balance from logs"""
        
        try:
            balance = 300.0  # Default
            
            # Read latest performance log
            log_files = ['logs/smartapelauncher.log', 'logs/tradinglogger.log']
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        
                        # Look for balance in reverse order (latest first)
                        for line in reversed(lines[-100:]):
                            if 'Balance: $' in line:
                                try:
                                    balance_str = line.split('Balance: $')[1].split(',')[0]
                                    balance = float(balance_str)
                                    break
                                except:
                                    continue
            
            return balance
        except:
            return 300.0
    
    async def get_last_trade_time(self) -> str:
        """Get timestamp of last trade"""
        
        try:
            # Find most recent trade in logs
            log_files = ['logs/tradinglogger.log', 'logs/smartapelauncher.log']
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        
                        for line in reversed(lines[-50:]):
                            if 'TRADE #' in line:
                                # Extract timestamp from log line
                                timestamp_str = line.split(' |')[0].strip()
                                return timestamp_str
            
            return "No trades yet"
        except:
            return "Unknown"
    
    async def count_active_alerts(self) -> int:
        """Count active alerts from error logs"""
        
        try:
            count = 0
            recent_time = datetime.now() - timedelta(hours=1)
            
            if os.path.exists('logs/error.log'):
                with open('logs/error.log', 'r') as f:
                    for line in f:
                        if 'ERROR' in line or 'WARNING' in line:
                            count += 1
            
            return count
        except:
            return 0
    
    async def handle_status_change(self, new_status: Dict[str, Any]):
        """Handle significant status changes"""
        
        old_status = self.bot_status
        
        # Bot stopped
        if old_status.get('running', False) and not new_status.get('running', False):
            await self.send_message("🚨 ALERT: Smart Ape Mode bot has STOPPED!")
        
        # Bot started
        if not old_status.get('running', False) and new_status.get('running', False):
            await self.send_message("✅ Smart Ape Mode bot has STARTED")
        
        # Large profit/loss change
        old_pnl = old_status.get('profit_loss', 0.0)
        new_pnl = new_status.get('profit_loss', 0.0)
        
        if abs(new_pnl - old_pnl) > 10.0:  # $10 change
            direction = "📈 PROFIT" if new_pnl > old_pnl else "📉 LOSS"
            await self.send_message(
                f"{direction}: ${new_pnl:+.2f} (Change: ${new_pnl - old_pnl:+.2f})"
            )
        
        # New alerts
        old_alerts = old_status.get('alerts_count', 0)
        new_alerts = new_status.get('alerts_count', 0)
        
        if new_alerts > old_alerts:
            await self.send_message(f"⚠️ NEW ALERTS: {new_alerts - old_alerts} new alerts detected")
    
    async def send_periodic_updates(self):
        """Send periodic status updates"""
        
        while self.monitoring_active:
            try:
                # Send hourly update
                await self.send_status_update()
                
                # Wait 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Error sending periodic update: {e}")
                await asyncio.sleep(3600)
    
    async def send_status_update(self):
        """Send comprehensive status update"""
        
        status = self.bot_status
        
        message = f"""📊 SMART APE MODE STATUS UPDATE
        
🔹 Status: {'🟢 RUNNING' if status.get('running', False) else '🔴 STOPPED'}
🔹 Balance: ${status.get('current_balance', 300.0):.2f}
🔹 P&L: ${status.get('profit_loss', 0.0):+.2f}
🔹 Trades Today: {status.get('trades_today', 0)}
🔹 Last Trade: {status.get('last_trade_time', 'N/A')}
🔹 Active Alerts: {status.get('alerts_count', 0)}

📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        await self.send_message(message)
    
    async def monitor_alerts(self):
        """Monitor for critical alerts"""
        
        while self.monitoring_active:
            try:
                # Check for emergency conditions
                await self.check_emergency_conditions()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error monitoring alerts: {e}")
                await asyncio.sleep(60)
    
    async def check_emergency_conditions(self):
        """Check for emergency conditions that need immediate attention"""
        
        try:
            status = self.bot_status
            
            # Large loss alert
            if status.get('profit_loss', 0.0) < -30.0:  # $30 loss
                await self.send_message(
                    f"🚨 EMERGENCY: Large loss detected! P&L: ${status['profit_loss']:+.2f}"
                )
            
            # Bot offline for too long
            if not status.get('running', False):
                offline_time = datetime.now() - status.get('timestamp', datetime.now())
                if offline_time > timedelta(minutes=10):
                    await self.send_message(
                        f"🚨 EMERGENCY: Bot offline for {offline_time.seconds // 60} minutes!"
                    )
            
            # Too many alerts
            if status.get('alerts_count', 0) > 10:
                await self.send_message(
                    f"⚠️ HIGH ALERT COUNT: {status['alerts_count']} active alerts"
                )
                
        except Exception as e:
            self.logger.error(f"Error checking emergency conditions: {e}")


async def main():
    """Main function"""
    
    # Get Telegram credentials from environment
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("❌ Telegram credentials not found in environment variables")
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return
    
    monitor = TelegramMonitor(bot_token, chat_id)
    
    try:
        print("📱 Telegram monitoring ready (demo mode)")
    except KeyboardInterrupt:
        print("\n📱 Telegram monitoring stopped")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 