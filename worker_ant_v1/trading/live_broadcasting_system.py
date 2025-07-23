"""
LIVE BROADCASTING SYSTEM - REAL-TIME TRADING BROADCASTS
=====================================================

Real-time broadcasting system for trading activities, system status,
and performance metrics to external platforms and monitoring systems.
"""

import asyncio
import json
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from worker_ant_v1.utils.logger import setup_logger

class BroadcastType(Enum):
    """Types of broadcasts"""
    TRADE_EXECUTION = "trade_execution"
    SYSTEM_STATUS = "system_status"
    PERFORMANCE_UPDATE = "performance_update"
    ALERT = "alert"
    OPPORTUNITY = "opportunity"
    ERROR = "error"

class BroadcastChannel(Enum):
    """Broadcast channels"""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    TWITTER = "twitter"
    WEBHOOK = "webhook"
    LOG_FILE = "log_file"

@dataclass
class BroadcastMessage:
    """Broadcast message structure"""
    message_id: str
    broadcast_type: BroadcastType
    channel: BroadcastChannel
    title: str
    content: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    metadata: Dict[str, Any] = field(default_factory=dict)

class LiveBroadcastingSystem:
    """Real-time broadcasting system for trading activities"""
    
    def __init__(self):
        self.logger = setup_logger("LiveBroadcastingSystem")
        
        # Configuration
        self.config = {
            'enabled_channels': [BroadcastChannel.LOG_FILE],  # Default to log file only
            'broadcast_interval_seconds': 30,
            'max_message_length': 1000,
            'rate_limit_messages_per_minute': 10,
            'webhook_url': None,
            'telegram_bot_token': None,
            'telegram_chat_id': None,
            'discord_webhook_url': None
        }
        
        # Message queue and storage
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.broadcast_history: List[BroadcastMessage] = []
        self.rate_limit_tracker: Dict[BroadcastChannel, List[datetime]] = {}
        
        # System state
        self.initialized = False
        self.broadcasting_active = False
        
        # HTTP session for webhooks
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self) -> bool:
        """Initialize the broadcasting system"""
        try:
            self.logger.info("📡 Initializing Live Broadcasting System...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Initialize rate limit trackers
            for channel in BroadcastChannel:
                self.rate_limit_tracker[channel] = []
            
            # Start background tasks
            asyncio.create_task(self._broadcast_loop())
            asyncio.create_task(self._cleanup_loop())
            
            self.initialized = True
            self.broadcasting_active = True
            self.logger.info("✅ Live Broadcasting System initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize broadcasting system: {e}")
            return False
    
    async def broadcast_trade_execution(self, trade_data: Dict[str, Any]):
        """Broadcast trade execution"""
        try:
            message = BroadcastMessage(
                message_id=f"trade_{trade_data.get('trade_id', 'unknown')}_{int(datetime.now().timestamp())}",
                broadcast_type=BroadcastType.TRADE_EXECUTION,
                channel=BroadcastChannel.LOG_FILE,  # Will be overridden by broadcast method
                title="Trade Executed",
                content=self._format_trade_message(trade_data),
                data=trade_data,
                timestamp=datetime.now(),
                priority=2 if trade_data.get('success', False) else 3
            )
            
            await self._queue_message(message)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting trade execution: {e}")
    
    async def broadcast_system_status(self, status_data: Dict[str, Any]):
        """Broadcast system status update"""
        try:
            message = BroadcastMessage(
                message_id=f"status_{int(datetime.now().timestamp())}",
                broadcast_type=BroadcastType.SYSTEM_STATUS,
                channel=BroadcastChannel.LOG_FILE,
                title="System Status Update",
                content=self._format_status_message(status_data),
                data=status_data,
                timestamp=datetime.now(),
                priority=1
            )
            
            await self._queue_message(message)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting system status: {e}")
    
    async def broadcast_performance_update(self, performance_data: Dict[str, Any]):
        """Broadcast performance update"""
        try:
            message = BroadcastMessage(
                message_id=f"perf_{int(datetime.now().timestamp())}",
                broadcast_type=BroadcastType.PERFORMANCE_UPDATE,
                channel=BroadcastChannel.LOG_FILE,
                title="Performance Update",
                content=self._format_performance_message(performance_data),
                data=performance_data,
                timestamp=datetime.now(),
                priority=2
            )
            
            await self._queue_message(message)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting performance update: {e}")
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """Broadcast alert"""
        try:
            message = BroadcastMessage(
                message_id=f"alert_{alert_data.get('alert_id', 'unknown')}_{int(datetime.now().timestamp())}",
                broadcast_type=BroadcastType.ALERT,
                channel=BroadcastChannel.LOG_FILE,
                title=alert_data.get('title', 'Alert'),
                content=self._format_alert_message(alert_data),
                data=alert_data,
                timestamp=datetime.now(),
                priority=alert_data.get('priority', 3)
            )
            
            await self._queue_message(message)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting alert: {e}")
    
    async def broadcast_opportunity(self, opportunity_data: Dict[str, Any]):
        """Broadcast trading opportunity"""
        try:
            message = BroadcastMessage(
                message_id=f"opp_{opportunity_data.get('token_address', 'unknown')}_{int(datetime.now().timestamp())}",
                broadcast_type=BroadcastType.OPPORTUNITY,
                channel=BroadcastChannel.LOG_FILE,
                title="Trading Opportunity",
                content=self._format_opportunity_message(opportunity_data),
                data=opportunity_data,
                timestamp=datetime.now(),
                priority=3
            )
            
            await self._queue_message(message)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting opportunity: {e}")
    
    async def broadcast_error(self, error_data: Dict[str, Any]):
        """Broadcast error"""
        try:
            message = BroadcastMessage(
                message_id=f"error_{int(datetime.now().timestamp())}",
                broadcast_type=BroadcastType.ERROR,
                channel=BroadcastChannel.LOG_FILE,
                title="System Error",
                content=self._format_error_message(error_data),
                data=error_data,
                timestamp=datetime.now(),
                priority=4
            )
            
            await self._queue_message(message)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting error: {e}")
    
    async def _queue_message(self, message: BroadcastMessage):
        """Queue a message for broadcasting"""
        try:
            await self.message_queue.put(message)
        except Exception as e:
            self.logger.error(f"Error queuing message: {e}")
    
    async def _broadcast_loop(self):
        """Main broadcasting loop"""
        while self.broadcasting_active:
            try:
                # Process messages from queue
                while not self.message_queue.empty():
                    message = await self.message_queue.get()
                    await self._broadcast_message(message)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Broadcast loop error: {e}")
                await asyncio.sleep(5)
    
    async def _broadcast_message(self, message: BroadcastMessage):
        """Broadcast a message to all enabled channels"""
        try:
            # Check rate limits
            if not self._check_rate_limit(message.channel):
                self.logger.warning(f"Rate limit exceeded for {message.channel.value}")
                return
            
            # Broadcast to all enabled channels
            for channel in self.config['enabled_channels']:
                message.channel = channel
                await self._send_to_channel(message)
            
            # Store in history
            self.broadcast_history.append(message)
            
            # Keep only last 1000 messages
            if len(self.broadcast_history) > 1000:
                self.broadcast_history = self.broadcast_history[-1000:]
            
        except Exception as e:
            self.logger.error(f"Error broadcasting message: {e}")
    
    async def _send_to_channel(self, message: BroadcastMessage):
        """Send message to specific channel"""
        try:
            if message.channel == BroadcastChannel.LOG_FILE:
                await self._send_to_log_file(message)
            elif message.channel == BroadcastChannel.WEBHOOK:
                await self._send_to_webhook(message)
            elif message.channel == BroadcastChannel.TELEGRAM:
                await self._send_to_telegram(message)
            elif message.channel == BroadcastChannel.DISCORD:
                await self._send_to_discord(message)
            elif message.channel == BroadcastChannel.TWITTER:
                await self._send_to_twitter(message)
            
        except Exception as e:
            self.logger.error(f"Error sending to {message.channel.value}: {e}")
    
    async def _send_to_log_file(self, message: BroadcastMessage):
        """Send message to log file"""
        try:
            log_entry = f"[BROADCAST] {message.timestamp.isoformat()} - {message.title}: {message.content}"
            
            if message.priority >= 3:
                self.logger.warning(log_entry)
            elif message.priority >= 2:
                self.logger.info(log_entry)
            else:
                self.logger.debug(log_entry)
            
        except Exception as e:
            self.logger.error(f"Error sending to log file: {e}")
    
    async def _send_to_webhook(self, message: BroadcastMessage):
        """Send message to webhook"""
        try:
            if not self.config['webhook_url'] or not self.session:
                return
            
            payload = {
                'message_id': message.message_id,
                'type': message.broadcast_type.value,
                'title': message.title,
                'content': message.content,
                'timestamp': message.timestamp.isoformat(),
                'priority': message.priority,
                'data': message.data
            }
            
            async with self.session.post(
                self.config['webhook_url'],
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    self.logger.warning(f"Webhook failed: {response.status}")
            
        except Exception as e:
            self.logger.error(f"Error sending to webhook: {e}")
    
    async def _send_to_telegram(self, message: BroadcastMessage):
        """Send message to Telegram"""
        try:
            if not self.config['telegram_bot_token'] or not self.config['telegram_chat_id'] or not self.session:
                return
            
            telegram_text = f"*{message.title}*\n\n{message.content}"
            
            url = f"https://api.telegram.org/bot{self.config['telegram_bot_token']}/sendMessage"
            payload = {
                'chat_id': self.config['telegram_chat_id'],
                'text': telegram_text,
                'parse_mode': 'Markdown'
            }
            
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    self.logger.warning(f"Telegram failed: {response.status}")
            
        except Exception as e:
            self.logger.error(f"Error sending to Telegram: {e}")
    
    async def _send_to_discord(self, message: BroadcastMessage):
        """Send message to Discord"""
        try:
            if not self.config['discord_webhook_url'] or not self.session:
                return
            
            # Create Discord embed
            embed = {
                'title': message.title,
                'description': message.content,
                'color': self._get_discord_color(message.priority),
                'timestamp': message.timestamp.isoformat(),
                'fields': []
            }
            
            # Add data fields
            for key, value in message.data.items():
                if isinstance(value, (str, int, float, bool)):
                    embed['fields'].append({
                        'name': key.replace('_', ' ').title(),
                        'value': str(value),
                        'inline': True
                    })
            
            payload = {'embeds': [embed]}
            
            async with self.session.post(
                self.config['discord_webhook_url'],
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 204:
                    self.logger.warning(f"Discord failed: {response.status}")
            
        except Exception as e:
            self.logger.error(f"Error sending to Discord: {e}")
    
    async def _send_to_twitter(self, message: BroadcastMessage):
        """Send message to Twitter"""
        try:
            # Placeholder - implement with Twitter API
            # This would require Twitter API credentials and proper authentication
            self.logger.info(f"Twitter broadcast (placeholder): {message.title}")
            
        except Exception as e:
            self.logger.error(f"Error sending to Twitter: {e}")
    
    def _get_discord_color(self, priority: int) -> int:
        """Get Discord embed color based on priority"""
        colors = {
            1: 0x00ff00,  # Green for low priority
            2: 0xffff00,  # Yellow for medium priority
            3: 0xff8800,  # Orange for high priority
            4: 0xff0000   # Red for critical priority
        }
        return colors.get(priority, 0x808080)  # Gray as default
    
    def _check_rate_limit(self, channel: BroadcastChannel) -> bool:
        """Check rate limit for channel"""
        try:
            current_time = datetime.now()
            channel_times = self.rate_limit_tracker[channel]
            
            # Remove old timestamps (older than 1 minute)
            channel_times[:] = [t for t in channel_times if (current_time - t).total_seconds() < 60]
            
            # Check if we're under the limit
            if len(channel_times) < self.config['rate_limit_messages_per_minute']:
                channel_times.append(current_time)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {e}")
            return True  # Allow if check fails
    
    def _format_trade_message(self, trade_data: Dict[str, Any]) -> str:
        """Format trade execution message"""
        try:
            success = trade_data.get('success', False)
            token_address = trade_data.get('token_address', 'Unknown')
            profit = trade_data.get('profit_sol', 0.0)
            wallet_id = trade_data.get('wallet_id', 'Unknown')
            
            if success:
                return f"✅ Trade executed successfully\nToken: {token_address[:8]}...\nProfit: {profit:.6f} SOL\nWallet: {wallet_id}"
            else:
                error = trade_data.get('error', 'Unknown error')
                return f"❌ Trade failed\nToken: {token_address[:8]}...\nError: {error}\nWallet: {wallet_id}"
            
        except Exception as e:
            self.logger.error(f"Error formatting trade message: {e}")
            return "Trade execution broadcast"
    
    def _format_status_message(self, status_data: Dict[str, Any]) -> str:
        """Format system status message"""
        try:
            status = status_data.get('status', 'Unknown')
            active_wallets = status_data.get('active_wallets', 0)
            total_profit = status_data.get('total_profit', 0.0)
            
            return f"System Status: {status}\nActive Wallets: {active_wallets}\nTotal Profit: {total_profit:.6f} SOL"
            
        except Exception as e:
            self.logger.error(f"Error formatting status message: {e}")
            return "System status update"
    
    def _format_performance_message(self, performance_data: Dict[str, Any]) -> str:
        """Format performance update message"""
        try:
            win_rate = performance_data.get('win_rate', 0.0)
            total_trades = performance_data.get('total_trades', 0)
            avg_profit = performance_data.get('avg_profit_per_trade', 0.0)
            
            return f"Performance Update\nWin Rate: {win_rate:.1%}\nTotal Trades: {total_trades}\nAvg Profit: {avg_profit:.6f} SOL"
            
        except Exception as e:
            self.logger.error(f"Error formatting performance message: {e}")
            return "Performance update"
    
    def _format_alert_message(self, alert_data: Dict[str, Any]) -> str:
        """Format alert message"""
        try:
            severity = alert_data.get('severity', 'Unknown')
            message = alert_data.get('message', 'No message')
            component = alert_data.get('component', 'Unknown')
            
            return f"🚨 Alert ({severity})\nComponent: {component}\nMessage: {message}"
            
        except Exception as e:
            self.logger.error(f"Error formatting alert message: {e}")
            return "Alert broadcast"
    
    def _format_opportunity_message(self, opportunity_data: Dict[str, Any]) -> str:
        """Format opportunity message"""
        try:
            token_address = opportunity_data.get('token_address', 'Unknown')
            confidence = opportunity_data.get('confidence_score', 0.0)
            risk = opportunity_data.get('risk_score', 1.0)
            recommendation = opportunity_data.get('recommendation', 'Unknown')
            
            return f"🎯 Trading Opportunity\nToken: {token_address[:8]}...\nConfidence: {confidence:.1%}\nRisk: {risk:.1%}\nRecommendation: {recommendation}"
            
        except Exception as e:
            self.logger.error(f"Error formatting opportunity message: {e}")
            return "Trading opportunity"
    
    def _format_error_message(self, error_data: Dict[str, Any]) -> str:
        """Format error message"""
        try:
            error_type = error_data.get('error_type', 'Unknown')
            message = error_data.get('message', 'No message')
            component = error_data.get('component', 'Unknown')
            
            return f"💥 System Error\nType: {error_type}\nComponent: {component}\nMessage: {message}"
            
        except Exception as e:
            self.logger.error(f"Error formatting error message: {e}")
            return "System error"
    
    async def _cleanup_loop(self):
        """Cleanup old broadcast history"""
        while self.broadcasting_active:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Remove old messages (older than 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.broadcast_history[:] = [
                    msg for msg in self.broadcast_history 
                    if msg.timestamp > cutoff_time
                ]
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    def configure_channels(self, channels: List[BroadcastChannel]):
        """Configure enabled broadcast channels"""
        try:
            self.config['enabled_channels'] = channels
            self.logger.info(f"Broadcast channels configured: {[c.value for c in channels]}")
        except Exception as e:
            self.logger.error(f"Error configuring channels: {e}")
    
    def configure_webhook(self, webhook_url: str):
        """Configure webhook URL"""
        try:
            self.config['webhook_url'] = webhook_url
            self.logger.info("Webhook URL configured")
        except Exception as e:
            self.logger.error(f"Error configuring webhook: {e}")
    
    def configure_telegram(self, bot_token: str, chat_id: str):
        """Configure Telegram bot"""
        try:
            self.config['telegram_bot_token'] = bot_token
            self.config['telegram_chat_id'] = chat_id
            self.logger.info("Telegram bot configured")
        except Exception as e:
            self.logger.error(f"Error configuring Telegram: {e}")
    
    def configure_discord(self, webhook_url: str):
        """Configure Discord webhook"""
        try:
            self.config['discord_webhook_url'] = webhook_url
            self.logger.info("Discord webhook configured")
        except Exception as e:
            self.logger.error(f"Error configuring Discord: {e}")
    
    def get_broadcast_status(self) -> Dict[str, Any]:
        """Get broadcasting system status"""
        try:
            return {
                'initialized': self.initialized,
                'broadcasting_active': self.broadcasting_active,
                'enabled_channels': [c.value for c in self.config['enabled_channels']],
                'queue_size': self.message_queue.qsize(),
                'history_size': len(self.broadcast_history),
                'webhook_configured': bool(self.config['webhook_url']),
                'telegram_configured': bool(self.config['telegram_bot_token']),
                'discord_configured': bool(self.config['discord_webhook_url'])
            }
            
        except Exception as e:
            self.logger.error(f"Error getting broadcast status: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown the broadcasting system"""
        try:
            self.logger.info("🛑 Shutting down broadcasting system...")
            self.broadcasting_active = False
            
            if self.session:
                await self.session.close()
            
            self.logger.info("✅ Broadcasting system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Global instance
_broadcasting_system = None

async def get_broadcasting_system() -> LiveBroadcastingSystem:
    """Get global broadcasting system instance"""
    global _broadcasting_system
    if _broadcasting_system is None:
        _broadcasting_system = LiveBroadcastingSystem()
        await _broadcasting_system.initialize()
    return _broadcasting_system 