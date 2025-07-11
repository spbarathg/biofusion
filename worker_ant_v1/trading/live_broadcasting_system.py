"""
LIVE BROADCASTING SYSTEM - SMART APE BEHAVIOR #15
=================================================

Real-time broadcasting of system stats and alerts:
- Live trade notifications
- Wallet performance updates  
- Risk level changes
- Profit/loss tracking
- Swarm health monitoring

Broadcasts to multiple channels: Console, Local Logs, API webhooks
"""

import asyncio
import json
import time
import websockets
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque, defaultdict

# Core dependencies (required)
import aiohttp

# Social media integrations REMOVED for pure on-chain operation
# NO DISCORD, TELEGRAM, OR TWITTER INTEGRATION

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Validate core dependencies
try:
    import aiohttp
    WEB_DEPS_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        f"Critical broadcasting dependencies missing: {e}. "
        "Install with: pip install aiohttp"
    ) from e

from worker_ant_v1.utils.logger import setup_logger

CONSOLE = "console"
FILE_LOG = "file_log"
API_WEBHOOK = "api_webhook"

class BroadcastChannel(Enum):
    CONSOLE = "console"
    WEBHOOK = "webhook"
    WEBSOCKET = "websocket"
    FILE_LOG = "file_log"

class AlertLevel(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class LiveAlert:
    """Live alert message"""
    
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    
    # Alert metadata
    component: str = "system"
    data: Dict[str, Any] = field(default_factory=dict)
    broadcast_channels: Set[BroadcastChannel] = field(default_factory=set)
    
    # Status tracking
    broadcasted: bool = False
    broadcast_results: Dict[BroadcastChannel, bool] = field(default_factory=dict)

@dataclass
class TradingUpdate:
    """Live trading update"""
    
    update_id: str
    wallet_id: str
    action: str  # "buy", "sell", "fake_buy", "stop_loss"
    token_symbol: str
    token_address: str
    amount_sol: float
    price: float
    timestamp: datetime
    
    # Trade metadata
    profit_loss: Optional[float] = None
    success: bool = True
    reason: str = ""
    confidence: float = 0.0

@dataclass
class SystemStats:
    """Live system statistics"""
    
    timestamp: datetime
    
    # Trading metrics
    total_trades_today: int = 0
    successful_trades_today: int = 0
    total_profit_today: float = 0.0
    win_rate_today: float = 0.0
    
    # Wallet metrics
    active_wallets: int = 0
    top_performer_id: str = ""
    worst_performer_id: str = ""
    average_fitness: float = 0.0
    
    # Risk metrics
    current_risk_level: str = "LOW"
    active_positions: int = 0
    vault_balance: float = 0.0
    
    # Evolution metrics
    current_generation: int = 1
    last_evolution: Optional[datetime] = None
    next_evolution: Optional[datetime] = None
    
    # System health
    uptime_hours: float = 0.0
    system_health: str = "HEALTHY"
    kill_switch_status: str = "ARMED"

class LiveBroadcastingSystem:
    """Real-time broadcasting system for Smart Ape Swarm"""
    
    def __init__(self):
        self.logger = setup_logger("LiveBroadcasting")
        
        # Broadcasting configuration
        self.enabled_channels = {BroadcastChannel.CONSOLE}  # Default to console only
        self.broadcast_interval = 60  # seconds
        self.alert_queue = asyncio.Queue(maxsize=1000)
        self.update_queue = asyncio.Queue(maxsize=1000)
        
        # Channel configurations - ONLY local/webhook channels (NO social media)
        self.webhook_urls = []
        
        # WebSocket server for real-time updates
        self.websocket_server = None
        self.websocket_clients = set()
        self.websocket_port = 8765
        
        # Statistics tracking
        self.current_stats = SystemStats(timestamp=datetime.utcnow())
        self.stats_history = deque(maxlen=1440)  # 24 hours of minute-by-minute stats
        
        # Alert tracking
        self.recent_alerts = deque(maxlen=100)
        self.alert_rate_limiter = defaultdict(lambda: deque(maxlen=10))
        
        # Broadcasting state
        self.broadcasting_active = False
        self.last_stats_broadcast = datetime.utcnow()
        
    async def initialize(self, config: Dict[str, Any] = None):
        """Initialize the broadcasting system"""
        
        self.logger.info("📡 Initializing Live Broadcasting System")
        
        if config:
            await self._configure_channels(config)
        
        # Start broadcasting tasks
        await self._start_broadcasting_tasks()
        
        # Start WebSocket server if enabled
        if BroadcastChannel.WEBSOCKET in self.enabled_channels:
            await self._start_websocket_server()
        
        self.broadcasting_active = True
        
        # Send startup message
        await self.broadcast_alert(
            level=AlertLevel.SUCCESS,
            title="🧬 Smart Ape Swarm Online",
            message="AI-only trading system initialized. All systems operational.",
            component="system",
            channels={BroadcastChannel.CONSOLE, BroadcastChannel.FILE_LOG}
        )
        
        self.logger.info("✅ Live Broadcasting System ready")
    
    async def _configure_channels(self, config: Dict[str, Any]):
        """Configure broadcasting channels - PURE ON-CHAIN ONLY"""
        
        # NO SOCIAL MEDIA INTEGRATION - Pure on-chain + AI only
        
        # Webhooks
        if config.get('webhook_urls'):
            self.webhook_urls = config['webhook_urls']
            self.enabled_channels.add(BroadcastChannel.WEBHOOK)
            self.logger.info(f"🔗 Webhook broadcasting enabled ({len(self.webhook_urls)} endpoints)")
        
        # WebSocket
        if config.get('enable_websocket', False):
            self.websocket_port = config.get('websocket_port', 8765)
            self.enabled_channels.add(BroadcastChannel.WEBSOCKET)
            self.logger.info(f"🌐 WebSocket broadcasting enabled (port {self.websocket_port})")
        
        # File logging always enabled
        self.enabled_channels.add(BroadcastChannel.FILE_LOG)
        self.logger.info("📝 File logging enabled")
    
    async def _start_broadcasting_tasks(self):
        """Start background broadcasting tasks"""
        
        # Alert processing task
        asyncio.create_task(self._alert_processor())
        
        # Trading update processor
        asyncio.create_task(self._update_processor())
        
        # Periodic stats broadcaster
        asyncio.create_task(self._stats_broadcaster())
        
        self.logger.info("🚀 Broadcasting tasks started")
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        
        try:
            self.websocket_server = await websockets.serve(
                self._websocket_handler,
                "localhost",
                self.websocket_port
            )
            self.logger.info(f"🌐 WebSocket server started on port {self.websocket_port}")
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
    
    async def _websocket_handler(self, websocket, path):
        """Handle WebSocket connections"""
        
        self.websocket_clients.add(websocket)
        self.logger.info(f"🔌 WebSocket client connected ({len(self.websocket_clients)} total)")
        
        try:
            # Send current stats to new client
            await self._send_websocket_message(websocket, {
                'type': 'stats',
                'data': asdict(self.current_stats)
            })
            
            # Keep connection alive
            await websocket.wait_closed()
            
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"WebSocket handler error: {e}")
        finally:
            self.websocket_clients.discard(websocket)
            self.logger.info(f"🔌 WebSocket client disconnected ({len(self.websocket_clients)} remaining)")
    
    async def broadcast_alert(self, level: AlertLevel, title: str, message: str,
                             component: str = "system", data: Dict[str, Any] = None,
                             channels: Set[BroadcastChannel] = None) -> str:
        """Broadcast an alert to configured channels"""
        
        alert_id = f"alert_{int(time.time() * 1000)}"
        
        alert = LiveAlert(
            alert_id=alert_id,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            component=component,
            data=data or {},
            broadcast_channels=channels or self.enabled_channels
        )
        
        # Add to queue for processing
        try:
            await self.alert_queue.put(alert)
        except asyncio.QueueFull:
            self.logger.warning("Alert queue full, dropping alert")
        
        return alert_id
    
    # TWITTER INTEGRATION COMPLETELY REMOVED - PURE ON-CHAIN + AI ONLY
    
    async def broadcast_trading_update(self, wallet_id: str, action: str, token_symbol: str,
                                      token_address: str, amount_sol: float, price: float,
                                      profit_loss: float = None, success: bool = True,
                                      reason: str = "", confidence: float = 0.0) -> str:
        """Broadcast a trading update"""
        
        update_id = f"trade_{int(time.time() * 1000)}"
        
        update = TradingUpdate(
            update_id=update_id,
            wallet_id=wallet_id,
            action=action,
            token_symbol=token_symbol,
            token_address=token_address,
            amount_sol=amount_sol,
            price=price,
            timestamp=datetime.utcnow(),
            profit_loss=profit_loss,
            success=success,
            reason=reason,
            confidence=confidence
        )
        
        # Add to queue for processing
        try:
            await self.update_queue.put(update)
        except asyncio.QueueFull:
            self.logger.warning("Update queue full, dropping update")
        
        return update_id
    
    async def update_system_stats(self, stats_update: Dict[str, Any]):
        """Update current system statistics"""
        
        # Update current stats
        for key, value in stats_update.items():
            if hasattr(self.current_stats, key):
                setattr(self.current_stats, key, value)
        
        self.current_stats.timestamp = datetime.utcnow()
        
        # Add to history
        self.stats_history.append(self.current_stats)
    
    async def _alert_processor(self):
        """Process alerts from the queue"""
        
        while self.broadcasting_active:
            try:
                # Get alert from queue
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=5.0)
                
                # Rate limiting check
                if self._is_rate_limited(alert):
                    continue
                
                # Broadcast to configured channels
                await self._broadcast_alert_to_channels(alert)
                
                # Add to recent alerts
                self.recent_alerts.append(alert)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Alert processor error: {e}")
    
    async def _update_processor(self):
        """Process trading updates from the queue"""
        
        while self.broadcasting_active:
            try:
                # Get update from queue
                update = await asyncio.wait_for(self.update_queue.get(), timeout=5.0)
                
                # Broadcast update
                await self._broadcast_trading_update(update)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Update processor error: {e}")
    
    async def _stats_broadcaster(self):
        """Broadcast periodic system statistics"""
        
        while self.broadcasting_active:
            try:
                # Check if it's time to broadcast stats
                if (datetime.utcnow() - self.last_stats_broadcast).total_seconds() >= self.broadcast_interval:
                    await self._broadcast_system_stats()
                    self.last_stats_broadcast = datetime.utcnow()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Stats broadcaster error: {e}")
    
    async def _broadcast_alert_to_channels(self, alert: LiveAlert):
        """Broadcast alert to all configured channels"""
        
        for channel in alert.broadcast_channels:
            try:
                if channel == BroadcastChannel.CONSOLE:
                    await self._broadcast_to_console(alert)
                            # SOCIAL MEDIA CHANNELS REMOVED - PURE ON-CHAIN + AI ONLY
                elif channel == BroadcastChannel.WEBHOOK:
                    await self._broadcast_to_webhooks(alert)
                elif channel == BroadcastChannel.WEBSOCKET:
                    await self._broadcast_to_websockets(alert)
                
                alert.broadcast_results[channel] = True
                
            except Exception as e:
                self.logger.error(f"Failed to broadcast to {channel.value}: {e}")
                alert.broadcast_results[channel] = False
        
        alert.broadcasted = True
    
    async def _broadcast_to_console(self, alert: LiveAlert):
        """Broadcast alert to console"""
        
        # Color-coded console output
        colors = {
            AlertLevel.INFO: "\033[94m",      # Blue
            AlertLevel.SUCCESS: "\033[92m",   # Green
            AlertLevel.WARNING: "\033[93m",   # Yellow
            AlertLevel.ERROR: "\033[91m",     # Red
            AlertLevel.CRITICAL: "\033[95m"   # Magenta
        }
        reset_color = "\033[0m"
        
        color = colors.get(alert.level, "")
        timestamp = alert.timestamp.strftime("%H:%M:%S")
        
        print(f"{color}[{timestamp}] {alert.level.value.upper()}: {alert.title}")
        print(f"  {alert.message}{reset_color}")
        
        if alert.data:
            for key, value in alert.data.items():
                print(f"  {key}: {value}")
    
    # SOCIAL MEDIA BROADCASTING REMOVED - PURE ON-CHAIN + AI ONLY
    
    async def _send_console_alert(self, alert: LiveAlert):
        """Send alert to console"""
        emoji_map = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.SUCCESS: "✅", 
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }
        
        emoji = emoji_map.get(alert.level, "📢")
        timestamp = alert.timestamp.strftime("%H:%M:%S")
        
        print(f"{emoji} [{timestamp}] {alert.title}")
        print(f"    {alert.message}")
        if alert.data:
            for key, value in alert.data.items():
                print(f"    {key}: {value}")
        print()
    
    async def _send_file_alert(self, alert: LiveAlert):
        """Send alert to log file"""
        log_message = f"[{alert.timestamp.isoformat()}] {alert.level.value.upper()}: {alert.title} - {alert.message}"
        if alert.data:
            log_message += f" | Data: {alert.data}"
        
        # Write to log file
        try:
            with open("logs/alerts.log", "a") as f:
                f.write(log_message + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write to log file: {e}")
    
    async def _send_webhook_alert(self, alert: LiveAlert):
        """Send alert to configured webhooks"""
        if not self.webhook_urls:
            return
            
        payload = {
            "timestamp": alert.timestamp.isoformat(),
            "level": alert.level.value,
            "title": alert.title,
            "message": alert.message,
            "component": alert.component,
            "data": alert.data
        }
        
        for webhook_url in self.webhook_urls:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(webhook_url, json=payload) as response:
                        if response.status not in [200, 204]:
                            self.logger.warning(f"Webhook failed: {response.status}")
            except Exception as e:
                self.logger.error(f"Webhook error: {e}")
    
    async def _send_websocket_alert(self, alert: LiveAlert):
        """Send alert to WebSocket clients"""
        if not self.websocket_clients:
            return
            
        message = {
            "type": "alert",
            "timestamp": alert.timestamp.isoformat(),
            "level": alert.level.value,
            "title": alert.title,
            "message": alert.message,
            "component": alert.component,
            "data": alert.data
        }
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self.websocket_clients:
            try:
                await client.send(json.dumps(message))
            except Exception:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    async def _broadcast_to_webhooks(self, alert: LiveAlert):
        """Broadcast alert to custom webhooks"""
        
        payload = {
            "alert_id": alert.alert_id,
            "level": alert.level.value,
            "title": alert.title,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "component": alert.component,
            "data": alert.data
        }
        
        async with aiohttp.ClientSession() as session:
            for webhook_url in self.webhook_urls:
                try:
                    async with session.post(webhook_url, json=payload) as response:
                        if response.status not in [200, 201, 204]:
                            self.logger.warning(f"Webhook {webhook_url} failed: {response.status}")
                except Exception as e:
                    self.logger.error(f"Webhook {webhook_url} error: {e}")
    
    async def _broadcast_to_websockets(self, alert: LiveAlert):
        """Broadcast alert to WebSocket clients"""
        
        if not self.websocket_clients:
            return
        
        message = {
            "type": "alert",
            "data": {
                "alert_id": alert.alert_id,
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "component": alert.component,
                "data": alert.data
            }
        }
        
        # Broadcast to all connected clients
        disconnected = set()
        for websocket in self.websocket_clients:
            try:
                await self._send_websocket_message(websocket, message)
            except:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected
    
    async def _send_websocket_message(self, websocket, message: Dict[str, Any]):
        """Send message to WebSocket client"""
        await websocket.send(json.dumps(message))
    
    async def _broadcast_trading_update(self, update: TradingUpdate):
        """Broadcast trading update"""
        
        # Create alert for trading update
        emoji = "🟢" if update.success else "🔴"
        action_emoji = {"buy": "🛒", "sell": "💰", "fake_buy": "🎭", "stop_loss": "🛑"}.get(update.action, "📊")
        
        title = f"{emoji} {action_emoji} {update.action.upper()} {update.token_symbol}"
        
        message = f"Wallet {update.wallet_id[-8:]} {update.action} {update.amount_sol:.4f} SOL"
        if update.profit_loss is not None:
            pnl_emoji = "📈" if update.profit_loss > 0 else "📉"
            message += f"\n{pnl_emoji} P&L: {update.profit_loss:.4f} SOL"
        
        data = {
            "wallet_id": update.wallet_id,
            "token_address": update.token_address,
            "price": update.price,
            "confidence": f"{update.confidence:.1%}"
        }
        
        if update.reason:
            data["reason"] = update.reason
        
        await self.broadcast_alert(
            level=AlertLevel.SUCCESS if update.success else AlertLevel.WARNING,
            title=title,
            message=message,
            component="trading",
            data=data,
            channels={BroadcastChannel.CONSOLE, BroadcastChannel.FILE_LOG, BroadcastChannel.WEBSOCKET}
        )
    
    async def _broadcast_system_stats(self):
        """Broadcast periodic system statistics"""
        
        stats = self.current_stats
        
        # Create comprehensive stats message
        title = "📊 Smart Ape Swarm Status"
        
        message = f"""
🎯 **Trading Performance**
• Trades today: {stats.total_trades_today} ({stats.win_rate_today:.1%} win rate)
• Profit today: {stats.total_profit_today:.4f} SOL
• Active positions: {stats.active_positions}

🤖 **Wallet Evolution**
• Generation: {stats.current_generation}
• Active wallets: {stats.active_wallets}
• Average fitness: {stats.average_fitness:.2f}

🛡️ **System Health**
• Risk level: {stats.current_risk_level}
• Vault balance: {stats.vault_balance:.4f} SOL
• Uptime: {stats.uptime_hours:.1f} hours
• Status: {stats.system_health}
        """.strip()
        
        data = {
            "kill_switch": stats.kill_switch_status,
            "next_evolution": stats.next_evolution.isoformat() if stats.next_evolution else "Unknown"
        }
        
        await self.broadcast_alert(
            level=AlertLevel.INFO,
            title=title,
            message=message,
            component="monitoring",
            data=data,
            channels={BroadcastChannel.CONSOLE, BroadcastChannel.WEBSOCKET}
        )
    
    def _is_rate_limited(self, alert: LiveAlert) -> bool:
        """Check if alert should be rate limited"""
        
        rate_key = f"{alert.component}_{alert.level.value}"
        current_time = time.time()
        
        # Clean old entries (older than 5 minutes)
        while (self.alert_rate_limiter[rate_key] and 
               current_time - self.alert_rate_limiter[rate_key][0] > 300):
            self.alert_rate_limiter[rate_key].popleft()
        
        # Check rate limit (max 10 alerts per 5 minutes per component/level)
        if len(self.alert_rate_limiter[rate_key]) >= 10:
            return True
        
        # Add current alert
        self.alert_rate_limiter[rate_key].append(current_time)
        return False
    
    async def shutdown(self):
        """Shutdown the broadcasting system"""
        
        self.logger.info("🔄 Shutting down Live Broadcasting System")
        
        # Send shutdown message
        await self.broadcast_alert(
            level=AlertLevel.WARNING,
            title="🧬 Smart Ape Swarm Shutting Down",
            message="Live broadcasting system is shutting down. All systems going offline.",
            component="system"
        )
        
        self.broadcasting_active = False
        
        # Close WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        self.logger.info("✅ Live Broadcasting System shut down")

# Factory function
async def create_live_broadcasting_system(config: Dict[str, Any] = None) -> LiveBroadcastingSystem:
    """Create and initialize the live broadcasting system"""
    broadcasting_system = LiveBroadcastingSystem()
    await broadcasting_system.initialize(config)
    return broadcasting_system 