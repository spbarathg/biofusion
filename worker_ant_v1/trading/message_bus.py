"""
NATS MESSAGE BUS - HIGH-PERFORMANCE INTER-SYSTEM COMMUNICATION
============================================================

NATS-based message bus system replacing asyncio.Queue for inter-component communication.
Provides high-performance, distributed messaging for the trading colony architecture.

Features:
- Real-time message routing between components
- Hierarchical subject-based routing (colony.*, swarm.*, squad.*, wallet.*)
- Message persistence and replay capabilities
- Load balancing and failover
- Performance monitoring and metrics
- Dead letter queue handling
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Set
from enum import Enum
import nats
from nats.aio.client import Client as NATS
from nats.aio.subscription import Subscription
import msgpack
import gzip

from worker_ant_v1.utils.logger import setup_logger


class MessagePriority(Enum):
    """Message priority levels"""
    EMERGENCY = "emergency"    # Kill switch, critical errors
    HIGH = "high"             # Trading signals, rug detection
    MEDIUM = "medium"         # Performance data, status updates
    LOW = "low"              # Logging, analytics


class MessageType(Enum):
    """Message types for routing and handling"""
    # Colony-level messages
    COLONY_COMMAND = "colony.command"
    COLONY_STATUS = "colony.status"
    COLONY_SHUTDOWN = "colony.shutdown"
    
    # Swarm-level messages
    SWARM_DECISION = "swarm.decision"
    SWARM_OPPORTUNITY = "swarm.opportunity"
    SWARM_STATUS = "swarm.status"
    
    # Squad-level messages
    SQUAD_FORMATION = "squad.formation"
    SQUAD_SIGNAL = "squad.signal"
    SQUAD_DISSOLUTION = "squad.dissolution"
    
    # Wallet-level messages
    WALLET_TRADE = "wallet.trade"
    WALLET_BALANCE = "wallet.balance"
    WALLET_STATUS = "wallet.status"
    
    # Trading messages
    TRADE_ORDER = "trade.order"
    TRADE_EXECUTION = "trade.execution"
    TRADE_RESULT = "trade.result"
    
    # Intelligence messages
    MARKET_SIGNAL = "intelligence.market"
    SENTIMENT_SIGNAL = "intelligence.sentiment"
    ML_PREDICTION = "intelligence.ml"
    
    # Safety messages
    KILL_SWITCH = "safety.kill_switch"
    RUG_DETECTION = "safety.rug_detection"
    ALERT = "safety.alert"
    
    # System messages
    SYSTEM_STATUS = "system.status"
    SYSTEM_ERROR = "system.error"
    SYSTEM_METRIC = "system.metric"


@dataclass
class MessageEnvelope:
    """Standardized message envelope for all communications"""
    message_id: str
    message_type: MessageType
    subject: str
    sender_id: str
    timestamp: datetime
    priority: MessagePriority
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Payload
    data: Dict[str, Any] = None
    
    # Message routing
    routing_key: Optional[str] = None
    broadcast: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'subject': self.subject,
            'sender_id': self.sender_id,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'data': self.data or {},
            'routing_key': self.routing_key,
            'broadcast': self.broadcast
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageEnvelope':
        """Create from dictionary"""
        return cls(
            message_id=data['message_id'],
            message_type=MessageType(data['message_type']),
            subject=data['subject'],
            sender_id=data['sender_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            priority=MessagePriority(data['priority']),
            correlation_id=data.get('correlation_id'),
            reply_to=data.get('reply_to'),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            data=data.get('data', {}),
            routing_key=data.get('routing_key'),
            broadcast=data.get('broadcast', False)
        )


@dataclass
class MessageBusStats:
    """Message bus performance statistics"""
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    avg_latency_ms: float = 0.0
    throughput_per_second: float = 0.0
    active_subscriptions: int = 0
    connection_status: str = "disconnected"
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0


class MessageBusConfig:
    """Message bus configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG"""
    
    def __init__(self):
        from worker_ant_v1.core.unified_config import get_trading_config
        config = get_trading_config()  # Force through unified config
        
        # NATS server configuration
        self.nats_servers = config.nats_servers.split(',')
        self.connection_timeout = config.nats_connection_timeout
        self.reconnect_time_wait = config.nats_reconnect_wait
        self.max_reconnect_attempts = config.nats_max_reconnect
        
        # Message configuration
        self.message_max_size = config.nats_max_message_size
        self.enable_compression = config.nats_enable_compression
        self.compression_threshold = config.nats_compression_threshold
        
        # Performance configuration
        self.enable_metrics = config.nats_enable_metrics
        self.stats_interval = config.nats_stats_interval
        self.dead_letter_queue = config.nats_dead_letter_queue
        
        # Persistence configuration
        self.enable_persistence = config.nats_enable_persistence
        self.stream_name = config.nats_stream_name
        self.retention_policy = config.nats_retention_policy


class MessageBus:
    """High-performance NATS message bus for inter-component communication"""
    
    def __init__(self, config: Optional[MessageBusConfig] = None):
        self.logger = setup_logger("MessageBus")
        self.config = config or MessageBusConfig()
        
        # NATS client
        self.nats_client: Optional[NATS] = None
        self.js = None  # JetStream context
        
        # Message handlers and subscriptions
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.subscriptions: Dict[str, Subscription] = {}
        self.wildcard_handlers: Dict[str, Callable] = {}
        
        # Performance tracking
        self.stats = MessageBusStats()
        self.message_latencies: List[float] = []
        self.start_time: Optional[datetime] = None
        
        # System state
        self.connected = False
        self.running = False
        
        # Component identification - CANONICAL ACCESS THROUGH UNIFIED CONFIG
        from worker_ant_v1.core.unified_config import get_trading_config
        unified_config = get_trading_config()
        self.component_id = unified_config.component_id or f"antbot_{uuid.uuid4().hex[:8]}"
        
        # Background tasks
        self.stats_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Message serialization
        self.use_msgpack = True  # More efficient than JSON
        
        self.logger.info(f"🚌 Message Bus initialized for component: {self.component_id}")

    async def connect(self) -> bool:
        """Connect to NATS server"""
        try:
            self.logger.info(f"🔌 Connecting to NATS servers: {self.config.nats_servers}")
            
            # Connection options
            options = {
                "servers": self.config.nats_servers,
                "name": f"antbot-{self.component_id}",
                "connect_timeout": self.config.connection_timeout,
                "reconnect_time_wait": self.config.reconnect_time_wait,
                "max_reconnect_attempts": self.config.max_reconnect_attempts,
                "error_cb": self._error_callback,
                "disconnected_cb": self._disconnected_callback,
                "reconnected_cb": self._reconnected_callback,
                "closed_cb": self._closed_callback
            }
            
            # Create NATS client
            self.nats_client = NATS()
            await self.nats_client.connect(**options)
            
            # Initialize JetStream if persistence is enabled
            if self.config.enable_persistence:
                self.js = self.nats_client.jetstream()
                await self._setup_jetstream()
            
            self.connected = True
            self.running = True
            self.start_time = datetime.utcnow()
            self.stats.connection_status = "connected"
            
            # Start background tasks
            if self.config.enable_metrics:
                self.stats_task = asyncio.create_task(self._stats_loop())
            
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            self.logger.info("✅ Connected to NATS message bus")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to NATS: {e}")
            self.stats.last_error = str(e)
            return False

    async def disconnect(self):
        """Disconnect from NATS server"""
        try:
            self.logger.info("🔌 Disconnecting from NATS...")
            
            self.running = False
            
            # Cancel background tasks
            if self.stats_task:
                self.stats_task.cancel()
            if self.health_check_task:
                self.health_check_task.cancel()
            
            # Unsubscribe from all topics
            for subscription in self.subscriptions.values():
                await subscription.unsubscribe()
            
            # Close NATS connection
            if self.nats_client and self.nats_client.is_connected:
                await self.nats_client.close()
            
            self.connected = False
            self.stats.connection_status = "disconnected"
            
            self.logger.info("✅ Disconnected from NATS message bus")
            
        except Exception as e:
            self.logger.error(f"❌ Error disconnecting from NATS: {e}")

    async def publish(self, message: MessageEnvelope) -> bool:
        """Publish message to NATS"""
        try:
            if not self.connected or not self.nats_client:
                self.logger.error("❌ Cannot publish: not connected to NATS")
                return False
            
            # Check message expiration
            if message.expires_at and datetime.utcnow() > message.expires_at:
                self.logger.warning(f"⏰ Message {message.message_id} expired, not publishing")
                return False
            
            # Serialize message
            serialized_data = await self._serialize_message(message)
            
            # Determine subject
            subject = message.subject
            
            # Publish message
            start_time = time.time()
            
            if message.reply_to:
                # Request-reply pattern
                await self.nats_client.publish(subject, serialized_data, reply=message.reply_to)
            else:
                # Fire-and-forget
                await self.nats_client.publish(subject, serialized_data)
            
            # Track performance
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.message_latencies.append(latency)
            self.stats.messages_sent += 1
            
            self.logger.debug(f"📤 Published message {message.message_id} to {subject}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to publish message {message.message_id}: {e}")
            self.stats.messages_failed += 1
            return False

    async def subscribe(self, subject: str, handler: Callable, queue_group: Optional[str] = None) -> bool:
        """Subscribe to messages on a subject"""
        try:
            if not self.connected or not self.nats_client:
                self.logger.error("❌ Cannot subscribe: not connected to NATS")
                return False
            
            # Create subscription
            subscription = await self.nats_client.subscribe(
                subject,
                cb=self._message_callback_wrapper(handler),
                queue=queue_group
            )
            
            self.subscriptions[subject] = subscription
            
            # Track handlers
            if subject not in self.message_handlers:
                self.message_handlers[subject] = []
            self.message_handlers[subject].append(handler)
            
            self.stats.active_subscriptions += 1
            
            self.logger.info(f"📥 Subscribed to subject: {subject}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to subscribe to {subject}: {e}")
            return False

    async def unsubscribe(self, subject: str) -> bool:
        """Unsubscribe from a subject"""
        try:
            if subject in self.subscriptions:
                await self.subscriptions[subject].unsubscribe()
                del self.subscriptions[subject]
                
                if subject in self.message_handlers:
                    del self.message_handlers[subject]
                
                self.stats.active_subscriptions -= 1
                
                self.logger.info(f"📤 Unsubscribed from subject: {subject}")
                return True
            else:
                self.logger.warning(f"⚠️ No subscription found for subject: {subject}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to unsubscribe from {subject}: {e}")
            return False

    async def request_reply(self, message: MessageEnvelope, timeout: float = 10.0) -> Optional[MessageEnvelope]:
        """Send request and wait for reply"""
        try:
            if not self.connected or not self.nats_client:
                self.logger.error("❌ Cannot send request: not connected to NATS")
                return None
            
            # Serialize message
            serialized_data = await self._serialize_message(message)
            
            # Send request and wait for reply
            start_time = time.time()
            response = await self.nats_client.request(
                message.subject,
                serialized_data,
                timeout=timeout
            )
            
            # Track performance
            latency = (time.time() - start_time) * 1000
            self.message_latencies.append(latency)
            
            # Deserialize response
            reply_message = await self._deserialize_message(response.data)
            
            self.logger.debug(f"🔄 Request-reply completed for {message.message_id}")
            return reply_message
            
        except asyncio.TimeoutError:
            self.logger.warning(f"⏰ Request timeout for message {message.message_id}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Request-reply failed for {message.message_id}: {e}")
            return None

    async def _serialize_message(self, message: MessageEnvelope) -> bytes:
        """Serialize message for transmission"""
        try:
            # Convert to dictionary
            message_dict = message.to_dict()
            
            # Choose serialization method
            if self.use_msgpack:
                serialized = msgpack.packb(message_dict)
            else:
                serialized = json.dumps(message_dict).encode('utf-8')
            
            # Compress if enabled and message is large enough
            if (self.config.enable_compression and 
                len(serialized) > self.config.compression_threshold):
                serialized = gzip.compress(serialized)
                message_dict['_compressed'] = True
            
            return serialized
            
        except Exception as e:
            self.logger.error(f"❌ Failed to serialize message: {e}")
            raise

    async def _deserialize_message(self, data: bytes) -> MessageEnvelope:
        """Deserialize message from transmission"""
        try:
            # Check if compressed
            try:
                # Try to decompress first (gzip header detection)
                if data[:2] == b'\x1f\x8b':
                    data = gzip.decompress(data)
            except:
                pass  # Not compressed
            
            # Choose deserialization method
            if self.use_msgpack:
                try:
                    message_dict = msgpack.unpackb(data, raw=False)
                except:
                    # Fallback to JSON
                    message_dict = json.loads(data.decode('utf-8'))
            else:
                message_dict = json.loads(data.decode('utf-8'))
            
            # Create message envelope
            return MessageEnvelope.from_dict(message_dict)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to deserialize message: {e}")
            raise

    def _message_callback_wrapper(self, handler: Callable):
        """Wrap message handler with error handling and metrics"""
        async def wrapper(msg):
            try:
                start_time = time.time()
                
                # Deserialize message
                message = await self._deserialize_message(msg.data)
                
                # Track stats
                self.stats.messages_received += 1
                
                # Call handler
                await handler(message)
                
                # Track latency
                latency = (time.time() - start_time) * 1000
                self.message_latencies.append(latency)
                
            except Exception as e:
                self.logger.error(f"❌ Error handling message: {e}")
                self.stats.messages_failed += 1
        
        return wrapper

    async def _setup_jetstream(self):
        """Setup JetStream for message persistence"""
        try:
            # Create stream if it doesn't exist
            stream_config = {
                'name': self.config.stream_name,
                'subjects': ['antbot.*'],
                'retention': self.config.retention_policy,
                'max_age': 86400,  # 24 hours
                'max_msgs': 1000000,  # 1M messages
                'max_bytes': 1073741824  # 1GB
            }
            
            try:
                await self.js.stream_info(self.config.stream_name)
                self.logger.info(f"✅ JetStream stream {self.config.stream_name} already exists")
            except:
                await self.js.add_stream(**stream_config)
                self.logger.info(f"✅ Created JetStream stream {self.config.stream_name}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to setup JetStream: {e}")

    async def _stats_loop(self):
        """Background stats collection loop"""
        last_sent = 0
        last_received = 0
        last_time = time.time()
        
        while self.running:
            try:
                await asyncio.sleep(self.config.stats_interval)
                
                current_time = time.time()
                time_delta = current_time - last_time
                
                # Calculate throughput
                sent_delta = self.stats.messages_sent - last_sent
                received_delta = self.stats.messages_received - last_received
                
                self.stats.throughput_per_second = (sent_delta + received_delta) / time_delta
                
                # Calculate average latency
                if self.message_latencies:
                    self.stats.avg_latency_ms = sum(self.message_latencies) / len(self.message_latencies)
                    
                    # Keep only recent latencies (sliding window)
                    if len(self.message_latencies) > 1000:
                        self.message_latencies = self.message_latencies[-500:]
                
                # Calculate uptime
                if self.start_time:
                    self.stats.uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
                
                # Log stats periodically
                if self.stats.messages_sent % 1000 == 0 and self.stats.messages_sent > 0:
                    self.logger.info(
                        f"📊 Message Bus Stats: "
                        f"Sent: {self.stats.messages_sent}, "
                        f"Received: {self.stats.messages_received}, "
                        f"Failed: {self.stats.messages_failed}, "
                        f"Throughput: {self.stats.throughput_per_second:.1f}/s, "
                        f"Latency: {self.stats.avg_latency_ms:.1f}ms"
                    )
                
                # Update for next iteration
                last_sent = self.stats.messages_sent
                last_received = self.stats.messages_received
                last_time = current_time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in stats loop: {e}")

    async def _health_check_loop(self):
        """Background health check loop"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check NATS connection
                if self.nats_client and not self.nats_client.is_connected:
                    self.logger.warning("⚠️ NATS connection lost, attempting reconnect...")
                    self.connected = False
                    self.stats.connection_status = "reconnecting"
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in health check: {e}")

    # NATS event callbacks
    async def _error_callback(self, e):
        """Handle NATS errors"""
        self.logger.error(f"❌ NATS error: {e}")
        self.stats.last_error = str(e)

    async def _disconnected_callback(self):
        """Handle NATS disconnection"""
        self.logger.warning("⚠️ Disconnected from NATS")
        self.connected = False
        self.stats.connection_status = "disconnected"

    async def _reconnected_callback(self):
        """Handle NATS reconnection"""
        self.logger.info("✅ Reconnected to NATS")
        self.connected = True
        self.stats.connection_status = "connected"

    async def _closed_callback(self):
        """Handle NATS connection closed"""
        self.logger.info("🔌 NATS connection closed")
        self.connected = False
        self.stats.connection_status = "closed"

    # High-level convenience methods

    async def send_colony_command(self, command: str, data: Dict[str, Any], target_swarm: Optional[str] = None) -> bool:
        """Send command from Colony Commander to swarms"""
        subject = f"colony.command.{target_swarm}" if target_swarm else "colony.command.all"
        
        message = MessageEnvelope(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.COLONY_COMMAND,
            subject=subject,
            sender_id=self.component_id,
            timestamp=datetime.utcnow(),
            priority=MessagePriority.HIGH,
            data={'command': command, **data}
        )
        
        return await self.publish(message)

    async def send_swarm_opportunity(self, opportunity_data: Dict[str, Any], swarm_id: str) -> bool:
        """Send trading opportunity to swarm"""
        message = MessageEnvelope(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SWARM_OPPORTUNITY,
            subject=f"swarm.{swarm_id}.opportunity",
            sender_id=self.component_id,
            timestamp=datetime.utcnow(),
            priority=MessagePriority.HIGH,
            data=opportunity_data
        )
        
        return await self.publish(message)

    async def send_trade_order(self, order_data: Dict[str, Any], wallet_id: str) -> bool:
        """Send trade order to specific wallet"""
        message = MessageEnvelope(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TRADE_ORDER,
            subject=f"wallet.{wallet_id}.trade",
            sender_id=self.component_id,
            timestamp=datetime.utcnow(),
            priority=MessagePriority.HIGH,
            data=order_data
        )
        
        return await self.publish(message)

    async def send_squad_signal(self, signal_data: Dict[str, Any], squad_type: str) -> bool:
        """Send signal to squad members"""
        message = MessageEnvelope(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SQUAD_SIGNAL,
            subject=f"squad.{squad_type}.signal",
            sender_id=self.component_id,
            timestamp=datetime.utcnow(),
            priority=MessagePriority.MEDIUM,
            data=signal_data,
            broadcast=True
        )
        
        return await self.publish(message)

    async def send_kill_switch(self, reason: str, component: Optional[str] = None) -> bool:
        """Send kill switch activation"""
        subject = f"safety.kill_switch.{component}" if component else "safety.kill_switch.all"
        
        message = MessageEnvelope(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.KILL_SWITCH,
            subject=subject,
            sender_id=self.component_id,
            timestamp=datetime.utcnow(),
            priority=MessagePriority.EMERGENCY,
            data={'reason': reason, 'component': component},
            broadcast=True
        )
        
        return await self.publish(message)

    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        return {
            'connected': self.connected,
            'component_id': self.component_id,
            'stats': asdict(self.stats),
            'active_subscriptions': list(self.subscriptions.keys()),
            'config': {
                'nats_servers': self.config.nats_servers,
                'enable_persistence': self.config.enable_persistence,
                'enable_compression': self.config.enable_compression
            }
        }


# Global message bus instance
_message_bus: Optional[MessageBus] = None


def get_message_bus_config() -> MessageBusConfig:
    """Get message bus configuration from environment"""
    return MessageBusConfig()


async def get_message_bus() -> MessageBus:
    """Get or create global message bus instance"""
    global _message_bus
    
    if _message_bus is None:
        config = get_message_bus_config()
        _message_bus = MessageBus(config)
        await _message_bus.connect()
    
    return _message_bus


async def shutdown_message_bus():
    """Shutdown global message bus"""
    global _message_bus
    
    if _message_bus:
        await _message_bus.disconnect()
        _message_bus = None


# Subject name helpers

def colony_subject(action: str, target: str = "all") -> str:
    """Generate colony-level subject"""
    return f"colony.{action}.{target}"


def swarm_subject(swarm_id: str, action: str) -> str:
    """Generate swarm-level subject"""
    return f"swarm.{swarm_id}.{action}"


def squad_subject(squad_type: str, action: str) -> str:
    """Generate squad-level subject"""
    return f"squad.{squad_type}.{action}"


def wallet_subject(wallet_id: str, action: str) -> str:
    """Generate wallet-level subject"""
    return f"wallet.{wallet_id}.{action}"


def intelligence_subject(intel_type: str) -> str:
    """Generate intelligence subject"""
    return f"intelligence.{intel_type}"


def safety_subject(safety_type: str, component: str = "all") -> str:
    """Generate safety subject"""
    return f"safety.{safety_type}.{component}"


def system_subject(system_type: str) -> str:
    """Generate system subject"""
    return f"system.{system_type}" 