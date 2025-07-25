"""
PRODUCTION-READY SECURE LOGGING SYSTEM
=====================================

Thread-safe, async-compatible logging system with TimescaleDB storage,
sensitive data masking, and comprehensive monitoring capabilities.
Enhanced for high-frequency trading data with time-series optimization.
"""

import asyncio
import hashlib
import json
import logging
import logging.handlers
import queue
import secrets
import structlog
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

# Import unified schemas to eliminate circular dependencies
from worker_ant_v1.core.schemas import TradeRecord, SystemEvent

# TimescaleDB integration - using lazy imports to avoid circular dependency

# Security imports
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False


@dataclass
class HourlyReport:
    """Comprehensive hourly performance report."""

    hour_start: str
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_volume_sol: float
    total_profit_sol: float
    total_loss_sol: float
    net_profit_sol: float
    win_rate_percent: float
    largest_win_sol: float
    largest_loss_sol: float
    average_latency_ms: float
    error_rate: float
    uptime_percent: float
    sharpe_ratio: float = 0.0

    def to_system_event(self, session_id: str):
        """Convert to TimescaleDB system event"""
        # Lazy import to avoid circular dependency
        from worker_ant_v1.core.database import SystemEvent as DBSystemEvent
        
        return DBSystemEvent(
            timestamp=datetime.fromisoformat(self.hour_start),
            event_id=str(uuid.uuid4()),
            event_type="hourly_report",
            component="trading_system",
            severity="INFO",
            message=f"Hourly report: {self.total_trades} trades, {self.win_rate_percent:.1f}% win rate",
            event_data=asdict(self),
            session_id=session_id
        )


class SecureDataProcessor:
    """Handles sensitive data encryption and sanitization."""

    def __init__(self, encryption_key: Optional[str] = None):
        """Initialize the secure data processor.

        Args:
            encryption_key: Optional encryption key for sensitive data
        """
        self.encryption_enabled = ENCRYPTION_AVAILABLE and encryption_key is not None

        if self.encryption_enabled:
            if isinstance(encryption_key, str):
                encryption_key = encryption_key.encode()

            self.fernet = Fernet(encryption_key)
        else:
            self.fernet = None

        # Sensitive field patterns to mask
        self.sensitive_patterns = [
            "private_key",
            "secret",
            "password",
            "token",
            "signature",
            "seed",
            "mnemonic",
            "key",
        ]

    def sanitize_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize log data by masking sensitive fields.

        Args:
            data: Dictionary containing log data

        Returns:
            Sanitized dictionary with masked sensitive data
        """
        sanitized = {}

        for key, value in data.items():
            key_lower = key.lower()

            # Check if this is a sensitive field
            is_sensitive = any(pattern in key_lower for pattern in self.sensitive_patterns)

            if is_sensitive:
                if isinstance(value, str) and len(value) > 8:
                    # Show first 4 and last 4 characters
                    sanitized[key] = f"{value[:4]}***{value[-4:]}"
                else:
                    sanitized[key] = "***MASKED***"
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = self.sanitize_log_data(value)
            else:
                sanitized[key] = value

        return sanitized

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data if encryption is enabled.

        Args:
            data: String data to encrypt

        Returns:
            Encrypted data or original data if encryption disabled
        """
        if not self.encryption_enabled:
            return data

        try:
            encrypted = self.fernet.encrypt(data.encode())
            return encrypted.decode()
        except Exception:
            # If encryption fails, return masked data
            return "***ENCRYPTION_FAILED***"

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data if encryption is enabled.

        Args:
            encrypted_data: Encrypted string data

        Returns:
            Decrypted data or original data if encryption disabled
        """
        if not self.encryption_enabled:
            return encrypted_data

        try:
            decrypted = self.fernet.decrypt(encrypted_data.encode())
            return decrypted.decode()
        except Exception:
            return "***DECRYPTION_FAILED***"


class AsyncDatabaseManager:
    """Thread-safe async database manager using TimescaleDB."""

    def __init__(self, db_path: str = None):
        """Initialize the database manager.

        Args:
            db_path: Path to the database file (legacy compatibility, ignored for TimescaleDB)
        """
        self.db_manager = None
        self.initialized = False
        self.logger = logging.getLogger("AsyncDatabaseManager")

    async def initialize(self):
        """Initialize TimescaleDB database manager."""
        if self.initialized:
            return

        try:
            # Lazy import to avoid circular dependency
            from worker_ant_v1.core.database import get_database_manager
            
            self.db_manager = await get_database_manager()
            self.initialized = True
            self.logger.info("✅ AsyncDatabaseManager initialized with TimescaleDB")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize AsyncDatabaseManager: {e}")
            raise

    async def insert_trade_record(self, trade: TradeRecord):
        """Insert trade record with error handling.

        Args:
            trade: Trade record to insert
        """
        try:
            if not self.initialized:
                await self.initialize()
                
            # Convert to TimescaleDB format and insert
            db_record = trade.to_db_record()
            await self.db_manager.insert_trade(db_record)
            
        except Exception as e:
            # Log error but don't fail the application
            self.logger.error(f"Database error inserting trade: {e}")

    async def insert_hourly_report(self, report: HourlyReport):
        """Insert hourly report.

        Args:
            report: Hourly report to insert
        """
        try:
            if not self.initialized:
                await self.initialize()
                
            # Convert to system event and insert
            session_id = secrets.token_hex(8)  # Generate session ID for report
            event = report.to_system_event(session_id)
            await self.db_manager.insert_system_event(event)
            
        except Exception as e:
            self.logger.error(f"Database error inserting hourly report: {e}")

    async def log_system_event(
        self, event_type: str, event_data: Dict[str, Any], severity: str = "INFO"
    ):
        """Log system event.

        Args:
            event_type: Type of event
            event_data: Event data
            severity: Event severity
        """
        try:
            if not self.initialized:
                await self.initialize()
                
            # Create system event
            event = DBSystemEvent(
                timestamp=datetime.utcnow(),
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                component="trading_system",
                severity=severity,
                message=f"System event: {event_type}",
                event_data=event_data
            )
            
            await self.db_manager.insert_system_event(event)
            
        except Exception as e:
            self.logger.error(f"Database error logging system event: {e}")

    async def get_trades_for_period(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get trades for a specific time period (for hourly reports).
        
        Args:
            start_time: Start of period
            end_time: End of period
            
        Returns:
            List of trade records
        """
        try:
            if not self.initialized:
                await self.initialize()
                
            return await self.db_manager.query_trades(start_time, end_time)
            
        except Exception as e:
            self.logger.error(f"Database error querying trades: {e}")
            return []

    # Legacy compatibility methods (maintain same interface)
    @asynccontextmanager
    async def get_connection(self):
        """Legacy compatibility - get database connection context."""
        # For TimescaleDB, we'll use the connection pool internally
        yield self

    async def execute(self, query: str, *params):
        """Legacy compatibility - execute query."""
        # This method is for backward compatibility only
        # New code should use the specific insert methods above
        pass

    async def commit(self):
        """Legacy compatibility - commit transaction."""
        # TimescaleDB auto-commits with our batch system
        pass


class ProductionLogger:
    """Production-ready logger with async support and security features."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the production logger.

        Args:
            config: Logger configuration
        """
        self.config = config
        self.session_id = secrets.token_hex(8)
        self.session_start = datetime.utcnow()
        self.trade_count = 0
        self.error_count = 0

        # Initialize components
        self.db_manager = AsyncDatabaseManager(
            config.get("db_path", "logs/trading.db")  # Legacy path, ignored
        )
        self.security_processor = SecureDataProcessor(
            config.get("encryption_key")
        )

        # Async queue for trade records
        self.trade_queue = asyncio.Queue()
        self.batch_size = config.get("batch_size", 10)
        self.batch_timeout = config.get("batch_timeout", 5.0)

        # Background tasks
        self._batch_task = None
        self._hourly_task = None
        self._monitor_task = None
        self._running = False

        # Setup structured logging
        self._setup_structured_logging()

    def _setup_structured_logging(self):
        """Setup structured logging with structlog."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self._security_processor,
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        self.logger = structlog.get_logger("ProductionLogger")

    def _security_processor(self, logger, method_name, event_dict):
        """Process events for security (sanitize sensitive data)."""
        return self.security_processor.sanitize_log_data(event_dict)

    async def start(self):
        """Start the logger and background tasks."""
        if self._running:
            return

        await self.db_manager.initialize()

        # Start background tasks
        self._batch_task = asyncio.create_task(self._batch_processor())
        self._hourly_task = asyncio.create_task(self._hourly_reporter())
        self._monitor_task = asyncio.create_task(self._system_monitor())

        self._running = True
        self.logger.info("Production logger started", session_id=self.session_id)

    async def stop(self):
        """Stop the logger and background tasks."""
        if not self._running:
            return

        self._running = False

        # Cancel background tasks
        if self._batch_task:
            self._batch_task.cancel()
        if self._hourly_task:
            self._hourly_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()

        # Flush remaining queue
        await self._flush_queue()

        self.logger.info("Production logger stopped")

    async def log_trade(self, trade_record: TradeRecord):
        """Log a trade record.

        Args:
            trade_record: Trade record to log
        """
        try:
            trade_record.mask_sensitive_data()
            await self.trade_queue.put(trade_record)
            self.trade_count += 1
        except Exception as e:
            self.error_count += 1
            self.logger.error("Failed to queue trade record", error=str(e))

    async def log_system_event(
        self, event_type: str, event_data: Dict[str, Any], severity: str = "INFO"
    ):
        """Log a system event.

        Args:
            event_type: Type of event
            event_data: Event data
            severity: Event severity
        """
        try:
            await self.db_manager.log_system_event(event_type, event_data, severity)
            self.logger.info(
                "System event logged",
                event_type=event_type,
                severity=severity,
                **event_data,
            )
        except Exception as e:
            self.error_count += 1
            self.logger.error("Failed to log system event", error=str(e))

    async def _batch_processor(self):
        """Process trade records in batches."""
        batch = []
        last_flush = time.time()

        while self._running:
            try:
                # Wait for next trade or timeout
                try:
                    trade = await asyncio.wait_for(
                        self.trade_queue.get(), timeout=self.batch_timeout
                    )
                    batch.append(trade)
                except asyncio.TimeoutError:
                    pass

                # Flush if batch is full or timeout reached
                if (
                    len(batch) >= self.batch_size
                    or (batch and time.time() - last_flush >= self.batch_timeout)
                ):
                    await self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in batch processor", error=str(e))
                await asyncio.sleep(1)

        # Flush remaining batch
        if batch:
            await self._flush_batch(batch)

    async def _flush_batch(self, batch: List[TradeRecord]):
        """Flush a batch of trade records to database."""
        for trade in batch:
            await self.db_manager.insert_trade_record(trade)

    async def _flush_queue(self):
        """Flush all remaining items in the queue."""
        batch = []
        while not self.trade_queue.empty():
            try:
                trade = self.trade_queue.get_nowait()
                batch.append(trade)
            except asyncio.QueueEmpty:
                break

        if batch:
            await self._flush_batch(batch)

    async def _hourly_reporter(self):
        """Generate hourly performance reports."""
        while self._running:
            try:
                # Wait until next hour boundary
                now = datetime.utcnow()
                next_hour = (now + timedelta(hours=1)).replace(
                    minute=0, second=0, microsecond=0
                )
                sleep_seconds = (next_hour - now).total_seconds()

                await asyncio.sleep(sleep_seconds)

                # Generate report for previous hour
                report_hour = now.replace(minute=0, second=0, microsecond=0)
                report = await self._generate_hourly_report(report_hour)

                if report:
                    await self.db_manager.insert_hourly_report(report)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in hourly reporter", error=str(e))
                await asyncio.sleep(3600)

    async def _generate_hourly_report(
        self, hour_start: datetime
    ) -> Optional[HourlyReport]:
        """Generate hourly performance report.

        Args:
            hour_start: Start of the hour to report on

        Returns:
            HourlyReport or None if generation fails
        """
        try:
            hour_end = hour_start + timedelta(hours=1)
            
            # Get trades for the hour from TimescaleDB
            trades = await self.db_manager.get_trades_for_period(hour_start, hour_end)

            if not trades:
                return HourlyReport(
                    hour_start=hour_start.isoformat(),
                    total_trades=0,
                    successful_trades=0,
                    failed_trades=0,
                    total_volume_sol=0.0,
                    total_profit_sol=0.0,
                    total_loss_sol=0.0,
                    net_profit_sol=0.0,
                    win_rate_percent=0.0,
                    largest_win_sol=0.0,
                    largest_loss_sol=0.0,
                    average_latency_ms=0.0,
                    error_rate=0.0,
                    uptime_percent=100.0,
                )

            # Calculate metrics
            total_trades = len(trades)
            successful_trades = sum(1 for t in trades if t.get('success', False))
            failed_trades = total_trades - successful_trades
            
            total_volume_sol = sum(t.get('amount_sol', 0) for t in trades)
            profits = [t.get('profit_loss_sol', 0) for t in trades if t.get('profit_loss_sol', 0) > 0]
            losses = [abs(t.get('profit_loss_sol', 0)) for t in trades if t.get('profit_loss_sol', 0) < 0]
            
            total_profit_sol = sum(profits)
            total_loss_sol = sum(losses)
            net_profit_sol = total_profit_sol - total_loss_sol
            
            win_rate_percent = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            largest_win_sol = max(profits) if profits else 0
            largest_loss_sol = max(losses) if losses else 0
            
            latencies = [t.get('latency_ms', 0) for t in trades if t.get('latency_ms') is not None]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            return HourlyReport(
                hour_start=hour_start.isoformat(),
                total_trades=total_trades,
                successful_trades=successful_trades,
                failed_trades=failed_trades,
                total_volume_sol=total_volume_sol,
                total_profit_sol=total_profit_sol,
                total_loss_sol=total_loss_sol,
                net_profit_sol=net_profit_sol,
                win_rate_percent=win_rate_percent,
                largest_win_sol=largest_win_sol,
                largest_loss_sol=largest_loss_sol,
                sharpe_ratio=0.0,  # Calculate if needed
                average_latency_ms=avg_latency,
                error_rate=(failed_trades / total_trades * 100) if total_trades > 0 else 0,
                uptime_percent=100.0,  # Calculate based on system events
            )

        except Exception as e:
            self.logger.error("Failed to generate hourly report", error=str(e))
            return None

    async def _system_monitor(self):
        """Monitor system health and log events."""
        while self._running:
            try:
                # Basic system monitoring
                import psutil

                system_stats = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage("/").percent,
                    "trade_count": self.trade_count,
                    "error_count": self.error_count,
                    "queue_size": self.trade_queue.qsize(),
                }

                # Log if concerning metrics
                if (
                    system_stats["cpu_percent"] > 80
                    or system_stats["memory_percent"] > 85
                    or system_stats["queue_size"] > 100
                ):
                    await self.log_system_event("system_warning", system_stats, "WARNING")

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in system monitor", error=str(e))
                await asyncio.sleep(60)


# Global logger instance
_production_logger: Optional[ProductionLogger] = None


def get_production_logger(config: Optional[Dict[str, Any]] = None) -> ProductionLogger:
    """Get or create global production logger instance.

    Args:
        config: Optional logger configuration

    Returns:
        ProductionLogger instance
    """
    global _production_logger

    if _production_logger is None:
        if config is None:
            config = {
                "db_path": "logs/trading.db",
                "batch_size": 10,
                "batch_timeout": 5.0,
            }

        _production_logger = ProductionLogger(config)

    return _production_logger


def setup_logger(name: str) -> logging.Logger:
    """Setup a logger with the given name.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name (alias for setup_logger).

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    return setup_logger(name) 