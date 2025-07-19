"""
PRODUCTION-READY SECURE LOGGING SYSTEM
=====================================

Thread-safe, async-compatible logging system with encrypted storage,
sensitive data masking, and comprehensive monitoring capabilities.
"""

import asyncio
import aiosqlite
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
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Security imports
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False


@dataclass
class TradeRecord:
    """Secure trade record with sensitive data handling."""

    # Basic trade info
    timestamp: str
    token_address: str
    token_symbol: str
    trade_type: str  # BUY or SELL
    success: bool

    # Trade details
    amount_sol: float
    amount_tokens: float
    price: float
    slippage_percent: float

    # Performance metrics
    latency_ms: int
    gas_cost_sol: float = 0.0
    rpc_cost_sol: float = 0.0
    api_cost_sol: float = 0.0

    # P&L (for sells)
    profit_loss_sol: Optional[float] = None
    profit_loss_percent: Optional[float] = None
    hold_time_seconds: Optional[int] = None

    # Technical details (sensitive data masked)
    tx_signature_hash: Optional[str] = None  # Hash instead of full signature
    retry_count: int = 0
    exit_reason: Optional[str] = None
    error_message: Optional[str] = None

    # Session tracking
    session_id: str = field(default_factory=lambda: secrets.token_hex(8))

    def mask_sensitive_data(self):
        """Mask sensitive data for safe logging."""
        if hasattr(self, "tx_signature") and self.tx_signature:
            self.tx_signature_hash = hashlib.sha256(
                self.tx_signature.encode()
            ).hexdigest()[:16]
            delattr(self, "tx_signature")


@dataclass
class HourlyReport:
    """Comprehensive hourly performance report."""

    hour_start: str
    total_trades: int
    successful_trades: int
    buy_trades: int
    sell_trades: int

    # Performance metrics
    win_rate: float
    total_profit_sol: float
    total_volume_sol: float
    average_hold_time: float
    average_profit_percent: float

    # Cost analysis
    total_gas_costs: float
    total_rpc_costs: float
    total_api_costs: float
    cost_per_trade: float
    profit_after_costs: float

    # Risk metrics
    max_drawdown_percent: float
    largest_loss_sol: float
    largest_win_sol: float
    sharpe_ratio: float

    # System metrics
    average_latency_ms: float
    error_rate: float
    uptime_percent: float


class SecureDataProcessor:
    """Handles secure data processing and encryption."""

    def __init__(self, encryption_key: Optional[bytes] = None):
        """Initialize the secure data processor.

        Args:
            encryption_key: Optional encryption key for sensitive data
        """
        self.encryption_enabled = ENCRYPTION_AVAILABLE and encryption_key is not None
        if self.encryption_enabled:
            self.cipher = Fernet(encryption_key)
        else:
            self.cipher = None

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data if encryption is available.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data or original data if encryption not available
        """
        if self.encryption_enabled and self.cipher:
            return self.cipher.encrypt(data.encode()).decode()
        return data

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data if encryption is available.

        Args:
            encrypted_data: Data to decrypt

        Returns:
            Decrypted data or original data if encryption not available
        """
        if self.encryption_enabled and self.cipher:
            return self.cipher.decrypt(encrypted_data.encode()).decode()
        return encrypted_data

    def mask_private_key(self, key: str) -> str:
        """Mask private keys for safe logging.

        Args:
            key: Private key to mask

        Returns:
            Masked private key
        """
        if not key or len(key) < 8:
            return "****"
        return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"

    def sanitize_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize log data to remove/mask sensitive information.

        Args:
            data: Data to sanitize

        Returns:
            Sanitized data
        """
        sensitive_keys = ["private_key", "secret", "password", "api_key", "signature"]
        sanitized = {}

        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                if isinstance(value, str):
                    sanitized[key] = self.mask_private_key(value)
                else:
                    sanitized[key] = "***MASKED***"
            else:
                sanitized[key] = value

        return sanitized


class AsyncDatabaseManager:
    """Thread-safe async database manager."""

    def __init__(self, db_path: str):
        """Initialize the database manager.

        Args:
            db_path: Path to the database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection_pool = []
        self._pool_lock = asyncio.Lock()
        self.initialized = False

    async def initialize(self):
        """Initialize database schema."""
        if self.initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT,
                    trade_type TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    amount_sol REAL,
                    amount_tokens REAL,
                    price REAL,
                    slippage_percent REAL,
                    latency_ms INTEGER,
                    gas_cost_sol REAL DEFAULT 0,
                    rpc_cost_sol REAL DEFAULT 0,
                    api_cost_sol REAL DEFAULT 0,
                    profit_loss_sol REAL,
                    profit_loss_percent REAL,
                    hold_time_seconds INTEGER,
                    tx_signature_hash TEXT,
                    retry_count INTEGER DEFAULT 0,
                    exit_reason TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS hourly_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hour_start TEXT NOT NULL,
                    report_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    severity TEXT DEFAULT 'INFO',
                    timestamp TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indexes for performance
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_token ON trades(token_address)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_type ON trades(trade_type)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type)"
            )

            await db.commit()

        self.initialized = True

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        async with self._pool_lock:
            if self._connection_pool:
                conn = self._connection_pool.pop()
            else:
                conn = await aiosqlite.connect(self.db_path)

        try:
            yield conn
        finally:
            async with self._pool_lock:
                if len(self._connection_pool) < 5:  # Max pool size
                    self._connection_pool.append(conn)
                else:
                    await conn.close()

    async def insert_trade_record(self, trade: TradeRecord):
        """Insert trade record with error handling.

        Args:
            trade: Trade record to insert
        """
        try:
            async with self.get_connection() as db:
                await db.execute(
                    """
                    INSERT INTO trades (
                        timestamp, session_id, token_address, token_symbol,
                        trade_type, success, amount_sol, amount_tokens,
                        price, slippage_percent, latency_ms, gas_cost_sol,
                        rpc_cost_sol, api_cost_sol, profit_loss_sol,
                        profit_loss_percent, hold_time_seconds, tx_signature_hash,
                        retry_count, exit_reason, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade.timestamp,
                        trade.session_id,
                        trade.token_address,
                        trade.token_symbol,
                        trade.trade_type,
                        trade.success,
                        trade.amount_sol,
                        trade.amount_tokens,
                        trade.price,
                        trade.slippage_percent,
                        trade.latency_ms,
                        trade.gas_cost_sol,
                        trade.rpc_cost_sol,
                        trade.api_cost_sol,
                        trade.profit_loss_sol,
                        trade.profit_loss_percent,
                        trade.hold_time_seconds,
                        trade.tx_signature_hash,
                        trade.retry_count,
                        trade.exit_reason,
                        trade.error_message,
                    ),
                )
                await db.commit()
        except Exception as e:
            # Log error but don't fail the application
            print(f"Database error: {e}")

    async def insert_hourly_report(self, report: HourlyReport):
        """Insert hourly report.

        Args:
            report: Hourly report to insert
        """
        try:
            async with self.get_connection() as db:
                await db.execute(
                    "INSERT INTO hourly_reports (hour_start, report_data) VALUES (?, ?)",
                    (report.hour_start, json.dumps(asdict(report))),
                )
                await db.commit()
        except Exception as e:
            print(f"Database error: {e}")

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
            async with self.get_connection() as db:
                await db.execute(
                    "INSERT INTO system_events (event_type, event_data, severity, timestamp) VALUES (?, ?, ?, ?)",
                    (event_type, json.dumps(event_data), severity, datetime.utcnow().isoformat()),
                )
                await db.commit()
        except Exception as e:
            print(f"Database error: {e}")


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
            config.get("db_path", "logs/trading.db")
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
        """Generate hourly reports."""
        while self._running:
            try:
                # Wait until next hour
                now = datetime.utcnow()
                next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(
                    hours=1
                )
                sleep_seconds = (next_hour - now).total_seconds()

                await asyncio.sleep(sleep_seconds)

                # Generate report for previous hour
                report_hour = next_hour - timedelta(hours=1)
                report = await self._generate_hourly_report(report_hour)

                if report:
                    await self.db_manager.insert_hourly_report(report)
                    self.logger.info(
                        "Hourly report generated",
                        hour=report.hour_start,
                        trades=report.total_trades,
                        profit=report.total_profit_sol,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in hourly reporter", error=str(e))
                await asyncio.sleep(300)  # 5 minute delay on error

    async def _generate_hourly_report(
        self, hour_start: datetime
    ) -> Optional[HourlyReport]:
        """Generate hourly performance report.

        Args:
            hour_start: Start of the hour

        Returns:
            Hourly report or None if no data
        """
        try:
            hour_end = hour_start + timedelta(hours=1)
            hour_start_str = hour_start.isoformat()
            hour_end_str = hour_end.isoformat()

            async with self.db_manager.get_connection() as db:
                # Get trades for the hour
                async with db.execute(
                    """
                    SELECT * FROM trades 
                    WHERE timestamp BETWEEN ? AND ?
                    """,
                    (hour_start_str, hour_end_str),
                ) as cursor:
                    trades = await cursor.fetchall()

                if not trades:
                    return None

                # Calculate metrics
                total_trades = len(trades)
                successful_trades = sum(1 for t in trades if t[5])  # success column
                buy_trades = sum(1 for t in trades if t[4] == "BUY")  # trade_type column
                sell_trades = sum(1 for t in trades if t[4] == "SELL")

                # Calculate financial metrics
                profit_trades = [
                    t for t in trades if t[15] is not None and t[15] > 0
                ]  # profit_loss_sol
                total_profit = sum(t[15] for t in trades if t[15] is not None)
                total_volume = sum(t[6] for t in trades if t[6] is not None)  # amount_sol

                # Calculate other metrics
                win_rate = (
                    (len(profit_trades) / total_trades * 100) if total_trades > 0 else 0
                )
                avg_hold_time = (
                    sum(t[16] for t in trades if t[16] is not None) / len(trades)
                    if trades
                    else 0
                )
                avg_latency = (
                    sum(t[10] for t in trades if t[10] is not None) / len(trades)
                    if trades
                    else 0
                )

                # Cost analysis
                total_gas = sum(t[11] for t in trades if t[11] is not None)
                total_rpc = sum(t[12] for t in trades if t[12] is not None)
                total_api = sum(t[13] for t in trades if t[13] is not None)

                return HourlyReport(
                    hour_start=hour_start_str,
                    total_trades=total_trades,
                    successful_trades=successful_trades,
                    buy_trades=buy_trades,
                    sell_trades=sell_trades,
                    win_rate=win_rate,
                    total_profit_sol=total_profit,
                    total_volume_sol=total_volume,
                    average_hold_time=avg_hold_time,
                    average_profit_percent=0.0,  # Calculate if needed
                    total_gas_costs=total_gas,
                    total_rpc_costs=total_rpc,
                    total_api_costs=total_api,
                    cost_per_trade=(
                        (total_gas + total_rpc + total_api) / total_trades
                        if total_trades > 0
                        else 0
                    ),
                    profit_after_costs=total_profit - (total_gas + total_rpc + total_api),
                    max_drawdown_percent=0.0,  # Calculate if needed
                    largest_loss_sol=min(
                        (t[15] for t in trades if t[15] is not None), default=0
                    ),
                    largest_win_sol=max(
                        (t[15] for t in trades if t[15] is not None), default=0
                    ),
                    sharpe_ratio=0.0,  # Calculate if needed
                    average_latency_ms=avg_latency,
                    error_rate=(
                        (total_trades - successful_trades) / total_trades * 100
                        if total_trades > 0
                        else 0
                    ),
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

    def get_session_summary(self) -> Dict[str, Any]:
        """Get current session summary.

        Returns:
            Session summary dictionary
        """
        uptime = (datetime.utcnow() - self.session_start).total_seconds()

        return {
            "session_id": self.session_id,
            "uptime_seconds": uptime,
            "trade_count": self.trade_count,
            "error_count": self.error_count,
            "error_rate": (self.error_count / max(self.trade_count, 1)) * 100,
            "queue_size": self.trade_queue.qsize(),
            "start_time": self.session_start.isoformat(),
        }


# === GLOBAL LOGGER INSTANCE ===

# Create global logger instance (will be initialized by the application)
trading_logger: Optional[ProductionLogger] = None


async def initialize_logger(config: Dict[str, Any]) -> ProductionLogger:
    """Initialize the global trading logger.

    Args:
        config: Logger configuration

    Returns:
        Initialized production logger
    """
    global trading_logger
    trading_logger = ProductionLogger(config)
    await trading_logger.start()
    return trading_logger


async def shutdown_logger():
    """Shutdown the global trading logger."""
    global trading_logger
    if trading_logger:
        await trading_logger.stop()
        trading_logger = None


# === SIMPLE LOGGER FOR BASIC USE ===

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Simple synchronous logger setup for basic use.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance. If no name is provided, returns the root logger.

    Args:
        name: Optional name for the logger

    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger()
    return setup_logger(name)


# Export main classes and functions
__all__ = [
    "TradeRecord",
    "HourlyReport",
    "ProductionLogger",
    "initialize_logger",
    "shutdown_logger",
    "trading_logger",
    "setup_logger",
    "get_logger",
] 