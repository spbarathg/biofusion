"""
TIMESCALEDB DATA ACCESS LAYER
============================

Production-ready TimescaleDB integration for high-frequency trading data storage.
Replaces SQLite with time-series optimized database for better performance and scalability.

Features:
- Async PostgreSQL/TimescaleDB connections with asyncpg
- Hypertable management for time-series data
- Batch insertion for high-frequency writes
- Connection pooling and retry logic
- Data retention policies
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import asyncpg
import numpy as np
from asyncpg import Pool, Connection

from worker_ant_v1.utils.logger import setup_logger


@dataclass
class DatabaseConfig:
    """TimescaleDB database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "antbot_trading"
    username: str = "antbot"
    password: str = ""
    pool_min_size: int = 10
    pool_max_size: int = 20
    command_timeout: float = 60.0
    statement_cache_size: int = 100
    max_cached_statement_lifetime: int = 300
    ssl_mode: str = "prefer"


@dataclass  
class TradeRecord:
    """Enhanced trade record for TimescaleDB storage"""
    # Time-series primary key
    timestamp: datetime
    
    # Trade identification
    trade_id: str
    session_id: str
    wallet_id: str
    
    # Token information
    token_address: str
    token_symbol: str
    token_name: Optional[str] = None
    
    # Trade details
    trade_type: str  # BUY, SELL
    success: bool
    amount_sol: float
    amount_tokens: float
    price: float
    slippage_percent: float
    
    # Performance metrics
    latency_ms: int
    gas_cost_sol: float = 0.0
    rpc_cost_sol: float = 0.0
    api_cost_sol: float = 0.0
    
    # P&L metrics
    profit_loss_sol: Optional[float] = None
    profit_loss_percent: Optional[float] = None
    hold_time_seconds: Optional[int] = None
    
    # Technical details
    tx_signature_hash: Optional[str] = None
    retry_count: int = 0
    exit_reason: Optional[str] = None
    error_message: Optional[str] = None
    
    # Market context
    market_cap_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    price_change_24h_percent: Optional[float] = None
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemEvent:
    """System event record for TimescaleDB"""
    timestamp: datetime
    event_id: str
    event_type: str
    component: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    event_data: Dict[str, Any]
    session_id: Optional[str] = None
    wallet_id: Optional[str] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class PerformanceMetric:
    """Performance metric record for TimescaleDB"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    component: str
    labels: Dict[str, str]
    aggregation_period: Optional[str] = None  # 1m, 5m, 1h, etc.
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CallerProfile:
    """Caller intelligence profile for TimescaleDB"""
    timestamp: datetime
    caller_id: str
    username: str
    platform: str
    
    # Profile metrics
    account_age_days: int
    follower_count: int
    verified_account: bool
    total_calls: int
    successful_calls: int
    success_rate: float
    avg_profit_percent: float
    trust_score: float
    
    # Risk assessment
    credibility_level: str
    manipulation_risk: str
    risk_indicators: List[str]
    
    # Updated fields
    first_seen: datetime
    last_seen: datetime
    
    # Metadata
    profile_data: Optional[Dict[str, Any]] = None


class TimescaleDBManager:
    """TimescaleDB connection and operation manager"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = setup_logger("TimescaleDBManager")
        self.pool: Optional[Pool] = None
        self.initialized = False
        
        # Batch configuration
        self.batch_size = 100
        self.batch_timeout = 5.0
        
        # Queues for batch operations
        self.trade_queue: asyncio.Queue = asyncio.Queue()
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.metric_queue: asyncio.Queue = asyncio.Queue()
        
        # Background tasks
        self._batch_tasks: List[asyncio.Task] = []
        self._running = False

    async def initialize(self) -> bool:
        """Initialize TimescaleDB connection pool and schema"""
        try:
            self.logger.info("🗄️ Initializing TimescaleDB connection...")
            
            # Create connection pool
            await self._create_connection_pool()
            
            # Initialize database schema
            await self._initialize_schema()
            
            # Start batch processing tasks
            await self._start_batch_processors()
            
            self.initialized = True
            self.logger.info("✅ TimescaleDB initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ TimescaleDB initialization failed: {e}")
            return False

    async def _create_connection_pool(self):
        """Create asyncpg connection pool"""
        dsn = (
            f"postgresql://{self.config.username}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
            f"?sslmode={self.config.ssl_mode}"
        )
        
        self.pool = await asyncpg.create_pool(
            dsn,
            min_size=self.config.pool_min_size,
            max_size=self.config.pool_max_size,
            command_timeout=self.config.command_timeout,
            statement_cache_size=self.config.statement_cache_size,
            max_cached_statement_lifetime=self.config.max_cached_statement_lifetime
        )

    async def _initialize_schema(self):
        """Initialize TimescaleDB schema with hypertables"""
        async with self.pool.acquire() as conn:
            # Enable TimescaleDB extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            
            # Create trades hypertable
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    timestamp TIMESTAMPTZ NOT NULL,
                    trade_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    wallet_id TEXT NOT NULL,
                    token_address TEXT NOT NULL,
                    token_symbol TEXT,
                    token_name TEXT,
                    trade_type TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    amount_sol DOUBLE PRECISION,
                    amount_tokens DOUBLE PRECISION,
                    price DOUBLE PRECISION,
                    slippage_percent DOUBLE PRECISION,
                    latency_ms INTEGER,
                    gas_cost_sol DOUBLE PRECISION DEFAULT 0,
                    rpc_cost_sol DOUBLE PRECISION DEFAULT 0,
                    api_cost_sol DOUBLE PRECISION DEFAULT 0,
                    profit_loss_sol DOUBLE PRECISION,
                    profit_loss_percent DOUBLE PRECISION,
                    hold_time_seconds INTEGER,
                    tx_signature_hash TEXT,
                    retry_count INTEGER DEFAULT 0,
                    exit_reason TEXT,
                    error_message TEXT,
                    market_cap_usd DOUBLE PRECISION,
                    volume_24h_usd DOUBLE PRECISION,
                    price_change_24h_percent DOUBLE PRECISION,
                    metadata JSONB,
                    PRIMARY KEY (timestamp, trade_id)
                );
            """)
            
            # Create hypertable for trades
            await conn.execute("""
                SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE);
            """)
            
            # Create system_events hypertable
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_events (
                    timestamp TIMESTAMPTZ NOT NULL,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    event_data JSONB,
                    session_id TEXT,
                    wallet_id TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time TIMESTAMPTZ,
                    PRIMARY KEY (timestamp, event_id)
                );
            """)
            
            await conn.execute("""
                SELECT create_hypertable('system_events', 'timestamp', if_not_exists => TRUE);
            """)
            
            # Create performance_metrics hypertable
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    timestamp TIMESTAMPTZ NOT NULL,
                    metric_name TEXT NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    unit TEXT,
                    component TEXT NOT NULL,
                    labels JSONB,
                    aggregation_period TEXT,
                    metadata JSONB,
                    PRIMARY KEY (timestamp, metric_name, component)
                );
            """)
            
            await conn.execute("""
                SELECT create_hypertable('performance_metrics', 'timestamp', if_not_exists => TRUE);
            """)
            
            # Create caller_profiles table (regular table, not hypertable)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS caller_profiles (
                    timestamp TIMESTAMPTZ NOT NULL,
                    caller_id TEXT NOT NULL,
                    username TEXT,
                    platform TEXT NOT NULL,
                    account_age_days INTEGER,
                    follower_count INTEGER,
                    verified_account BOOLEAN,
                    total_calls INTEGER,
                    successful_calls INTEGER,
                    success_rate DOUBLE PRECISION,
                    avg_profit_percent DOUBLE PRECISION,
                    trust_score DOUBLE PRECISION,
                    credibility_level TEXT,
                    manipulation_risk TEXT,
                    risk_indicators TEXT[],
                    first_seen TIMESTAMPTZ,
                    last_seen TIMESTAMPTZ,
                    profile_data JSONB,
                    PRIMARY KEY (timestamp, caller_id)
                );
            """)
            
            await conn.execute("""
                SELECT create_hypertable('caller_profiles', 'timestamp', if_not_exists => TRUE);
            """)
            
            # Create indexes for better query performance
            await self._create_indexes(conn)
            
            # Set up data retention policies
            await self._setup_retention_policies(conn)

    async def _create_indexes(self, conn: Connection):
        """Create indexes for better query performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_trades_token_address ON trades (token_address, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_trades_wallet_id ON trades (wallet_id, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_trades_session_id ON trades (session_id, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_trades_success ON trades (success, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_trades_type ON trades (trade_type, timestamp DESC);",
            
            "CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events (event_type, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_system_events_component ON system_events (component, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_system_events_severity ON system_events (severity, timestamp DESC);",
            
            "CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics (metric_name, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_metrics_component ON performance_metrics (component, timestamp DESC);",
            
            "CREATE INDEX IF NOT EXISTS idx_caller_profiles_id ON caller_profiles (caller_id, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_caller_profiles_platform ON caller_profiles (platform, timestamp DESC);"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)

    async def _setup_retention_policies(self, conn: Connection):
        """Set up data retention policies"""
        # Keep trades for 1 year
        await conn.execute("""
            SELECT add_retention_policy('trades', INTERVAL '1 year', if_not_exists => TRUE);
        """)
        
        # Keep system events for 6 months
        await conn.execute("""
            SELECT add_retention_policy('system_events', INTERVAL '6 months', if_not_exists => TRUE);
        """)
        
        # Keep performance metrics for 3 months
        await conn.execute("""
            SELECT add_retention_policy('performance_metrics', INTERVAL '3 months', if_not_exists => TRUE);
        """)
        
        # Keep caller profiles for 1 year
        await conn.execute("""
            SELECT add_retention_policy('caller_profiles', INTERVAL '1 year', if_not_exists => TRUE);
        """)

    async def _start_batch_processors(self):
        """Start background batch processing tasks"""
        self._running = True
        
        self._batch_tasks = [
            asyncio.create_task(self._process_trade_batch()),
            asyncio.create_task(self._process_event_batch()),
            asyncio.create_task(self._process_metric_batch())
        ]

    async def _process_trade_batch(self):
        """Process trades in batches"""
        batch = []
        last_flush = asyncio.get_event_loop().time()
        
        while self._running:
            try:
                # Wait for trade or timeout
                try:
                    trade = await asyncio.wait_for(
                        self.trade_queue.get(), timeout=self.batch_timeout
                    )
                    batch.append(trade)
                except asyncio.TimeoutError:
                    pass
                
                current_time = asyncio.get_event_loop().time()
                
                # Flush batch if full or timeout reached
                if (len(batch) >= self.batch_size or 
                    (batch and current_time - last_flush >= self.batch_timeout)):
                    
                    await self._flush_trade_batch(batch)
                    batch = []
                    last_flush = current_time
                    
            except Exception as e:
                self.logger.error(f"Error in trade batch processor: {e}")
                await asyncio.sleep(1)
        
        # Flush remaining batch
        if batch:
            await self._flush_trade_batch(batch)

    async def _flush_trade_batch(self, batch: List[TradeRecord]):
        """Flush a batch of trades to TimescaleDB"""
        if not batch:
            return
            
        try:
            async with self.pool.acquire() as conn:
                # Prepare data for batch insert
                data = []
                for trade in batch:
                    data.append((
                        trade.timestamp,
                        trade.trade_id,
                        trade.session_id,
                        trade.wallet_id,
                        trade.token_address,
                        trade.token_symbol,
                        trade.token_name,
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
                        trade.market_cap_usd,
                        trade.volume_24h_usd,
                        trade.price_change_24h_percent,
                        json.dumps(trade.metadata) if trade.metadata else None
                    ))
                
                # Batch insert
                await conn.executemany("""
                    INSERT INTO trades (
                        timestamp, trade_id, session_id, wallet_id, token_address,
                        token_symbol, token_name, trade_type, success, amount_sol,
                        amount_tokens, price, slippage_percent, latency_ms,
                        gas_cost_sol, rpc_cost_sol, api_cost_sol, profit_loss_sol,
                        profit_loss_percent, hold_time_seconds, tx_signature_hash,
                        retry_count, exit_reason, error_message, market_cap_usd,
                        volume_24h_usd, price_change_24h_percent, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28)
                """, data)
                
                self.logger.debug(f"✅ Flushed {len(batch)} trades to TimescaleDB")
                
        except Exception as e:
            self.logger.error(f"❌ Error flushing trade batch: {e}")

    async def _process_event_batch(self):
        """Process system events in batches"""
        batch = []
        last_flush = asyncio.get_event_loop().time()
        
        while self._running:
            try:
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(), timeout=self.batch_timeout
                    )
                    batch.append(event)
                except asyncio.TimeoutError:
                    pass
                
                current_time = asyncio.get_event_loop().time()
                
                if (len(batch) >= self.batch_size or 
                    (batch and current_time - last_flush >= self.batch_timeout)):
                    
                    await self._flush_event_batch(batch)
                    batch = []
                    last_flush = current_time
                    
            except Exception as e:
                self.logger.error(f"Error in event batch processor: {e}")
                await asyncio.sleep(1)
        
        if batch:
            await self._flush_event_batch(batch)

    async def _flush_event_batch(self, batch: List[SystemEvent]):
        """Flush a batch of system events to TimescaleDB"""
        if not batch:
            return
            
        try:
            async with self.pool.acquire() as conn:
                data = []
                for event in batch:
                    data.append((
                        event.timestamp,
                        event.event_id,
                        event.event_type,
                        event.component,
                        event.severity,
                        event.message,
                        json.dumps(event.event_data),
                        event.session_id,
                        event.wallet_id,
                        event.resolved,
                        event.resolution_time
                    ))
                
                await conn.executemany("""
                    INSERT INTO system_events (
                        timestamp, event_id, event_type, component, severity,
                        message, event_data, session_id, wallet_id, resolved,
                        resolution_time
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, data)
                
                self.logger.debug(f"✅ Flushed {len(batch)} events to TimescaleDB")
                
        except Exception as e:
            self.logger.error(f"❌ Error flushing event batch: {e}")

    async def _process_metric_batch(self):
        """Process performance metrics in batches"""
        batch = []
        last_flush = asyncio.get_event_loop().time()
        
        while self._running:
            try:
                try:
                    metric = await asyncio.wait_for(
                        self.metric_queue.get(), timeout=self.batch_timeout
                    )
                    batch.append(metric)
                except asyncio.TimeoutError:
                    pass
                
                current_time = asyncio.get_event_loop().time()
                
                if (len(batch) >= self.batch_size or 
                    (batch and current_time - last_flush >= self.batch_timeout)):
                    
                    await self._flush_metric_batch(batch)
                    batch = []
                    last_flush = current_time
                    
            except Exception as e:
                self.logger.error(f"Error in metric batch processor: {e}")
                await asyncio.sleep(1)
        
        if batch:
            await self._flush_metric_batch(batch)

    async def _flush_metric_batch(self, batch: List[PerformanceMetric]):
        """Flush a batch of performance metrics to TimescaleDB"""
        if not batch:
            return
            
        try:
            async with self.pool.acquire() as conn:
                data = []
                for metric in batch:
                    data.append((
                        metric.timestamp,
                        metric.metric_name,
                        metric.value,
                        metric.unit,
                        metric.component,
                        json.dumps(metric.labels),
                        metric.aggregation_period,
                        json.dumps(metric.metadata) if metric.metadata else None
                    ))
                
                await conn.executemany("""
                    INSERT INTO performance_metrics (
                        timestamp, metric_name, value, unit, component,
                        labels, aggregation_period, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (timestamp, metric_name, component) 
                    DO UPDATE SET value = EXCLUDED.value, labels = EXCLUDED.labels
                """, data)
                
                self.logger.debug(f"✅ Flushed {len(batch)} metrics to TimescaleDB")
                
        except Exception as e:
            self.logger.error(f"❌ Error flushing metric batch: {e}")

    # Public API methods
    async def insert_trade(self, trade: TradeRecord):
        """Insert a trade record (queued for batch processing)"""
        await self.trade_queue.put(trade)

    async def insert_system_event(self, event: SystemEvent):
        """Insert a system event (queued for batch processing)"""
        await self.event_queue.put(event)

    async def insert_performance_metric(self, metric: PerformanceMetric):
        """Insert a performance metric (queued for batch processing)"""
        await self.metric_queue.put(metric)

    async def insert_caller_profile(self, profile: CallerProfile):
        """Insert caller profile immediately (low frequency)"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO caller_profiles (
                        timestamp, caller_id, username, platform, account_age_days,
                        follower_count, verified_account, total_calls, successful_calls,
                        success_rate, avg_profit_percent, trust_score, credibility_level,
                        manipulation_risk, risk_indicators, first_seen, last_seen,
                        profile_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                    ON CONFLICT (timestamp, caller_id) 
                    DO UPDATE SET 
                        username = EXCLUDED.username,
                        total_calls = EXCLUDED.total_calls,
                        successful_calls = EXCLUDED.successful_calls,
                        success_rate = EXCLUDED.success_rate,
                        trust_score = EXCLUDED.trust_score,
                        last_seen = EXCLUDED.last_seen
                """, 
                profile.timestamp, profile.caller_id, profile.username, profile.platform,
                profile.account_age_days, profile.follower_count, profile.verified_account,
                profile.total_calls, profile.successful_calls, profile.success_rate,
                profile.avg_profit_percent, profile.trust_score, profile.credibility_level,
                profile.manipulation_risk, profile.risk_indicators, profile.first_seen,
                profile.last_seen, json.dumps(profile.profile_data) if profile.profile_data else None
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error inserting caller profile: {e}")

    async def query_trades(self, 
                          start_time: datetime, 
                          end_time: datetime,
                          wallet_id: Optional[str] = None,
                          token_address: Optional[str] = None,
                          success_only: bool = False,
                          limit: int = 1000) -> List[Dict[str, Any]]:
        """Query trades within time range with filters"""
        try:
            async with self.pool.acquire() as conn:
                conditions = ["timestamp >= $1", "timestamp <= $2"]
                params = [start_time, end_time]
                param_count = 2
                
                if wallet_id:
                    param_count += 1
                    conditions.append(f"wallet_id = ${param_count}")
                    params.append(wallet_id)
                
                if token_address:
                    param_count += 1
                    conditions.append(f"token_address = ${param_count}")
                    params.append(token_address)
                
                if success_only:
                    conditions.append("success = TRUE")
                
                param_count += 1
                params.append(limit)
                
                query = f"""
                    SELECT * FROM trades 
                    WHERE {' AND '.join(conditions)}
                    ORDER BY timestamp DESC 
                    LIMIT ${param_count}
                """
                
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error querying trades: {e}")
            return []

    async def get_performance_summary(self, 
                                    start_time: datetime, 
                                    end_time: datetime) -> Dict[str, Any]:
        """Get performance summary for time period"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE success = TRUE) as successful_trades,
                        COUNT(*) FILTER (WHERE success = FALSE) as failed_trades,
                        COALESCE(SUM(profit_loss_sol), 0) as total_profit_sol,
                        COALESCE(AVG(profit_loss_sol), 0) as avg_profit_sol,
                        COALESCE(AVG(latency_ms), 0) as avg_latency_ms,
                        COALESCE(AVG(slippage_percent), 0) as avg_slippage_percent,
                        COUNT(DISTINCT wallet_id) as active_wallets,
                        COUNT(DISTINCT token_address) as unique_tokens
                    FROM trades 
                    WHERE timestamp >= $1 AND timestamp <= $2
                """, start_time, end_time)
                
                return dict(result) if result else {}
                
        except Exception as e:
            self.logger.error(f"❌ Error getting performance summary: {e}")
            return {}

    async def shutdown(self):
        """Shutdown the database manager"""
        self.logger.info("🛑 Shutting down TimescaleDB manager...")
        
        # Stop batch processors
        self._running = False
        
        # Cancel background tasks
        for task in self._batch_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._batch_tasks, return_exceptions=True)
        
        # Flush remaining data
        await self._flush_remaining_data()
        
        # Close connection pool
        if self.pool:
            await self.pool.close()
        
        self.logger.info("✅ TimescaleDB manager shutdown complete")

    async def _flush_remaining_data(self):
        """Flush any remaining data in queues"""
        try:
            # Flush remaining trades
            trade_batch = []
            while not self.trade_queue.empty():
                try:
                    trade = self.trade_queue.get_nowait()
                    trade_batch.append(trade)
                except asyncio.QueueEmpty:
                    break
            
            if trade_batch:
                await self._flush_trade_batch(trade_batch)
            
            # Flush remaining events
            event_batch = []
            while not self.event_queue.empty():
                try:
                    event = self.event_queue.get_nowait()
                    event_batch.append(event)
                except asyncio.QueueEmpty:
                    break
            
            if event_batch:
                await self._flush_event_batch(event_batch)
            
            # Flush remaining metrics
            metric_batch = []
            while not self.metric_queue.empty():
                try:
                    metric = self.metric_queue.get_nowait()
                    metric_batch.append(metric)
                except asyncio.QueueEmpty:
                    break
            
            if metric_batch:
                await self._flush_metric_batch(metric_batch)
                
        except Exception as e:
            self.logger.error(f"❌ Error flushing remaining data: {e}")


# Global TimescaleDB manager instance
_db_manager: Optional[TimescaleDBManager] = None


def get_database_config() -> DatabaseConfig:
    """Get database configuration from environment variables"""
    return DatabaseConfig(
        host=os.getenv("TIMESCALEDB_HOST", "localhost"),
        port=int(os.getenv("TIMESCALEDB_PORT", "5432")),
        database=os.getenv("TIMESCALEDB_DATABASE", "antbot_trading"),
        username=os.getenv("TIMESCALEDB_USERNAME", "antbot"),
        password=os.getenv("TIMESCALEDB_PASSWORD", ""),
        pool_min_size=int(os.getenv("TIMESCALEDB_POOL_MIN_SIZE", "10")),
        pool_max_size=int(os.getenv("TIMESCALEDB_POOL_MAX_SIZE", "20")),
        ssl_mode=os.getenv("TIMESCALEDB_SSL_MODE", "prefer")
    )


async def get_database_manager() -> TimescaleDBManager:
    """Get or create global database manager instance"""
    global _db_manager
    
    if _db_manager is None:
        config = get_database_config()
        _db_manager = TimescaleDBManager(config)
        await _db_manager.initialize()
    
    return _db_manager


async def shutdown_database_manager():
    """Shutdown global database manager"""
    global _db_manager
    
    if _db_manager:
        await _db_manager.shutdown()
        _db_manager = None 