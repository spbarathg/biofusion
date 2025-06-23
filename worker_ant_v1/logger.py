"""
Logging and KPI Tracking for Worker Ant V1
==========================================

Handles all trade logging, performance metrics, and result storage.
"""

import time
import sqlite3
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict

from .config import monitoring_config


@dataclass
class TradeResult:
    """Single trade result record"""
    
    timestamp: str
    token_address: str
    token_symbol: str
    trade_type: str  # 'BUY' or 'SELL'
    amount_sol: float
    amount_tokens: float
    price: float
    slippage_percent: float
    latency_ms: int
    tx_signature: str
    success: bool
    error_message: Optional[str] = None
    
    # Performance metrics (filled on sell)
    profit_loss_sol: Optional[float] = None
    profit_loss_percent: Optional[float] = None
    hold_time_seconds: Optional[int] = None


@dataclass
class SessionMetrics:
    """Session-level performance metrics"""
    
    session_start: str
    total_trades: int = 0
    successful_trades: int = 0
    total_profit_sol: float = 0.0
    total_volume_sol: float = 0.0
    win_rate_percent: float = 0.0
    average_profit_percent: float = 0.0
    max_drawdown_percent: float = 0.0
    average_slippage_percent: float = 0.0
    average_latency_ms: float = 0.0


class TradingLogger:
    """Main logging and metrics tracking class"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_database()
        self.session_start = datetime.utcnow().isoformat()
        self.daily_loss_sol = 0.0
        self.last_trade_time = 0
        self.active_positions: Dict[str, TradeResult] = {}
        
    def setup_logging(self):
        """Configure logging to file and console"""
        
        # Create logs directory
        log_dir = Path(monitoring_config.log_file).parent
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, monitoring_config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(monitoring_config.log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('WorkerAntV1')
        self.logger.info("Trading logger initialized")
        
    def setup_database(self):
        """Initialize SQLite database for trade storage"""
        
        db_dir = Path(monitoring_config.db_path).parent
        db_dir.mkdir(exist_ok=True)
        
        self.conn = sqlite3.connect(monitoring_config.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                token_address TEXT NOT NULL,
                token_symbol TEXT,
                trade_type TEXT NOT NULL,
                amount_sol REAL NOT NULL,
                amount_tokens REAL NOT NULL,
                price REAL NOT NULL,
                slippage_percent REAL NOT NULL,
                latency_ms INTEGER NOT NULL,
                tx_signature TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                profit_loss_sol REAL,
                profit_loss_percent REAL,
                hold_time_seconds INTEGER
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date TEXT PRIMARY KEY,
                total_trades INTEGER,
                successful_trades INTEGER,
                total_profit_sol REAL,
                win_rate_percent REAL,
                max_drawdown_percent REAL
            )
        """)
        
        self.conn.commit()
        self.logger.info(f"Database initialized: {monitoring_config.db_path}")
        
    def log_trade(self, trade: TradeResult):
        """Log a single trade to database and file"""
        
        # Store to database
        self.conn.execute("""
            INSERT INTO trades (
                timestamp, token_address, token_symbol, trade_type,
                amount_sol, amount_tokens, price, slippage_percent,
                latency_ms, tx_signature, success, error_message,
                profit_loss_sol, profit_loss_percent, hold_time_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.timestamp, trade.token_address, trade.token_symbol, trade.trade_type,
            trade.amount_sol, trade.amount_tokens, trade.price, trade.slippage_percent,
            trade.latency_ms, trade.tx_signature, trade.success, trade.error_message,
            trade.profit_loss_sol, trade.profit_loss_percent, trade.hold_time_seconds
        ))
        self.conn.commit()
        
        # Log to file
        if trade.success:
            self.logger.info(f"TRADE: {trade.trade_type} {trade.token_symbol} "
                           f"Amount: {trade.amount_sol:.4f} SOL "
                           f"Price: {trade.price:.8f} "
                           f"Slippage: {trade.slippage_percent:.2f}% "
                           f"Latency: {trade.latency_ms}ms")
            if trade.profit_loss_percent is not None:
                self.logger.info(f"P&L: {trade.profit_loss_percent:.2f}% "
                               f"({trade.profit_loss_sol:.4f} SOL)")
        else:
            self.logger.error(f"FAILED TRADE: {trade.trade_type} {trade.token_symbol} "
                            f"Error: {trade.error_message}")
        
        # Update tracking
        if trade.trade_type == 'BUY' and trade.success:
            self.active_positions[trade.token_address] = trade
        elif trade.trade_type == 'SELL' and trade.success:
            if trade.token_address in self.active_positions:
                del self.active_positions[trade.token_address]
            if trade.profit_loss_sol:
                self.daily_loss_sol += min(0, trade.profit_loss_sol)
                
    def check_safety_limits(self) -> bool:
        """Check if we've hit daily loss limits or trade frequency limits"""
        
        from .config import config
        
        # Check daily loss limit
        if abs(self.daily_loss_sol) >= config.max_daily_loss_sol:
            self.logger.warning(f"Daily loss limit reached: {self.daily_loss_sol:.4f} SOL")
            return False
            
        # Check trade frequency
        current_time = time.time()
        if (current_time - self.last_trade_time) < config.min_time_between_trades_seconds:
            return False
            
        return True
        
    def update_last_trade_time(self):
        """Update the last trade timestamp"""
        self.last_trade_time = time.time()
        
    def get_session_metrics(self) -> SessionMetrics:
        """Calculate current session performance metrics"""
        
        # Query trades from this session
        cursor = self.conn.execute("""
            SELECT * FROM trades WHERE timestamp >= ? AND success = 1
        """, (self.session_start,))
        
        trades = cursor.fetchall()
        
        if not trades:
            return SessionMetrics(session_start=self.session_start)
            
        # Calculate metrics
        total_trades = len(trades)
        successful_trades = len([t for t in trades if t[10]])  # success column
        
        profit_trades = [t for t in trades if t[13] and t[13] > 0]  # profit_loss_sol column
        win_rate = (len(profit_trades) / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum(t[13] for t in trades if t[13]) or 0
        total_volume = sum(t[4] for t in trades)  # amount_sol column
        
        avg_profit = sum(t[14] for t in trades if t[14]) / len(trades) if trades else 0
        avg_slippage = sum(t[7] for t in trades) / len(trades) if trades else 0
        avg_latency = sum(t[8] for t in trades) / len(trades) if trades else 0
        
        # Calculate max drawdown (simplified)
        running_pnl = 0
        peak = 0
        max_drawdown = 0
        
        for trade in trades:
            if trade[13]:  # profit_loss_sol
                running_pnl += trade[13]
                if running_pnl > peak:
                    peak = running_pnl
                drawdown = (peak - running_pnl) / peak * 100 if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        
        return SessionMetrics(
            session_start=self.session_start,
            total_trades=total_trades,
            successful_trades=successful_trades,
            total_profit_sol=total_profit,
            total_volume_sol=total_volume,
            win_rate_percent=win_rate,
            average_profit_percent=avg_profit,
            max_drawdown_percent=max_drawdown,
            average_slippage_percent=avg_slippage,
            average_latency_ms=avg_latency
        )
        
    def get_recent_trades(self, limit: int = 10) -> List[TradeResult]:
        """Get most recent trades"""
        
        cursor = self.conn.execute("""
            SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        
        trades = []
        for row in cursor.fetchall():
            trades.append(TradeResult(
                timestamp=row[1],
                token_address=row[2],
                token_symbol=row[3],
                trade_type=row[4],
                amount_sol=row[5],
                amount_tokens=row[6],
                price=row[7],
                slippage_percent=row[8],
                latency_ms=row[9],
                tx_signature=row[10],
                success=row[11],
                error_message=row[12],
                profit_loss_sol=row[13],
                profit_loss_percent=row[14],
                hold_time_seconds=row[15]
            ))
            
        return trades
        
    def export_metrics(self, filepath: str):
        """Export current metrics to JSON file"""
        
        metrics = self.get_session_metrics()
        recent_trades = self.get_recent_trades(20)
        
        export_data = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'session_metrics': asdict(metrics),
            'recent_trades': [asdict(trade) for trade in recent_trades],
            'active_positions': {addr: asdict(trade) for addr, trade in self.active_positions.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
        self.logger.info(f"Metrics exported to {filepath}")
        
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()


# Global logger instance
trading_logger = TradingLogger() 