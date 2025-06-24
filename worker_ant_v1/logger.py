"""
Optimized Trade Logger with Cost Tracking
========================================

High-performance logging system with batch processing, hourly summaries,
and comprehensive cost tracking for deployment readiness.
"""

import sqlite3
import logging
import time
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Thread, Lock
import os

from worker_ant_v1.config import monitoring_config, trading_config, deployment_config

@dataclass
class TradeRecord:
    """Enhanced trade record with cost tracking"""
    
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
    
    # Technical details
    tx_signature: Optional[str] = None
    retry_count: int = 0
    exit_reason: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class HourlyReport:
    """Hourly performance summary"""
    
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


class OptimizedTradeLogger:
    """High-performance trade logger with cost optimization"""
    
    def __init__(self):
        self.db_path = monitoring_config.db_path
        self.hourly_log_path = monitoring_config.hourly_summary_file
        self.conn: Optional[sqlite3.Connection] = None
        
        # Batch processing
        self.batch_queue: List[TradeRecord] = []
        self.batch_lock = Lock()
        self.last_batch_write = time.time()
        
        # Performance tracking
        self.session_start = datetime.utcnow().isoformat()
        self.active_positions: Dict[str, TradeRecord] = {}
        self.hourly_data: Dict[str, List[TradeRecord]] = {}
        
        # Cost tracking
        self.daily_costs = {
            'gas': 0.0,
            'rpc': 0.0,
            'api': 0.0,
            'total': 0.0
        }
        
        # Logger setup
        self.logger = logging.getLogger("OptimizedLogger")
        self._setup_logger()
        
        # Background processing
        self.batch_processor_running = True
        if monitoring_config.batch_logs:
            self.batch_thread = Thread(target=self._batch_processor, daemon=True)
            self.batch_thread.start()
        
        # Hourly reporting
        if monitoring_config.enable_hourly_summary:
            self.hourly_thread = Thread(target=self._hourly_reporter, daemon=True)
            self.hourly_thread.start()
    
    def _setup_logger(self):
        """Setup optimized logging configuration"""
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, monitoring_config.log_level))
        console_handler.setFormatter(formatter)
        
        # File handler (only for important events)
        if monitoring_config.log_file:
            file_handler = logging.FileHandler(monitoring_config.log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.addHandler(console_handler)
        self.logger.setLevel(getattr(logging, monitoring_config.log_level))
        
        # Reduce spam from external libraries
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("solana").setLevel(logging.WARNING)
    
    async def setup_database(self):
        """Initialize optimized database schema"""
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Create optimized table schema
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
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
                tx_signature TEXT,
                retry_count INTEGER DEFAULT 0,
                exit_reason TEXT,
                error_message TEXT
            )
        """)
        
        # Create indexes for performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_token ON trades(token_address)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON trades(trade_type)")
        
        # Create hourly summary table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS hourly_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hour_start TEXT NOT NULL UNIQUE,
                total_trades INTEGER,
                successful_trades INTEGER,
                buy_trades INTEGER,
                sell_trades INTEGER,
                win_rate REAL,
                total_profit_sol REAL,
                total_volume_sol REAL,
                average_hold_time REAL,
                average_profit_percent REAL,
                total_gas_costs REAL,
                total_rpc_costs REAL,
                total_api_costs REAL,
                cost_per_trade REAL,
                profit_after_costs REAL,
                max_drawdown_percent REAL,
                largest_loss_sol REAL,
                largest_win_sol REAL,
                sharpe_ratio REAL
            )
        """)
        
        self.conn.commit()
        self.logger.info("Optimized database initialized")
    
    def log_trade(self, trade_record: TradeRecord):
        """Log trade with batch processing"""
        
        # Update cost tracking
        self._update_cost_tracking(trade_record)
        
        # Add to batch queue
        if monitoring_config.batch_logs:
            with self.batch_lock:
                self.batch_queue.append(trade_record)
                
                # Force write if batch is full
                if len(self.batch_queue) >= monitoring_config.batch_size:
                    self._write_batch_to_db()
        else:
            # Immediate write
            self._write_single_record(trade_record)
        
        # Update active positions tracking
        if trade_record.trade_type == "BUY" and trade_record.success:
            self.active_positions[trade_record.token_address] = trade_record
        elif trade_record.trade_type == "SELL" and trade_record.success:
            if trade_record.token_address in self.active_positions:
                del self.active_positions[trade_record.token_address]
        
        # Log important events immediately
        if trade_record.success:
            if trade_record.trade_type == "BUY":
                self.logger.info(
                    f"BUY {trade_record.token_symbol}: "
                    f"{trade_record.amount_sol:.4f} SOL @ {trade_record.price:.6f} "
                    f"({trade_record.latency_ms}ms)"
                )
            else:  # SELL
                profit_str = ""
                if trade_record.profit_loss_percent is not None:
                    profit_str = f" | {trade_record.profit_loss_percent:+.2f}%"
                
                self.logger.info(
                    f"SELL {trade_record.token_symbol}: "
                    f"{trade_record.amount_sol:.4f} SOL{profit_str} "
                    f"({trade_record.exit_reason})"
                )
        else:
            self.logger.warning(
                f"FAILED {trade_record.trade_type} {trade_record.token_symbol}: "
                f"{trade_record.error_message}"
            )
    
    def _update_cost_tracking(self, trade: TradeRecord):
        """Update daily cost tracking"""
        
        self.daily_costs['gas'] += trade.gas_cost_sol
        self.daily_costs['rpc'] += trade.rpc_cost_sol  
        self.daily_costs['api'] += trade.api_cost_sol
        self.daily_costs['total'] = (
            self.daily_costs['gas'] + 
            self.daily_costs['rpc'] + 
            self.daily_costs['api']
        )
        
        # Alert if costs are too high
        if (self.daily_costs['total'] >= monitoring_config.cost_alert_threshold_sol):
            self.logger.warning(
                f"Daily costs high: {self.daily_costs['total']:.4f} SOL"
            )
    
    def _write_single_record(self, trade: TradeRecord):
        """Write single trade record to database"""
        
        if not self.conn:
            return
            
        try:
            self.conn.execute("""
                INSERT INTO trades (
                    timestamp, token_address, token_symbol, trade_type, success,
                    amount_sol, amount_tokens, price, slippage_percent, latency_ms,
                    gas_cost_sol, rpc_cost_sol, api_cost_sol,
                    profit_loss_sol, profit_loss_percent, hold_time_seconds,
                    tx_signature, retry_count, exit_reason, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.timestamp, trade.token_address, trade.token_symbol,
                trade.trade_type, trade.success, trade.amount_sol, trade.amount_tokens,
                trade.price, trade.slippage_percent, trade.latency_ms,
                trade.gas_cost_sol, trade.rpc_cost_sol, trade.api_cost_sol,
                trade.profit_loss_sol, trade.profit_loss_percent, trade.hold_time_seconds,
                trade.tx_signature, trade.retry_count, trade.exit_reason, trade.error_message
            ))
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Database write error: {e}")
    
    def _write_batch_to_db(self):
        """Write batch of trades to database"""
        
        if not self.conn or not self.batch_queue:
            return
            
        try:
            # Prepare batch data
            batch_data = []
            for trade in self.batch_queue:
                batch_data.append((
                    trade.timestamp, trade.token_address, trade.token_symbol,
                    trade.trade_type, trade.success, trade.amount_sol, trade.amount_tokens,
                    trade.price, trade.slippage_percent, trade.latency_ms,
                    trade.gas_cost_sol, trade.rpc_cost_sol, trade.api_cost_sol,
                    trade.profit_loss_sol, trade.profit_loss_percent, trade.hold_time_seconds,
                    trade.tx_signature, trade.retry_count, trade.exit_reason, trade.error_message
                ))
            
            # Batch insert
            self.conn.executemany("""
                INSERT INTO trades (
                    timestamp, token_address, token_symbol, trade_type, success,
                    amount_sol, amount_tokens, price, slippage_percent, latency_ms,
                    gas_cost_sol, rpc_cost_sol, api_cost_sol,
                    profit_loss_sol, profit_loss_percent, hold_time_seconds,
                    tx_signature, retry_count, exit_reason, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch_data)
            
            self.conn.commit()
            
            self.logger.debug(f"Batch wrote {len(self.batch_queue)} trades to database")
            
            # Clear batch
            self.batch_queue.clear()
            self.last_batch_write = time.time()
            
        except Exception as e:
            self.logger.error(f"Batch write error: {e}")
    
    def _batch_processor(self):
        """Background batch processor"""
        
        while self.batch_processor_running:
            try:
                time.sleep(5)  # Check every 5 seconds
                
                current_time = time.time()
                
                # Write batch if timeout reached or queue is full
                with self.batch_lock:
                    if (self.batch_queue and 
                        (len(self.batch_queue) >= monitoring_config.batch_size or
                         current_time - self.last_batch_write >= monitoring_config.batch_timeout_seconds)):
                        
                        self._write_batch_to_db()
                        
            except Exception as e:
                self.logger.error(f"Batch processor error: {e}")
    
    def _hourly_reporter(self):
        """Background hourly report generator"""
        
        while self.batch_processor_running:
            try:
                time.sleep(300)  # Check every 5 minutes
                
                current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
                last_hour = current_hour - timedelta(hours=1)
                
                # Generate report for completed hour
                self._generate_hourly_report(last_hour)
                
            except Exception as e:
                self.logger.error(f"Hourly reporter error: {e}")
    
    def _generate_hourly_report(self, hour_start: datetime):
        """Generate comprehensive hourly report"""
        
        if not self.conn:
            return
            
        hour_start_str = hour_start.isoformat()
        hour_end_str = (hour_start + timedelta(hours=1)).isoformat()
        
        try:
            # Get trades for the hour
            cursor = self.conn.execute("""
                SELECT * FROM trades 
                WHERE timestamp >= ? AND timestamp < ?
                ORDER BY timestamp
            """, (hour_start_str, hour_end_str))
            
            trades = cursor.fetchall()
            
            if not trades:
                return
            
            # Calculate metrics
            report = self._calculate_hourly_metrics(trades, hour_start_str)
            
            # Save to database
            self._save_hourly_report(report)
            
            # Write to log file
            self._write_hourly_log(report)
            
            # Check for alerts
            self._check_hourly_alerts(report)
            
        except Exception as e:
            self.logger.error(f"Hourly report generation error: {e}")
    
    def _calculate_hourly_metrics(self, trades: List, hour_start: str) -> HourlyReport:
        """Calculate comprehensive hourly metrics"""
        
        total_trades = len(trades)
        successful_trades = sum(1 for t in trades if t[5])  # success column
        buy_trades = sum(1 for t in trades if t[4] == "BUY")  # trade_type column
        sell_trades = sum(1 for t in trades if t[4] == "SELL")
        
        # Performance metrics
        profitable_sells = [t for t in trades if t[4] == "SELL" and t[5] and t[14] and t[14] > 0]
        win_rate = (len(profitable_sells) / sell_trades * 100) if sell_trades > 0 else 0
        
        total_profit_sol = sum(t[14] for t in trades if t[14]) or 0
        total_volume_sol = sum(t[6] for t in trades if t[6]) or 0
        
        hold_times = [t[16] for t in trades if t[16]]
        average_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0
        
        profit_percentages = [t[15] for t in trades if t[15]]
        average_profit_percent = sum(profit_percentages) / len(profit_percentages) if profit_percentages else 0
        
        # Cost analysis
        total_gas_costs = sum(t[11] for t in trades if t[11]) or 0
        total_rpc_costs = sum(t[12] for t in trades if t[12]) or 0
        total_api_costs = sum(t[13] for t in trades if t[13]) or 0
        
        cost_per_trade = (total_gas_costs + total_rpc_costs + total_api_costs) / total_trades if total_trades > 0 else 0
        profit_after_costs = total_profit_sol - (total_gas_costs + total_rpc_costs + total_api_costs)
        
        # Risk metrics
        losses = [t[14] for t in trades if t[14] and t[14] < 0]
        wins = [t[14] for t in trades if t[14] and t[14] > 0]
        
        largest_loss_sol = min(losses) if losses else 0
        largest_win_sol = max(wins) if wins else 0
        
        # Simplified Sharpe ratio calculation
        if profit_percentages:
            returns_std = (sum((x - average_profit_percent) ** 2 for x in profit_percentages) / len(profit_percentages)) ** 0.5
            sharpe_ratio = average_profit_percent / returns_std if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Max drawdown (simplified)
        running_pnl = 0
        peak = 0
        max_drawdown = 0
        
        for trade in trades:
            if trade[14]:  # profit_loss_sol
                running_pnl += trade[14]
                if running_pnl > peak:
                    peak = running_pnl
                if peak > 0:
                    drawdown = (peak - running_pnl) / peak * 100
                    max_drawdown = max(max_drawdown, drawdown)
        
        return HourlyReport(
            hour_start=hour_start,
            total_trades=total_trades,
            successful_trades=successful_trades,
            buy_trades=buy_trades,
            sell_trades=sell_trades,
            win_rate=win_rate,
            total_profit_sol=total_profit_sol,
            total_volume_sol=total_volume_sol,
            average_hold_time=average_hold_time,
            average_profit_percent=average_profit_percent,
            total_gas_costs=total_gas_costs,
            total_rpc_costs=total_rpc_costs,
            total_api_costs=total_api_costs,
            cost_per_trade=cost_per_trade,
            profit_after_costs=profit_after_costs,
            max_drawdown_percent=max_drawdown,
            largest_loss_sol=largest_loss_sol,
            largest_win_sol=largest_win_sol,
            sharpe_ratio=sharpe_ratio
        )
    
    def _save_hourly_report(self, report: HourlyReport):
        """Save hourly report to database"""
        
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO hourly_reports (
                    hour_start, total_trades, successful_trades, buy_trades, sell_trades,
                    win_rate, total_profit_sol, total_volume_sol, average_hold_time,
                    average_profit_percent, total_gas_costs, total_rpc_costs, total_api_costs,
                    cost_per_trade, profit_after_costs, max_drawdown_percent,
                    largest_loss_sol, largest_win_sol, sharpe_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.hour_start, report.total_trades, report.successful_trades,
                report.buy_trades, report.sell_trades, report.win_rate,
                report.total_profit_sol, report.total_volume_sol, report.average_hold_time,
                report.average_profit_percent, report.total_gas_costs, report.total_rpc_costs,
                report.total_api_costs, report.cost_per_trade, report.profit_after_costs,
                report.max_drawdown_percent, report.largest_loss_sol, report.largest_win_sol,
                report.sharpe_ratio
            ))
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Hourly report save error: {e}")
    
    def _write_hourly_log(self, report: HourlyReport):
        """Write hourly summary to log file"""
        
        try:
            summary = (
                f"\n{'='*60}\n"
                f"HOURLY SUMMARY: {report.hour_start[:13]}:00\n"
                f"{'='*60}\n"
                f"Trades: {report.total_trades} ({report.successful_trades} successful)\n"
                f"Buy/Sell: {report.buy_trades}/{report.sell_trades}\n"
                f"Win Rate: {report.win_rate:.1f}%\n"
                f"Profit: {report.total_profit_sol:+.4f} SOL ({report.average_profit_percent:+.2f}% avg)\n"
                f"Volume: {report.total_volume_sol:.2f} SOL\n"
                f"Hold Time: {report.average_hold_time:.1f}s avg\n"
                f"Costs: {report.total_gas_costs + report.total_rpc_costs + report.total_api_costs:.4f} SOL\n"
                f"  - Gas: {report.total_gas_costs:.4f} SOL\n"
                f"  - RPC: {report.total_rpc_costs:.4f} SOL\n"
                f"  - API: {report.total_api_costs:.4f} SOL\n"
                f"Net Profit: {report.profit_after_costs:+.4f} SOL\n"
                f"Cost/Trade: {report.cost_per_trade:.4f} SOL\n"
                f"Max Drawdown: {report.max_drawdown_percent:.2f}%\n"
                f"Best/Worst: {report.largest_win_sol:+.4f}/{report.largest_loss_sol:+.4f} SOL\n"
                f"Sharpe: {report.sharpe_ratio:.2f}\n"
                f"{'='*60}\n"
            )
            
            # Write to hourly log file
            if self.hourly_log_path:
                os.makedirs(os.path.dirname(self.hourly_log_path), exist_ok=True)
                with open(self.hourly_log_path, 'a') as f:
                    f.write(summary)
            
            # Also log to main logger
            self.logger.info(f"Hourly Report Generated: {report.profit_after_costs:+.4f} SOL net profit")
            
        except Exception as e:
            self.logger.error(f"Hourly log write error: {e}")
    
    def _check_hourly_alerts(self, report: HourlyReport):
        """Check hourly metrics for alerts"""
        
        # Alert conditions
        alerts = []
        
        if report.win_rate < 30:
            alerts.append(f"Low win rate: {report.win_rate:.1f}%")
            
        if report.cost_per_trade > trading_config.max_rpc_cost_per_trade * 10:
            alerts.append(f"High cost per trade: {report.cost_per_trade:.4f} SOL")
            
        if report.profit_after_costs < -0.5:
            alerts.append(f"Significant losses: {report.profit_after_costs:.4f} SOL")
            
        if report.max_drawdown_percent > 25:
            alerts.append(f"High drawdown: {report.max_drawdown_percent:.1f}%")
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"ALERT: {alert}")
    
    def get_session_summary(self) -> Dict:
        """Get current session performance summary"""
        
        if not self.conn:
            return {}
            
        try:
            cursor = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_trades,
                    SUM(CASE WHEN trade_type = 'BUY' THEN 1 ELSE 0 END) as buy_trades,
                    SUM(CASE WHEN trade_type = 'SELL' THEN 1 ELSE 0 END) as sell_trades,
                    SUM(profit_loss_sol) as total_profit,
                    AVG(CASE WHEN trade_type = 'SELL' AND profit_loss_percent IS NOT NULL 
                        THEN profit_loss_percent ELSE NULL END) as avg_profit_percent,
                    SUM(gas_cost_sol + rpc_cost_sol + api_cost_sol) as total_costs
                FROM trades 
                WHERE timestamp >= ?
            """, (self.session_start,))
            
            row = cursor.fetchone()
            
            if row:
                # Calculate win rate
                sell_wins = self.conn.execute("""
                    SELECT COUNT(*) FROM trades 
                    WHERE timestamp >= ? AND trade_type = 'SELL' 
                    AND success = 1 AND profit_loss_sol > 0
                """, (self.session_start,)).fetchone()[0]
                
                win_rate = (sell_wins / row[3] * 100) if row[3] > 0 else 0
                
                return {
                    "session_duration_hours": (datetime.utcnow() - datetime.fromisoformat(self.session_start)).total_seconds() / 3600,
                    "total_trades": row[0],
                    "successful_trades": row[1], 
                    "buy_trades": row[2],
                    "sell_trades": row[3],
                    "win_rate": win_rate,
                    "total_profit_sol": row[4] or 0,
                    "average_profit_percent": row[5] or 0,
                    "total_costs_sol": row[6] or 0,
                    "net_profit_sol": (row[4] or 0) - (row[6] or 0),
                    "active_positions": len(self.active_positions),
                    "daily_costs": self.daily_costs
                }
            
        except Exception as e:
            self.logger.error(f"Session summary error: {e}")
        
        return {}
    
    def check_safety_limits(self) -> bool:
        """Check if trading should continue based on costs and performance"""
        
        # Check daily cost limits
        if self.daily_costs['total'] >= monitoring_config.cost_alert_threshold_sol:
            self.logger.warning("Daily cost limit reached")
            return False
        
        # Check cost percentage vs profit
        session_summary = self.get_session_summary()
        if session_summary:
            net_profit = session_summary.get('net_profit_sol', 0)
            total_costs = session_summary.get('total_costs_sol', 0)
            
            if net_profit > 0 and total_costs > 0:
                cost_percentage = (total_costs / net_profit) * 100
                if cost_percentage > trading_config.alert_cost_threshold_percent:
                    self.logger.warning(f"Costs too high: {cost_percentage:.1f}% of profit")
                    return False
        
        return True
    
    def shutdown(self):
        """Clean shutdown of logger"""
        
        self.logger.info("Shutting down optimized logger...")
        
        # Stop background processors
        self.batch_processor_running = False
        
        # Write final batch
        if monitoring_config.batch_logs:
            with self.batch_lock:
                if self.batch_queue:
                    self._write_batch_to_db()
        
        # Close database
        if self.conn:
            self.conn.close()
        
        self.logger.info("Logger shutdown complete")


# Global logger instance
optimized_logger = OptimizedTradeLogger() 