"""
Smart Ape Mode - Comprehensive Data Logger
=========================================

Complete data capture system for analysis and optimization:
- All trading decisions with full context
- AI model predictions and validation
- Market data and signals
- System performance metrics
- Error tracking and debugging
- Decision trees and reasoning
"""

import json
import logging
import asyncio
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import psutil
import threading
from contextlib import contextmanager
import hashlib
import uuid

class ComprehensiveLogger:
    """Comprehensive logging system for all bot activities"""
    
    def __init__(self, log_dir: str = "comprehensive_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Session tracking
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start = datetime.now()
        
        # Data storage
        self.trade_logs = []
        self.ai_prediction_logs = []
        self.market_data_logs = []
        self.system_performance_logs = []
        self.error_logs = []
        self.decision_logs = []
        
        # Performance tracking
        self.trade_counter = 0
        self.prediction_counter = 0
        self.error_counter = 0
        
        # Setup logging files
        self._setup_comprehensive_logging()
        
        # Start background logging
        self.logging_active = True
        self._start_background_logging()
    
    def _setup_comprehensive_logging(self):
        """Setup all logging files with structured formats"""
        
        # Create session info
        session_info = {
            'session_id': self.session_id,
            'start_time': self.session_start.isoformat(),
            'bot_version': 'Smart_Ape_Mode_v1',
            'python_version': f"{psutil.sys.version}",
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'platform': psutil.os.name,
                'disk_space': psutil.disk_usage('/').free
            }
        }
        
        with open(self.log_dir / f'session_{self.session_id}.json', 'w') as f:
            json.dump(session_info, f, indent=2, default=str)
    
    def _start_background_logging(self):
        """Start background logging thread"""
        def background_logger():
            while self.logging_active:
                try:
                    self._log_system_performance()
                    self._flush_logs_to_disk()
                    time.sleep(30)  # Log every 30 seconds
                except Exception as e:
                    print(f"Background logging error: {e}")
        
        self.bg_thread = threading.Thread(target=background_logger, daemon=True)
        self.bg_thread.start()
    
    # =================== TRADING LOGS ===================
    
    def log_trading_decision(self, 
                           token_symbol: str,
                           decision: str,  # "buy", "sell", "hold", "skip"
                           reasoning: Dict[str, Any],
                           market_data: Dict[str, Any],
                           ai_signals: Dict[str, Any],
                           risk_assessment: Dict[str, Any],
                           execution_result: Optional[Dict[str, Any]] = None):
        """Log complete trading decision with full context"""
        
        self.trade_counter += 1
        
        trade_log = {
            'log_id': f"trade_{self.session_id}_{self.trade_counter}",
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'trade_number': self.trade_counter,
            
            # Core decision
            'token_symbol': token_symbol,
            'decision': decision,
            'reasoning': reasoning,
            
            # Market context
            'market_data': {
                'price': market_data.get('current_price'),
                'volume_24h': market_data.get('volume_24h'),
                'price_change_24h': market_data.get('price_change_24h'),
                'market_cap': market_data.get('market_cap'),
                'liquidity': market_data.get('liquidity'),
                'holder_count': market_data.get('holder_count'),
                'top_holders_percentage': market_data.get('top_holders_percentage'),
                'trading_activity': market_data.get('trading_activity'),
                'dex_info': market_data.get('dex_info')
            },
            
            # AI Signals
            'ai_signals': {
                'ml_predictor': {
                    'prediction': ai_signals.get('ml_prediction'),
                    'confidence': ai_signals.get('ml_confidence'),
                    'predicted_price': ai_signals.get('predicted_price'),
                    'direction': ai_signals.get('predicted_direction'),
                    'hold_duration': ai_signals.get('hold_duration')
                },
                'sentiment_analyzer': {
                    'overall_sentiment': ai_signals.get('sentiment_score'),
                    'confidence': ai_signals.get('sentiment_confidence'),
                    'social_buzz': ai_signals.get('social_buzz'),
                    'momentum': ai_signals.get('sentiment_momentum'),
                    'mention_count': ai_signals.get('mention_count')
                },
                'technical_analyzer': {
                    'overall_signal': ai_signals.get('technical_signal'),
                    'rsi': ai_signals.get('rsi'),
                    'macd': ai_signals.get('macd'),
                    'bollinger_position': ai_signals.get('bollinger_position'),
                    'volume_spike': ai_signals.get('volume_spike'),
                    'breakout_probability': ai_signals.get('breakout_probability')
                }
            },
            
            # Risk Assessment
            'risk_assessment': {
                'position_size': risk_assessment.get('position_size'),
                'max_loss': risk_assessment.get('max_loss'),
                'stop_loss': risk_assessment.get('stop_loss'),
                'take_profit': risk_assessment.get('take_profit'),
                'risk_score': risk_assessment.get('risk_score'),
                'portfolio_impact': risk_assessment.get('portfolio_impact'),
                'correlation_risk': risk_assessment.get('correlation_risk')
            },
            
            # Execution details
            'execution': execution_result or {},
            
            # System context
            'system_context': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'active_trades': len(self.trade_logs),
                'bot_uptime': (datetime.now() - self.session_start).total_seconds()
            }
        }
        
        self.trade_logs.append(trade_log)
        self._save_log(trade_log, 'trading_decisions')
    
    def log_trade_execution(self, 
                          trade_id: str,
                          execution_details: Dict[str, Any],
                          transaction_hash: str,
                          gas_used: float,
                          actual_price: float,
                          slippage: float,
                          execution_time: float):
        """Log trade execution details"""
        
        execution_log = {
            'log_id': f"execution_{self.session_id}_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'trade_id': trade_id,
            'transaction_hash': transaction_hash,
            'execution_details': execution_details,
            'performance_metrics': {
                'gas_used': gas_used,
                'actual_price': actual_price,
                'slippage_percent': slippage,
                'execution_time_ms': execution_time,
                'success': execution_details.get('success', False)
            }
        }
        
        self._save_log(execution_log, 'trade_executions')
    
    def log_trade_outcome(self,
                         trade_id: str,
                         entry_price: float,
                         exit_price: float,
                         profit_loss: float,
                         profit_loss_percentage: float,
                         hold_duration: timedelta,
                         exit_reason: str,
                         ai_prediction_accuracy: Dict[str, float]):
        """Log final trade outcome for analysis"""
        
        outcome_log = {
            'log_id': f"outcome_{self.session_id}_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'trade_id': trade_id,
            'outcome': {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_loss_absolute': profit_loss,
                'profit_loss_percentage': profit_loss_percentage,
                'hold_duration_minutes': hold_duration.total_seconds() / 60,
                'exit_reason': exit_reason,
                'successful': profit_loss > 0
            },
            'ai_accuracy_validation': {
                'ml_predictor_accuracy': ai_prediction_accuracy.get('ml_accuracy'),
                'sentiment_correlation': ai_prediction_accuracy.get('sentiment_correlation'),
                'technical_signal_accuracy': ai_prediction_accuracy.get('technical_accuracy'),
                'combined_signal_accuracy': ai_prediction_accuracy.get('combined_accuracy')
            }
        }
        
        self._save_log(outcome_log, 'trade_outcomes')
    
    # =================== AI MODEL LOGS ===================
    
    def log_ai_prediction(self,
                         model_name: str,
                         token_symbol: str,
                         prediction: Dict[str, Any],
                         input_features: Dict[str, Any],
                         confidence: float,
                         inference_time: float):
        """Log AI model predictions with full context"""
        
        self.prediction_counter += 1
        
        prediction_log = {
            'log_id': f"prediction_{self.session_id}_{self.prediction_counter}",
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'prediction_number': self.prediction_counter,
            'model_name': model_name,
            'token_symbol': token_symbol,
            'prediction': prediction,
            'input_features': input_features,
            'model_performance': {
                'confidence': confidence,
                'inference_time_ms': inference_time,
                'feature_count': len(input_features),
                'prediction_horizon': prediction.get('horizon_minutes', 0)
            }
        }
        
        self.ai_prediction_logs.append(prediction_log)
        self._save_log(prediction_log, 'ai_predictions')
    
    def log_ai_model_performance(self,
                               model_name: str,
                               accuracy_metrics: Dict[str, float],
                               performance_metrics: Dict[str, float],
                               drift_metrics: Dict[str, float],
                               training_info: Optional[Dict[str, Any]] = None):
        """Log AI model performance and health"""
        
        performance_log = {
            'log_id': f"model_perf_{self.session_id}_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'model_name': model_name,
            'accuracy_metrics': accuracy_metrics,
            'performance_metrics': performance_metrics,
            'drift_metrics': drift_metrics,
            'training_info': training_info or {}
        }
        
        self._save_log(performance_log, 'ai_model_performance')
    
    # =================== MARKET DATA LOGS ===================
    
    def log_market_data(self,
                       token_symbol: str,
                       price_data: Dict[str, Any],
                       volume_data: Dict[str, Any],
                       liquidity_data: Dict[str, Any],
                       social_data: Dict[str, Any],
                       technical_indicators: Dict[str, Any]):
        """Log comprehensive market data"""
        
        market_log = {
            'log_id': f"market_{self.session_id}_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'token_symbol': token_symbol,
            'price_data': price_data,
            'volume_data': volume_data,
            'liquidity_data': liquidity_data,
            'social_data': social_data,
            'technical_indicators': technical_indicators
        }
        
        self.market_data_logs.append(market_log)
        self._save_log(market_log, 'market_data')
    
    # =================== SYSTEM LOGS ===================
    
    def _log_system_performance(self):
        """Log system performance metrics"""
        
        performance_log = {
            'log_id': f"system_{self.session_id}_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'system_metrics': {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_io': {
                    'bytes_sent': psutil.net_io_counters().bytes_sent,
                    'bytes_recv': psutil.net_io_counters().bytes_recv
                },
                'process_info': {
                    'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                    'cpu_percent': psutil.Process().cpu_percent(),
                    'threads': psutil.Process().num_threads()
                }
            },
            'bot_metrics': {
                'trades_completed': self.trade_counter,
                'predictions_made': self.prediction_counter,
                'errors_encountered': self.error_counter,
                'uptime_seconds': (datetime.now() - self.session_start).total_seconds()
            }
        }
        
        self.system_performance_logs.append(performance_log)
        self._save_log(performance_log, 'system_performance')
    
    # =================== ERROR LOGS ===================
    
    def log_error(self,
                  error_type: str,
                  error_message: str,
                  context: Dict[str, Any],
                  traceback_info: str,
                  severity: str = "error"):
        """Log errors with full context"""
        
        self.error_counter += 1
        
        error_log = {
            'log_id': f"error_{self.session_id}_{self.error_counter}",
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'error_number': self.error_counter,
            'error_info': {
                'type': error_type,
                'message': error_message,
                'severity': severity,
                'traceback': traceback_info
            },
            'context': context,
            'system_state': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'active_trades': len(self.trade_logs),
                'bot_uptime': (datetime.now() - self.session_start).total_seconds()
            }
        }
        
        self.error_logs.append(error_log)
        self._save_log(error_log, 'errors')
    
    # =================== DECISION LOGS ===================
    
    def log_decision_tree(self,
                         decision_type: str,
                         inputs: Dict[str, Any],
                         decision_process: List[Dict[str, Any]],
                         final_decision: Any,
                         confidence: float,
                         reasoning: str):
        """Log decision-making process for analysis"""
        
        decision_log = {
            'log_id': f"decision_{self.session_id}_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'decision_type': decision_type,
            'inputs': inputs,
            'decision_process': decision_process,
            'final_decision': final_decision,
            'confidence': confidence,
            'reasoning': reasoning
        }
        
        self.decision_logs.append(decision_log)
        self._save_log(decision_log, 'decisions')
    
    # =================== UTILITY METHODS ===================
    
    def _save_log(self, log_data: Dict[str, Any], log_type: str):
        """Save log to appropriate file"""
        
        # Daily log files
        date_str = datetime.now().strftime('%Y%m%d')
        log_file = self.log_dir / f'{log_type}_{date_str}.jsonl'
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_data, default=str) + '\n')
        except Exception as e:
            print(f"Error saving log: {e}")
    
    def _flush_logs_to_disk(self):
        """Flush in-memory logs to disk"""
        
        try:
            # Create comprehensive session summary
            session_summary = {
                'session_id': self.session_id,
                'session_duration': (datetime.now() - self.session_start).total_seconds(),
                'summary_timestamp': datetime.now().isoformat(),
                'statistics': {
                    'total_trades': self.trade_counter,
                    'total_predictions': self.prediction_counter,
                    'total_errors': self.error_counter,
                    'trades_per_hour': self.trade_counter / max(1, (datetime.now() - self.session_start).total_seconds() / 3600),
                    'error_rate': self.error_counter / max(1, self.trade_counter + self.prediction_counter)
                },
                'performance_summary': {
                    'avg_cpu': sum(log.get('system_metrics', {}).get('cpu_usage', 0) for log in self.system_performance_logs[-10:]) / max(1, len(self.system_performance_logs[-10:])),
                    'avg_memory': sum(log.get('system_metrics', {}).get('memory_usage', 0) for log in self.system_performance_logs[-10:]) / max(1, len(self.system_performance_logs[-10:]))
                }
            }
            
            # Save session summary
            with open(self.log_dir / f'session_summary_{self.session_id}.json', 'w') as f:
                json.dump(session_summary, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error flushing logs: {e}")
    
    @contextmanager
    def log_operation(self, operation_name: str, context: Dict[str, Any] = None):
        """Context manager for logging operations"""
        
        start_time = time.time()
        operation_id = f"op_{self.session_id}_{int(start_time)}"
        
        try:
            self.log_decision_tree(
                decision_type=f"start_{operation_name}",
                inputs=context or {},
                decision_process=[{"step": "operation_start", "timestamp": datetime.now().isoformat()}],
                final_decision="proceed",
                confidence=1.0,
                reasoning=f"Starting {operation_name}"
            )
            
            yield operation_id
            
            # Log successful completion
            end_time = time.time()
            self.log_decision_tree(
                decision_type=f"complete_{operation_name}",
                inputs={"operation_id": operation_id, "duration": end_time - start_time},
                decision_process=[{"step": "operation_complete", "timestamp": datetime.now().isoformat()}],
                final_decision="success",
                confidence=1.0,
                reasoning=f"Successfully completed {operation_name}"
            )
            
        except Exception as e:
            # Log error
            self.log_error(
                error_type=f"{operation_name}_error",
                error_message=str(e),
                context={"operation_id": operation_id, "operation_name": operation_name},
                traceback_info=traceback.format_exc(),
                severity="error"
            )
            raise
    
    def generate_analysis_dataset(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate comprehensive dataset for analysis"""
        
        # Filter logs by date if specified
        if start_date is None:
            start_date = self.session_start
        if end_date is None:
            end_date = datetime.now()
        
        # Compile all data
        dataset = {
            'metadata': {
                'session_id': self.session_id,
                'generation_time': datetime.now().isoformat(),
                'data_period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'record_counts': {
                    'trades': len(self.trade_logs),
                    'predictions': len(self.ai_prediction_logs),
                    'market_data': len(self.market_data_logs),
                    'system_performance': len(self.system_performance_logs),
                    'errors': len(self.error_logs),
                    'decisions': len(self.decision_logs)
                }
            },
            'trading_data': self.trade_logs,
            'ai_predictions': self.ai_prediction_logs,
            'market_data': self.market_data_logs,
            'system_performance': self.system_performance_logs,
            'errors': self.error_logs,
            'decisions': self.decision_logs
        }
        
        # Save analysis dataset
        analysis_file = self.log_dir / f'analysis_dataset_{self.session_id}_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
        with open(analysis_file, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        return dataset
    
    def stop_logging(self):
        """Stop the logging system"""
        self.logging_active = False
        self._flush_logs_to_disk()
        print(f"Comprehensive logging stopped. Session: {self.session_id}")

# Global logger instance
_comprehensive_logger = None

def get_comprehensive_logger() -> ComprehensiveLogger:
    """Get the global comprehensive logger instance"""
    global _comprehensive_logger
    if _comprehensive_logger is None:
        _comprehensive_logger = ComprehensiveLogger()
    return _comprehensive_logger

# Convenience functions for easy logging
def log_trade(token_symbol: str, decision: str, reasoning: Dict, market_data: Dict, 
              ai_signals: Dict, risk_assessment: Dict, execution_result: Dict = None):
    """Convenience function for logging trades"""
    get_comprehensive_logger().log_trading_decision(
        token_symbol, decision, reasoning, market_data, ai_signals, risk_assessment, execution_result
    )

def log_prediction(model_name: str, token_symbol: str, prediction: Dict, 
                  input_features: Dict, confidence: float, inference_time: float):
    """Convenience function for logging AI predictions"""
    get_comprehensive_logger().log_ai_prediction(
        model_name, token_symbol, prediction, input_features, confidence, inference_time
    )

def log_error_with_context(error_type: str, error_message: str, context: Dict, severity: str = "error"):
    """Convenience function for logging errors"""
    get_comprehensive_logger().log_error(
        error_type, error_message, context, traceback.format_exc(), severity
    )

def log_decision(decision_type: str, inputs: Dict, process: List[Dict], 
                decision: Any, confidence: float, reasoning: str):
    """Convenience function for logging decisions"""
    get_comprehensive_logger().log_decision_tree(
        decision_type, inputs, process, decision, confidence, reasoning
    )

def generate_analysis_data() -> Dict[str, Any]:
    """Generate complete analysis dataset"""
    return get_comprehensive_logger().generate_analysis_dataset()

if __name__ == "__main__":
    # Demo usage
    logger = ComprehensiveLogger()
    
    # Example trade log
    logger.log_trading_decision(
        token_symbol="EXAMPLE",
        decision="buy",
        reasoning={"signal_strength": 0.8, "risk_acceptable": True},
        market_data={"current_price": 0.001, "volume_24h": 100000},
        ai_signals={"ml_confidence": 0.75, "sentiment_score": 0.6},
        risk_assessment={"position_size": 0.05, "max_loss": 0.02}
    )
    
    print(f"Comprehensive logging started. Session ID: {logger.session_id}")
    print("All data will be captured for analysis!")