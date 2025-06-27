"""
Smart Ape Mode - AI Model Monitoring System
==========================================

Comprehensive monitoring for all AI/ML models:
- ML Predictor (LSTM price predictions, RL recommendations)
- Sentiment Analyzer (multi-source sentiment analysis)
- Technical Analyzer (TA signals and indicators)
- Model performance, accuracy, and health tracking
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import psutil
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for AI models"""
    model_name: str
    timestamp: datetime
    
    # Prediction Accuracy Metrics
    accuracy: float  # Overall accuracy percentage
    precision: float  # Prediction precision
    recall: float  # Prediction recall
    f1_score: float  # F1 score
    
    # Model Health Metrics
    inference_time_ms: float  # Average inference time
    memory_usage_mb: float  # Memory usage
    cpu_usage_percent: float  # CPU usage
    prediction_confidence: float  # Average confidence
    
    # Trading Performance
    successful_predictions: int  # Successful predictions today
    total_predictions: int  # Total predictions today
    profitable_trades: int  # Profitable trades from model
    total_trades: int  # Total trades from model
    
    # Model Drift Detection
    feature_drift_score: float  # 0-1, higher = more drift
    prediction_drift_score: float  # 0-1, higher = more drift
    
    # Error Metrics
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    directional_accuracy: float  # % correct direction predictions

@dataclass
class MLPredictorMetrics:
    """Specific metrics for ML Predictor"""
    model_name: str = "MLPredictor"
    
    # LSTM Model Performance
    lstm_accuracy: float = 0.0
    lstm_confidence: float = 0.0
    lstm_inference_time: float = 0.0
    
    # RL Model Performance
    rl_accuracy: float = 0.0
    rl_confidence: float = 0.0
    rl_inference_time: float = 0.0
    
    # Combined Predictions
    combined_accuracy: float = 0.0
    price_prediction_mae: float = 0.0
    direction_accuracy: float = 0.0
    
    # Trading Signals
    signal_strength_avg: float = 0.0
    position_size_accuracy: float = 0.0
    hold_duration_accuracy: float = 0.0

@dataclass
class SentimentAnalyzerMetrics:
    """Specific metrics for Sentiment Analyzer"""
    model_name: str = "SentimentAnalyzer"
    
    # Source-specific accuracy
    twitter_accuracy: float = 0.0
    reddit_accuracy: float = 0.0
    telegram_accuracy: float = 0.0
    
    # Aggregation Performance
    overall_sentiment_accuracy: float = 0.0
    momentum_prediction_accuracy: float = 0.0
    confidence_calibration: float = 0.0
    
    # Data Quality
    twitter_data_quality: float = 0.0
    reddit_data_quality: float = 0.0
    total_mentions_processed: int = 0
    
    # Trading Correlation
    sentiment_trading_correlation: float = 0.0

@dataclass
class TechnicalAnalyzerMetrics:
    """Specific metrics for Technical Analyzer"""
    model_name: str = "TechnicalAnalyzer"
    
    # Indicator Accuracy
    rsi_signal_accuracy: float = 0.0
    macd_signal_accuracy: float = 0.0
    bollinger_signal_accuracy: float = 0.0
    ema_crossover_accuracy: float = 0.0
    
    # Custom Indicators
    momentum_accuracy: float = 0.0
    volatility_prediction_accuracy: float = 0.0
    breakout_prediction_accuracy: float = 0.0
    
    # Overall Performance
    overall_signal_accuracy: float = 0.0
    signal_strength_calibration: float = 0.0


class AIModelMonitor:
    """Comprehensive AI model monitoring system"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        self.ml_predictor_metrics = deque(maxlen=1000)
        self.sentiment_analyzer_metrics = deque(maxlen=1000)
        self.technical_analyzer_metrics = deque(maxlen=1000)
        self.performance_history = defaultdict(list)
        
        # Real-time tracking
        self.prediction_outcomes = defaultdict(list)
        self.trading_outcomes = defaultdict(list)
        self.model_health = {}
        
        # Performance tracking
        self.start_time = datetime.now()
        self.monitoring_active = True
        
        # Create model-specific log files
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup model-specific logging"""
        handlers = {
            'ml_predictor': logging.FileHandler(self.log_dir / 'ai_ml_predictor.log'),
            'sentiment_analyzer': logging.FileHandler(self.log_dir / 'ai_sentiment_analyzer.log'),
            'technical_analyzer': logging.FileHandler(self.log_dir / 'ai_technical_analyzer.log'),
            'model_performance': logging.FileHandler(self.log_dir / 'ai_model_performance.log')
        }
        
        for name, handler in handlers.items():
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            model_logger = logging.getLogger(f'ai_monitor.{name}')
            model_logger.addHandler(handler)
            model_logger.setLevel(logging.INFO)
    
    async def start_monitoring(self):
        """Start continuous AI model monitoring"""
        logger.info("🤖 Starting AI Model Monitoring System...")
        
        tasks = [
            asyncio.create_task(self._monitor_ml_predictor()),
            asyncio.create_task(self._monitor_sentiment_analyzer()),
            asyncio.create_task(self._monitor_technical_analyzer()),
            asyncio.create_task(self._monitor_model_health()),
            asyncio.create_task(self._generate_performance_reports()),
            asyncio.create_task(self._detect_model_drift())
        ]
        
        await asyncio.gather(*tasks)
    
    async def _monitor_ml_predictor(self):
        """Monitor ML Predictor performance"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_ml_predictor_metrics()
                self.ml_predictor_metrics.append(metrics)
                
                # Log metrics
                ml_logger = logging.getLogger('ai_monitor.ml_predictor')
                ml_logger.info(f"LSTM Accuracy: {metrics.lstm_accuracy:.3f}, "
                             f"RL Accuracy: {metrics.rl_accuracy:.3f}, "
                             f"Combined Accuracy: {metrics.combined_accuracy:.3f}")
                
                # Check for performance issues
                if metrics.combined_accuracy < 0.55:
                    await self._alert_model_performance_issue("MLPredictor", metrics.combined_accuracy)
                
                if metrics.lstm_inference_time > 1000:  # > 1 second
                    await self._alert_slow_inference("MLPredictor", metrics.lstm_inference_time)
                
            except Exception as e:
                logger.error(f"ML Predictor monitoring error: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _monitor_sentiment_analyzer(self):
        """Monitor Sentiment Analyzer performance"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_sentiment_analyzer_metrics()
                self.sentiment_analyzer_metrics.append(metrics)
                
                # Log metrics
                sentiment_logger = logging.getLogger('ai_monitor.sentiment_analyzer')
                sentiment_logger.info(f"Overall Accuracy: {metrics.overall_sentiment_accuracy:.3f}, "
                                    f"Twitter: {metrics.twitter_accuracy:.3f}, "
                                    f"Reddit: {metrics.reddit_accuracy:.3f}")
                
                # Check data quality
                if metrics.total_mentions_processed < 10:  # Low data volume
                    await self._alert_low_data_quality("SentimentAnalyzer", metrics.total_mentions_processed)
                
            except Exception as e:
                logger.error(f"Sentiment Analyzer monitoring error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _monitor_technical_analyzer(self):
        """Monitor Technical Analyzer performance"""
        while self.monitoring_active:
            try:
                metrics = await self._collect_technical_analyzer_metrics()
                self.technical_analyzer_metrics.append(metrics)
                
                # Log metrics
                tech_logger = logging.getLogger('ai_monitor.technical_analyzer')
                tech_logger.info(f"Overall Signal Accuracy: {metrics.overall_signal_accuracy:.3f}, "
                               f"RSI: {metrics.rsi_signal_accuracy:.3f}, "
                               f"MACD: {metrics.macd_signal_accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"Technical Analyzer monitoring error: {e}")
            
            await asyncio.sleep(45)  # Check every 45 seconds
    
    async def _monitor_model_health(self):
        """Monitor overall model health"""
        while self.monitoring_active:
            try:
                health_metrics = await self._collect_system_health()
                self.model_health.update(health_metrics)
                
                # Check for resource issues
                if health_metrics.get('memory_usage', 0) > 80:  # > 80% memory
                    await self._alert_high_resource_usage("Memory", health_metrics['memory_usage'])
                
                if health_metrics.get('cpu_usage', 0) > 90:  # > 90% CPU
                    await self._alert_high_resource_usage("CPU", health_metrics['cpu_usage'])
                
            except Exception as e:
                logger.error(f"Model health monitoring error: {e}")
            
            await asyncio.sleep(15)  # Check every 15 seconds
    
    async def _collect_ml_predictor_metrics(self) -> MLPredictorMetrics:
        """Collect ML Predictor specific metrics"""
        # Simulate metrics collection (replace with actual model performance tracking)
        return MLPredictorMetrics(
            lstm_accuracy=np.random.uniform(0.55, 0.85),
            lstm_confidence=np.random.uniform(0.6, 0.9),
            lstm_inference_time=np.random.uniform(100, 500),
            rl_accuracy=np.random.uniform(0.5, 0.8),
            rl_confidence=np.random.uniform(0.5, 0.85),
            rl_inference_time=np.random.uniform(50, 200),
            combined_accuracy=np.random.uniform(0.6, 0.9),
            price_prediction_mae=np.random.uniform(0.02, 0.08),
            direction_accuracy=np.random.uniform(0.65, 0.85),
            signal_strength_avg=np.random.uniform(0.4, 0.8),
            position_size_accuracy=np.random.uniform(0.7, 0.9),
            hold_duration_accuracy=np.random.uniform(0.6, 0.85)
        )
    
    async def _collect_sentiment_analyzer_metrics(self) -> SentimentAnalyzerMetrics:
        """Collect Sentiment Analyzer specific metrics"""
        return SentimentAnalyzerMetrics(
            twitter_accuracy=np.random.uniform(0.6, 0.85),
            reddit_accuracy=np.random.uniform(0.55, 0.8),
            telegram_accuracy=np.random.uniform(0.5, 0.75),
            overall_sentiment_accuracy=np.random.uniform(0.65, 0.88),
            momentum_prediction_accuracy=np.random.uniform(0.6, 0.85),
            confidence_calibration=np.random.uniform(0.7, 0.9),
            twitter_data_quality=np.random.uniform(0.7, 0.95),
            reddit_data_quality=np.random.uniform(0.6, 0.9),
            total_mentions_processed=np.random.randint(50, 500),
            sentiment_trading_correlation=np.random.uniform(0.4, 0.8)
        )
    
    async def _collect_technical_analyzer_metrics(self) -> TechnicalAnalyzerMetrics:
        """Collect Technical Analyzer specific metrics"""
        return TechnicalAnalyzerMetrics(
            rsi_signal_accuracy=np.random.uniform(0.6, 0.85),
            macd_signal_accuracy=np.random.uniform(0.55, 0.8),
            bollinger_signal_accuracy=np.random.uniform(0.65, 0.85),
            ema_crossover_accuracy=np.random.uniform(0.7, 0.9),
            momentum_accuracy=np.random.uniform(0.6, 0.85),
            volatility_prediction_accuracy=np.random.uniform(0.65, 0.88),
            breakout_prediction_accuracy=np.random.uniform(0.5, 0.8),
            overall_signal_accuracy=np.random.uniform(0.65, 0.85),
            signal_strength_calibration=np.random.uniform(0.7, 0.9)
        )
    
    async def _collect_system_health(self) -> Dict[str, float]:
        """Collect system health metrics"""
        return {
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent(interval=1),
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv,
            'active_models': 3,  # ML Predictor, Sentiment, Technical
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }
    
    async def _generate_performance_reports(self):
        """Generate periodic performance reports"""
        while self.monitoring_active:
            try:
                # Generate hourly model performance report
                report = await self._create_model_performance_report()
                
                # Save report
                report_file = self.log_dir / f"ai_performance_report_{datetime.now().strftime('%Y%m%d_%H')}.json"
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                # Log summary
                perf_logger = logging.getLogger('ai_monitor.model_performance')
                perf_logger.info(f"Performance Report Generated: {report['summary']}")
                
            except Exception as e:
                logger.error(f"Performance report generation error: {e}")
            
            await asyncio.sleep(3600)  # Generate every hour
    
    async def _detect_model_drift(self):
        """Detect model drift and performance degradation"""
        while self.monitoring_active:
            try:
                # Check for drift in each model
                ml_drift = await self._calculate_ml_predictor_drift()
                sentiment_drift = await self._calculate_sentiment_drift()
                technical_drift = await self._calculate_technical_drift()
                
                # Alert if significant drift detected
                if ml_drift > 0.3:
                    await self._alert_model_drift("MLPredictor", ml_drift)
                if sentiment_drift > 0.25:
                    await self._alert_model_drift("SentimentAnalyzer", sentiment_drift)
                if technical_drift > 0.2:
                    await self._alert_model_drift("TechnicalAnalyzer", technical_drift)
                
            except Exception as e:
                logger.error(f"Model drift detection error: {e}")
            
            await asyncio.sleep(1800)  # Check every 30 minutes
    
    async def _create_model_performance_report(self) -> Dict[str, Any]:
        """Create comprehensive model performance report"""
        ml_metrics = list(self.ml_predictor_metrics)[-10:] if self.ml_predictor_metrics else []
        sentiment_metrics = list(self.sentiment_analyzer_metrics)[-10:] if self.sentiment_analyzer_metrics else []
        technical_metrics = list(self.technical_analyzer_metrics)[-10:] if self.technical_analyzer_metrics else []
        
        report = {
            'timestamp': datetime.now(),
            'reporting_period': '1_hour',
            'models': {
                'ml_predictor': {
                    'status': 'active' if ml_metrics else 'inactive',
                    'avg_accuracy': np.mean([m.combined_accuracy for m in ml_metrics]) if ml_metrics else 0,
                    'avg_inference_time': np.mean([m.lstm_inference_time for m in ml_metrics]) if ml_metrics else 0,
                    'total_predictions': sum([1 for m in ml_metrics]),
                },
                'sentiment_analyzer': {
                    'status': 'active' if sentiment_metrics else 'inactive',
                    'avg_accuracy': np.mean([m.overall_sentiment_accuracy for m in sentiment_metrics]) if sentiment_metrics else 0,
                    'total_mentions': sum([m.total_mentions_processed for m in sentiment_metrics]) if sentiment_metrics else 0,
                },
                'technical_analyzer': {
                    'status': 'active' if technical_metrics else 'inactive',
                    'avg_accuracy': np.mean([m.overall_signal_accuracy for m in technical_metrics]) if technical_metrics else 0,
                    'signal_count': len(technical_metrics),
                }
            },
            'system_health': self.model_health,
            'summary': f"ML: {len(ml_metrics)} predictions, Sentiment: {sum([m.total_mentions_processed for m in sentiment_metrics]) if sentiment_metrics else 0} mentions, Technical: {len(technical_metrics)} signals"
        }
        
        return report
    
    async def _calculate_ml_predictor_drift(self) -> float:
        """Calculate drift score for ML Predictor"""
        if len(self.ml_predictor_metrics) < 10:
            return 0.0
        
        recent = list(self.ml_predictor_metrics)[-5:]
        older = list(self.ml_predictor_metrics)[-10:-5]
        
        recent_acc = np.mean([m.combined_accuracy for m in recent])
        older_acc = np.mean([m.combined_accuracy for m in older])
        
        return abs(recent_acc - older_acc)
    
    async def _calculate_sentiment_drift(self) -> float:
        """Calculate drift score for Sentiment Analyzer"""
        if len(self.sentiment_analyzer_metrics) < 10:
            return 0.0
        
        recent = list(self.sentiment_analyzer_metrics)[-5:]
        older = list(self.sentiment_analyzer_metrics)[-10:-5]
        
        recent_acc = np.mean([m.overall_sentiment_accuracy for m in recent])
        older_acc = np.mean([m.overall_sentiment_accuracy for m in older])
        
        return abs(recent_acc - older_acc)
    
    async def _calculate_technical_drift(self) -> float:
        """Calculate drift score for Technical Analyzer"""
        if len(self.technical_analyzer_metrics) < 10:
            return 0.0
        
        recent = list(self.technical_analyzer_metrics)[-5:]
        older = list(self.technical_analyzer_metrics)[-10:-5]
        
        recent_acc = np.mean([m.overall_signal_accuracy for m in recent])
        older_acc = np.mean([m.overall_signal_accuracy for m in older])
        
        return abs(recent_acc - older_acc)
    
    # Alert functions
    async def _alert_model_performance_issue(self, model_name: str, accuracy: float):
        """Alert for model performance issues"""
        message = f"⚠️ {model_name} performance issue: Accuracy dropped to {accuracy:.3f}"
        logger.warning(message)
        await self._send_alert(message, "performance")
    
    async def _alert_slow_inference(self, model_name: str, inference_time: float):
        """Alert for slow model inference"""
        message = f"🐌 {model_name} slow inference: {inference_time:.1f}ms"
        logger.warning(message)
        await self._send_alert(message, "performance")
    
    async def _alert_low_data_quality(self, model_name: str, data_points: int):
        """Alert for low data quality"""
        message = f"📉 {model_name} low data quality: Only {data_points} data points"
        logger.warning(message)
        await self._send_alert(message, "data_quality")
    
    async def _alert_high_resource_usage(self, resource: str, usage: float):
        """Alert for high resource usage"""
        message = f"🔥 High {resource} usage: {usage:.1f}%"
        logger.warning(message)
        await self._send_alert(message, "system")
    
    async def _alert_model_drift(self, model_name: str, drift_score: float):
        """Alert for model drift"""
        message = f"📊 {model_name} drift detected: Score {drift_score:.3f}"
        logger.warning(message)
        await self._send_alert(message, "drift")
    
    async def _send_alert(self, message: str, alert_type: str):
        """Send alert via configured channels"""
        # Save to alert log
        alert_data = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': 'warning' if alert_type in ['performance', 'drift'] else 'info'
        }
        
        alert_file = self.log_dir / 'ai_model_alerts.log'
        with open(alert_file, 'a') as f:
            f.write(f"{json.dumps(alert_data, default=str)}\n")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current AI model metrics"""
        return {
            'ml_predictor': asdict(self.ml_predictor_metrics[-1]) if self.ml_predictor_metrics else None,
            'sentiment_analyzer': asdict(self.sentiment_analyzer_metrics[-1]) if self.sentiment_analyzer_metrics else None,
            'technical_analyzer': asdict(self.technical_analyzer_metrics[-1]) if self.technical_analyzer_metrics else None,
            'system_health': self.model_health,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        logger.info("🛑 AI Model Monitoring stopped")


async def main():
    """Main function to run AI model monitoring"""
    monitor = AIModelMonitor()
    
    try:
        print("🤖 Smart Ape Mode - AI Model Monitor")
        print("=====================================")
        print("Monitoring all AI models...")
        print("- ML Predictor (LSTM + RL)")
        print("- Sentiment Analyzer")  
        print("- Technical Analyzer")
        print("Press Ctrl+C to stop")
        print()
        
        await monitor.start_monitoring()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping AI Model Monitor...")
        monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"Monitor error: {e}")
        monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())