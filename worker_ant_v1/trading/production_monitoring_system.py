"""
ENHANCED PRODUCTION MONITORING SYSTEM
===================================

Enterprise-grade monitoring with real-time metrics, intelligent alerting,
performance tracking, predictive analytics, and system observability.
Enhanced with AI-powered anomaly detection and predictive maintenance.
"""

import asyncio
import threading
from typing import (
    Dict, List, Optional, Any, Union
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import psutil
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from worker_ant_v1.utils.logger import setup_logger


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AnomalyType(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    UNUSUAL_TRADING_PATTERN = "unusual_trading_pattern"
    SYSTEM_RESOURCE_SPIKE = "system_resource_spike"
    NETWORK_LATENCY_ANOMALY = "network_latency_anomaly"
    AI_MODEL_DRIFT = "ai_model_drift"
    PREDICTIVE_FAILURE = "predictive_failure"


@dataclass
class Metric:
    """Individual metric data point with enhanced metadata"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""
    confidence: float = 1.0
    anomaly_score: Optional[float] = None
    trend_direction: Optional[str] = None  # up, down, stable
    volatility: Optional[float] = None


@dataclass
class Alert:
    """Enhanced system alert with AI-powered context"""
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    component: str
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    anomaly_type: Optional[AnomalyType] = None
    confidence_score: float = 1.0
    predicted_impact: Optional[str] = None
    recommended_action: Optional[str] = None
    related_metrics: List[str] = field(default_factory=list)
    historical_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemSnapshot:
    """Complete system state snapshot with predictive insights"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    network_io_bytes: Dict[str, int]
    active_threads: int
    open_connections: int
    component_health: Dict[str, str]
    trading_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    predicted_system_load: Optional[float] = None
    anomaly_detected: bool = False
    maintenance_recommended: bool = False
    ai_model_health: Dict[str, Any] = field(default_factory=dict)
    predictive_insights: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictiveInsight:
    """AI-powered predictive insight"""
    insight_id: str
    insight_type: str
    prediction: str
    confidence: float
    timestamp: datetime
    time_horizon: timedelta
    factors: List[str]
    impact_score: float
    recommended_action: Optional[str] = None


class EnhancedMetricsCollector:
    """Enhanced metrics collector with AI-powered analysis"""

    def __init__(self):
        self.logger = setup_logger("EnhancedMetricsCollector")
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self.metric_definitions = {}
        self.collection_interval = 1.0  # seconds
        self.collection_task = None
        self.running = False

        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.anomaly_models = {}
        self.trend_analyzers = {}
        self.prediction_models = {}
        self.historical_patterns = defaultdict(lambda: deque(maxlen=1000))

        self._init_metric_definitions()
        self._register_all_metrics()

    def _init_metric_definitions(self):
        """Initialize metric definitions with categorized metrics"""
        # AI performance metrics
        self.ai_performance_metrics = {
            'ai_prediction_accuracy': {
                'type': MetricType.GAUGE,
                'unit': '%'
            },
            'ai_model_latency_ms': {
                'type': MetricType.HISTOGRAM,
                'unit': 'ms'
            },
            'ai_confidence_score': {
                'type': MetricType.GAUGE,
                'unit': 'score'
            },
            'ai_model_drift_score': {
                'type': MetricType.GAUGE,
                'unit': 'score'
            },
            'ai_feature_importance': {
                'type': MetricType.GAUGE,
                'unit': 'importance'
            },
            'ai_prediction_variance': {
                'type': MetricType.GAUGE,
                'unit': 'variance'
            },
            'ai_ensemble_agreement': {
                'type': MetricType.GAUGE,
                'unit': 'agreement'
            },
            'ai_learning_rate': {
                'type': MetricType.GAUGE,
                'unit': 'rate'
            },
            'ai_memory_usage_mb': {
                'type': MetricType.GAUGE,
                'unit': 'MB'
            },
            'ai_gpu_utilization': {
                'type': MetricType.GAUGE,
                'unit': '%'
            }
        }

        # System metrics
        self.system_metrics = {
            'cpu_usage_percent': {
                'type': MetricType.GAUGE,
                'unit': '%'
            },
            'memory_usage_percent': {
                'type': MetricType.GAUGE,
                'unit': '%'
            },
            'memory_usage_mb': {
                'type': MetricType.GAUGE,
                'unit': 'MB'
            },
            'disk_usage_percent': {
                'type': MetricType.GAUGE,
                'unit': '%'
            },
            'network_bytes_sent': {
                'type': MetricType.COUNTER,
                'unit': 'bytes'
            },
            'network_bytes_recv': {
                'type': MetricType.COUNTER,
                'unit': 'bytes'
            },
            'active_threads': {
                'type': MetricType.GAUGE,
                'unit': 'count'
            },
            'open_file_descriptors': {
                'type': MetricType.GAUGE,
                'unit': 'count'
            },
            'cpu_temperature': {
                'type': MetricType.GAUGE,
                'unit': '°C'
            },
            'gpu_utilization': {
                'type': MetricType.GAUGE,
                'unit': '%'
            },
            'gpu_memory_usage': {
                'type': MetricType.GAUGE,
                'unit': '%'
            },
            'network_latency_ms': {
                'type': MetricType.HISTOGRAM,
                'unit': 'ms'
            },
            'disk_io_operations': {
                'type': MetricType.COUNTER,
                'unit': 'ops/sec'
            },
            'process_priority': {
                'type': MetricType.GAUGE,
                'unit': 'priority'
            }
        }

        # Trading metrics
        self.trading_metrics = {
            'trades_executed_total': {
                'type': MetricType.COUNTER,
                'unit': 'count'
            },
            'trades_successful_total': {
                'type': MetricType.COUNTER,
                'unit': 'count'
            },
            'trades_failed_total': {
                'type': MetricType.COUNTER,
                'unit': 'count'
            },
            'profit_total_sol': {
                'type': MetricType.GAUGE,
                'unit': 'SOL'
            },
            'current_balance_sol': {
                'type': MetricType.GAUGE,
                'unit': 'SOL'
            },
            'active_positions': {
                'type': MetricType.GAUGE,
                'unit': 'count'
            },
            'avg_trade_duration_seconds': {
                'type': MetricType.GAUGE,
                'unit': 'sec'
            },
            'avg_profit_percent': {
                'type': MetricType.GAUGE,
                'unit': '%'
            },
            'win_rate_percent': {
                'type': MetricType.GAUGE,
                'unit': '%'
            },
            'profit_factor': {
                'type': MetricType.GAUGE,
                'unit': 'ratio'
            },
            'max_drawdown_percent': {
                'type': MetricType.GAUGE,
                'unit': '%'
            },
            'sharpe_ratio': {
                'type': MetricType.GAUGE,
                'unit': 'ratio'
            },
            'risk_adjusted_return': {
                'type': MetricType.GAUGE,
                'unit': '%'
            },
            'position_sizing_accuracy': {
                'type': MetricType.GAUGE,
                'unit': '%'
            },
            'market_timing_accuracy': {
                'type': MetricType.GAUGE,
                'unit': '%'
            },
            'liquidity_impact_score': {
                'type': MetricType.GAUGE,
                'unit': 'score'
            },
            'slippage_avg_percent': {
                'type': MetricType.GAUGE,
                'unit': '%'
            },
            'gas_optimization_score': {
                'type': MetricType.GAUGE,
                'unit': 'score'
            }
        }

    def _register_all_metrics(self):
        """Register all metric definitions"""
        all_metrics = {
            **self.system_metrics,
            **self.trading_metrics,
            **self.ai_performance_metrics
        }
        for name, definition in all_metrics.items():
            self.register_metric(name, definition['type'], definition['unit'])

    def register_metric(self, name: str, metric_type: MetricType, unit: str = ""):
        """Register a new metric"""
        if name not in self.metric_definitions:
            self.metric_definitions[name] = {
                'type': metric_type,
                'unit': unit,
                'created_at': datetime.utcnow()
            }

    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        labels: Dict[str, str] = None
    ):
        """Record a metric value"""
        if name not in self.metric_definitions:
            raise ValueError(f"Metric {name} not registered")

        metric = Metric(
            name=name,
            value=value,
            metric_type=self.metric_definitions[name]['type'],
            timestamp=datetime.utcnow(),
            labels=labels or {},
            unit=self.metric_definitions[name]['unit']
        )
        self.metrics[name].append(metric)

    async def start(self):
        """Start metrics collection"""
        if self.running:
            return

        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())

    async def stop(self):
        """Stop metrics collection"""
        if not self.running:
            return

        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
            self.collection_task = None

    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in collection loop: {str(e)}")
                await asyncio.sleep(self.collection_interval)

    async def _collect_metrics(self):
        """Collect all metrics concurrently"""
        try:
            await asyncio.gather(
                self._collect_system_metrics(),
                self._collect_ai_performance_metrics(),
                self._collect_trading_metrics()
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error(
                f"Error collecting metrics: {str(e)}",
                exc_info=True
            )

    async def _collect_system_metrics(self):
        """Collect system resource metrics and performance data"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_metric('cpu_usage_percent', cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric('memory_usage_percent', memory.percent)
            self.record_metric(
                'memory_usage_mb',
                memory.used / 1024 / 1024
            )

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric('disk_usage_percent', disk.percent)

            # Temperature metrics
            try:
                temps = psutil.sensors_temperatures().get('coretemp', [])
                if temps:
                    self.record_metric('cpu_temperature', temps[0].current)
            except (AttributeError, KeyError, psutil.Error) as e:
                self.logger.debug(
                    f"CPU temperature not available: {str(e)}"
                )

            # Network metrics
            net_io = psutil.net_io_counters()
            self.record_metric('network_bytes_sent', net_io.bytes_sent)
            self.record_metric('network_bytes_recv', net_io.bytes_recv)

            # Thread metrics
            self.record_metric('active_threads', threading.active_count())

            # File descriptor metrics
            if hasattr(psutil.Process(), 'num_fds'):
                proc = psutil.Process()
                self.record_metric(
                    'open_file_descriptors',
                    proc.num_fds()
                )

        except psutil.Error as e:
            self.logger.error(
                f"Error collecting system metrics: {str(e)}",
                exc_info=True
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error in system metrics: {str(e)}",
                exc_info=True
            )

    async def _collect_ai_performance_metrics(self):
        """Collect AI model performance and health metrics"""
        try:
            # Simulate AI metrics collection
            # In production, replace with actual AI model metrics
            self.record_metric(
                'ai_prediction_accuracy',
                np.random.uniform(85, 99)
            )
            self.record_metric(
                'ai_model_latency_ms',
                np.random.uniform(10, 50)
            )
            self.record_metric(
                'ai_confidence_score',
                np.random.uniform(0.8, 0.99)
            )
            self.record_metric(
                'ai_model_drift_score',
                np.random.uniform(0.01, 0.1)
            )
            self.record_metric(
                'ai_feature_importance',
                np.random.uniform(0.5, 1.0)
            )
            self.record_metric(
                'ai_prediction_variance',
                np.random.uniform(0.01, 0.2)
            )
            self.record_metric(
                'ai_ensemble_agreement',
                np.random.uniform(0.8, 1.0)
            )
            self.record_metric(
                'ai_learning_rate',
                np.random.uniform(0.001, 0.01)
            )
            self.record_metric(
                'ai_memory_usage_mb',
                np.random.uniform(100, 500)
            )
            self.record_metric(
                'ai_gpu_utilization',
                np.random.uniform(30, 80)
            )

        except ValueError as e:
            self.logger.error(
                f"Error recording AI metrics: {str(e)}",
                exc_info=True
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error in AI metrics: {str(e)}",
                exc_info=True
            )

    async def _collect_trading_metrics(self):
        """Collect trading performance and execution metrics"""
        try:
            # Simulate trading metrics collection
            # In production, replace with actual trading metrics
            self.record_metric(
                'trades_executed_total',
                np.random.randint(1000, 5000)
            )
            self.record_metric(
                'trades_successful_total',
                np.random.randint(800, 4500)
            )
            self.record_metric(
                'trades_failed_total',
                np.random.randint(10, 100)
            )
            self.record_metric(
                'profit_total_sol',
                np.random.uniform(100, 1000)
            )
            self.record_metric(
                'current_balance_sol',
                np.random.uniform(5000, 10000)
            )
            self.record_metric(
                'active_positions',
                np.random.randint(5, 20)
            )
            self.record_metric(
                'avg_trade_duration_seconds',
                np.random.uniform(60, 300)
            )
            self.record_metric(
                'avg_profit_percent',
                np.random.uniform(0.5, 2.5)
            )
            self.record_metric(
                'win_rate_percent',
                np.random.uniform(55, 75)
            )
            self.record_metric(
                'profit_factor',
                np.random.uniform(1.5, 3.0)
            )
            self.record_metric(
                'max_drawdown_percent',
                np.random.uniform(5, 15)
            )
            self.record_metric(
                'sharpe_ratio',
                np.random.uniform(1.5, 3.0)
            )
            self.record_metric(
                'risk_adjusted_return',
                np.random.uniform(10, 25)
            )
            self.record_metric(
                'position_sizing_accuracy',
                np.random.uniform(85, 95)
            )
            self.record_metric(
                'market_timing_accuracy',
                np.random.uniform(60, 80)
            )
            self.record_metric(
                'liquidity_impact_score',
                np.random.uniform(0.8, 0.95)
            )
            self.record_metric(
                'slippage_avg_percent',
                np.random.uniform(0.1, 0.5)
            )
            self.record_metric(
                'gas_optimization_score',
                np.random.uniform(0.7, 0.9)
            )

        except ValueError as e:
            self.logger.error(
                f"Error recording trading metrics: {str(e)}",
                exc_info=True
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error in trading metrics: {str(e)}",
                exc_info=True
            )

    def get_metric_history(
        self,
        name: str,
        duration_minutes: int = 60
    ) -> List[Metric]:
        """Get historical metrics for specified duration"""
        if name not in self.metrics:
            return []

        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        return [
            m for m in self.metrics[name]
            if m.timestamp >= cutoff_time
        ]

    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """Get the most recent value for a metric"""
        if not self.metrics.get(name):
            return None
        return self.metrics[name][-1]

    def get_system_snapshot(self) -> SystemSnapshot:
        """Get comprehensive system state snapshot with predictions"""
        current_time = datetime.utcnow()
        metrics_history = []
        for metric_list in self.metrics.values():
            metrics_history.extend(list(metric_list))

        # Get latest system metrics
        snapshot = SystemSnapshot(
            timestamp=current_time,
            cpu_usage_percent=self._get_latest_value(
                'cpu_usage_percent',
                0.0
            ),
            memory_usage_percent=self._get_latest_value(
                'memory_usage_percent',
                0.0
            ),
            memory_usage_mb=self._get_latest_value(
                'memory_usage_mb',
                0.0
            ),
            disk_usage_percent=self._get_latest_value(
                'disk_usage_percent',
                0.0
            ),
            network_io_bytes={
                'sent': self._get_latest_value('network_bytes_sent', 0),
                'received': self._get_latest_value('network_bytes_recv', 0)
            },
            active_threads=self._get_latest_value('active_threads', 0),
            open_connections=self._get_latest_value(
                'open_file_descriptors',
                0
            ),
            component_health=self._assess_component_health(),
            trading_metrics=self._get_trading_metrics_snapshot(),
            performance_metrics=self._get_performance_metrics_snapshot(),
            predicted_system_load=self._predict_system_load(
                metrics_history
            ),
            anomaly_detected=bool(
                self._detect_anomalies(metrics_history)
            ),
            maintenance_recommended=self._check_maintenance_needs(),
            ai_model_health=self._assess_ai_model_health(),
            predictive_insights=self._generate_predictive_insights()
        )
        return snapshot

    def _get_latest_value(
        self,
        metric_name: str,
        default: Any = None
    ) -> Any:
        """Get latest value for a metric with default fallback"""
        metric = self.get_latest_metric(metric_name)
        return metric.value if metric else default

    def _analyze_metric_trend(self, metric_name: str) -> Dict[str, Any]:
        """Analyze trend direction and volatility for a metric"""
        try:
            # Get recent history (last hour)
            history = self.get_metric_history(metric_name, 60)
            if not history or len(history) < 2:
                return {
                    'direction': 'stable',
                    'volatility': 0.0,
                    'confidence': 0.0
                }

            # Extract values and timestamps
            values = np.array([m.value for m in history])
            timestamps = np.array([
                m.timestamp.timestamp() for m in history
            ])

            # Calculate trend using linear regression
            z = np.polyfit(timestamps, values, 1)
            slope = z[0]

            # Calculate volatility (standard deviation)
            volatility = float(np.std(values))

            # Determine trend direction and confidence
            abs_slope = abs(slope)
            if abs_slope < 0.001:
                direction = 'stable'
                confidence = 0.8
            else:
                direction = 'increasing' if slope > 0 else 'decreasing'
                confidence = min(1.0, abs_slope * 10)

            return {
                'direction': direction,
                'volatility': volatility,
                'confidence': confidence,
                'slope': float(slope)
            }

        except Exception as e:
            self.logger.error(
                f"Error analyzing trend for {metric_name}: {str(e)}",
                exc_info=True
            )
            return {
                'direction': 'unknown',
                'volatility': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }

    def _assess_component_health(self) -> Dict[str, str]:
        """Assess health status of system components"""
        health_status = {}
        
        try:
            # CPU health
            cpu_usage = self._get_latest_value('cpu_usage_percent', 0)
            health_status['cpu'] = (
                'critical' if cpu_usage > 90
                else 'warning' if cpu_usage > 75
                else 'healthy'
            )

            # Memory health
            mem_usage = self._get_latest_value('memory_usage_percent', 0)
            health_status['memory'] = (
                'critical' if mem_usage > 90
                else 'warning' if mem_usage > 80
                else 'healthy'
            )

            # Disk health
            disk_usage = self._get_latest_value('disk_usage_percent', 0)
            health_status['disk'] = (
                'critical' if disk_usage > 90
                else 'warning' if disk_usage > 75
                else 'healthy'
            )

            # Network health
            net_latency = self._get_latest_value('network_latency_ms', 0)
            health_status['network'] = (
                'critical' if net_latency > 1000
                else 'warning' if net_latency > 500
                else 'healthy'
            )

            # AI model health
            model_drift = self._get_latest_value('ai_model_drift_score', 0)
            health_status['ai_model'] = (
                'critical' if model_drift > 0.5
                else 'warning' if model_drift > 0.3
                else 'healthy'
            )

        except Exception as e:
            self.logger.error(
                f"Error assessing component health: {str(e)}",
                exc_info=True
            )
            health_status['error'] = str(e)

        return health_status

    def _get_trading_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current trading performance snapshot"""
        try:
            return {
                'total_trades': self._get_latest_value(
                    'trades_executed_total',
                    0
                ),
                'success_rate': self._get_latest_value(
                    'win_rate_percent',
                    0.0
                ),
                'profit_total': self._get_latest_value(
                    'profit_total_sol',
                    0.0
                ),
                'current_balance': self._get_latest_value(
                    'current_balance_sol',
                    0.0
                ),
                'active_positions': self._get_latest_value(
                    'active_positions',
                    0
                ),
                'performance_metrics': {
                    'sharpe_ratio': self._get_latest_value(
                        'sharpe_ratio',
                        0.0
                    ),
                    'max_drawdown': self._get_latest_value(
                        'max_drawdown_percent',
                        0.0
                    ),
                    'profit_factor': self._get_latest_value(
                        'profit_factor',
                        0.0
                    )
                }
            }
        except Exception as e:
            self.logger.error(
                f"Error getting trading metrics snapshot: {str(e)}",
                exc_info=True
            )
            return {}

    def _get_performance_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current system performance snapshot"""
        try:
            return {
                'cpu_usage': self._get_latest_value(
                    'cpu_usage_percent',
                    0.0
                ),
                'memory_usage': self._get_latest_value(
                    'memory_usage_percent',
                    0.0
                ),
                'disk_usage': self._get_latest_value(
                    'disk_usage_percent',
                    0.0
                ),
                'network_io': {
                    'sent': self._get_latest_value(
                        'network_bytes_sent',
                        0
                    ),
                    'received': self._get_latest_value(
                        'network_bytes_recv',
                        0
                    )
                }
            }
        except Exception as e:
            self.logger.error(
                f"Error getting performance metrics snapshot: {str(e)}",
                exc_info=True
            )
            return {}

    def _predict_system_load(
        self,
        metrics_history: List[Metric]
    ) -> float:
        """Predict future system load using historical metrics"""
        try:
            if not metrics_history:
                return 0.0

            # Extract CPU usage values and timestamps
            cpu_metrics = [
                m for m in metrics_history
                if m.name == 'cpu_usage_percent'
            ]
            if not cpu_metrics:
                return 0.0

            # Prepare data for prediction
            values = np.array([m.value for m in cpu_metrics])
            timestamps = np.array([
                m.timestamp.timestamp() for m in cpu_metrics
            ])

            # Simple linear regression for prediction
            if len(values) < 2:
                return values[0]

            # Use numpy's polyfit for linear regression
            z = np.polyfit(timestamps, values, 1)
            p = np.poly1d(z)

            # Predict 5 minutes into the future
            future_time = datetime.utcnow().timestamp() + 300
            prediction = float(p(future_time))

            # Ensure prediction is within reasonable bounds
            return max(0.0, min(100.0, prediction))

        except Exception as e:
            self.logger.error(
                f"Error predicting system load: {str(e)}",
                exc_info=True
            )
            return 0.0

    def _detect_anomalies(
        self,
        metrics_history: List[Metric]
    ) -> List[int]:
        """Detect anomalies in system metrics using Isolation Forest"""
        try:
            if not metrics_history:
                return []

            # Prepare data for anomaly detection
            data = []
            for metric in metrics_history:
                if metric.name in {
                    'cpu_usage_percent',
                    'memory_usage_percent',
                    'disk_usage_percent'
                }:
                    data.append([
                        metric.value,
                        metric.timestamp.timestamp()
                    ])

            if not data:
                return []

            # Convert to numpy array and scale
            X = np.array(data)
            X_scaled = self.scaler.fit_transform(X)

            # Detect anomalies
            predictions = self.anomaly_detector.fit_predict(X_scaled)
            anomaly_indices = np.where(predictions == -1)[0]

            return anomaly_indices.tolist()

        except Exception as e:
            self.logger.error(
                f"Error detecting anomalies: {str(e)}",
                exc_info=True
            )
            return []

    def _check_maintenance_needs(self) -> bool:
        """Check if system maintenance is recommended"""
        try:
            # Check system metrics thresholds
            cpu_high = self._get_latest_value(
                'cpu_usage_percent',
                0
            ) > 80
            mem_high = self._get_latest_value(
                'memory_usage_percent',
                0
            ) > 85
            disk_high = self._get_latest_value(
                'disk_usage_percent',
                0
            ) > 85

            # Check AI model health
            model_drift = self._get_latest_value(
                'ai_model_drift_score',
                0
            ) > 0.4
            low_accuracy = self._get_latest_value(
                'ai_prediction_accuracy',
                100
            ) < 80

            # Check trading performance
            high_failure = self._get_latest_value(
                'trades_failed_total',
                0
            ) > 100
            low_profit = self._get_latest_value(
                'profit_factor',
                2.0
            ) < 1.2

            return any([
                cpu_high,
                mem_high,
                disk_high,
                model_drift,
                low_accuracy,
                high_failure,
                low_profit
            ])

        except Exception as e:
            self.logger.error(
                f"Error checking maintenance needs: {str(e)}",
                exc_info=True
            )
            return False

    def _assess_ai_model_health(self) -> Dict[str, Any]:
        """Assess AI model health and performance metrics"""
        try:
            return {
                'accuracy': self._get_latest_value(
                    'ai_prediction_accuracy',
                    0.0
                ),
                'latency': self._get_latest_value(
                    'ai_model_latency_ms',
                    0.0
                ),
                'drift_score': self._get_latest_value(
                    'ai_model_drift_score',
                    0.0
                ),
                'confidence': self._get_latest_value(
                    'ai_confidence_score',
                    0.0
                ),
                'resource_usage': {
                    'memory_mb': self._get_latest_value(
                        'ai_memory_usage_mb',
                        0.0
                    ),
                    'gpu_utilization': self._get_latest_value(
                        'ai_gpu_utilization',
                        0.0
                    )
                }
            }
        except Exception as e:
            self.logger.error(
                f"Error assessing AI model health: {str(e)}",
                exc_info=True
            )
            return {}

    def _generate_predictive_insights(self) -> Dict[str, Any]:
        """Generate predictive insights for system optimization"""
        try:
            return {
                'system_health_prediction': {
                    'cpu_trend': self._analyze_metric_trend(
                        'cpu_usage_percent'
                    ),
                    'memory_trend': self._analyze_metric_trend(
                        'memory_usage_percent'
                    ),
                    'disk_trend': self._analyze_metric_trend(
                        'disk_usage_percent'
                    )
                },
                'trading_performance_prediction': {
                    'profit_trend': self._analyze_metric_trend(
                        'profit_total_sol'
                    ),
                    'win_rate_trend': self._analyze_metric_trend(
                        'win_rate_percent'
                    )
                },
                'ai_performance_prediction': {
                    'accuracy_trend': self._analyze_metric_trend(
                        'ai_prediction_accuracy'
                    ),
                    'latency_trend': self._analyze_metric_trend(
                        'ai_model_latency_ms'
                    )
                }
            }
        except Exception as e:
            self.logger.error(
                f"Error generating predictive insights: {str(e)}",
                exc_info=True
            )
            return {} 