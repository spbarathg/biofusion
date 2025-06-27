"""
Smart Ape Mode - AI Model Dashboard
==================================

Web dashboard for real-time AI model monitoring:
- Model performance charts and metrics
- Prediction accuracy tracking
- Model health and resource usage
- Drift detection visualization
"""

from flask import Flask, render_template_string, jsonify, request
import json
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from ai_model_monitor import AIModelMonitor
import time

app = Flask(__name__)

# Global monitor instance
monitor = None
monitoring_thread = None

# HTML Template for AI Model Dashboard
AI_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Ape Mode - AI Model Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 15px;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            background: linear-gradient(45deg, #00ff88, #00aaff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .model-card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .model-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .model-title {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }
        
        .status-active { background: #00ff88; box-shadow: 0 0 10px #00ff88; }
        .status-warning { background: #ffaa00; box-shadow: 0 0 10px #ffaa00; }
        .status-error { background: #ff4444; box-shadow: 0 0 10px #ff4444; }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }
        
        .metric-label {
            font-weight: 500;
            opacity: 0.9;
        }
        
        .metric-value {
            font-weight: bold;
            font-family: 'Courier New', monospace;
        }
        
        .performance-chart {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
        
        .system-health {
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .health-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .health-metric {
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
        }
        
        .health-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .refresh-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #00ff88;
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .refresh-btn:hover {
            background: #00aa66;
            transform: scale(1.05);
        }
        
        .last-update {
            text-align: center;
            opacity: 0.7;
            margin-top: 20px;
            font-size: 0.9em;
        }
        
        .alert-section {
            background: rgba(255,100,100,0.2);
            border: 1px solid rgba(255,100,100,0.5);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .alert-item {
            background: rgba(0,0,0,0.3);
            padding: 10px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #ff4444;
        }
        
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .refresh-btn {
                position: static;
                margin: 20px auto;
                display: block;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Smart Ape Mode - AI Model Dashboard</h1>
        <p>Real-time monitoring of ML Predictor, Sentiment Analyzer, and Technical Analyzer</p>
    </div>
    
    <button class="refresh-btn" onclick="refreshData()">🔄 Refresh</button>
    
    <div class="dashboard-grid">
        <!-- ML Predictor Card -->
        <div class="model-card">
            <div class="model-title">
                🧠 ML Predictor
                <span class="status-indicator" id="ml-status"></span>
            </div>
            <div class="metric-row">
                <span class="metric-label">LSTM Accuracy:</span>
                <span class="metric-value" id="lstm-accuracy">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">RL Accuracy:</span>
                <span class="metric-value" id="rl-accuracy">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Combined Accuracy:</span>
                <span class="metric-value" id="combined-accuracy">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Direction Accuracy:</span>
                <span class="metric-value" id="direction-accuracy">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Inference Time:</span>
                <span class="metric-value" id="inference-time">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Price MAE:</span>
                <span class="metric-value" id="price-mae">-</span>
            </div>
        </div>
        
        <!-- Sentiment Analyzer Card -->
        <div class="model-card">
            <div class="model-title">
                💭 Sentiment Analyzer
                <span class="status-indicator" id="sentiment-status"></span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Overall Accuracy:</span>
                <span class="metric-value" id="sentiment-accuracy">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Twitter Accuracy:</span>
                <span class="metric-value" id="twitter-accuracy">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Reddit Accuracy:</span>
                <span class="metric-value" id="reddit-accuracy">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Mentions Processed:</span>
                <span class="metric-value" id="mentions-processed">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Data Quality:</span>
                <span class="metric-value" id="data-quality">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Trading Correlation:</span>
                <span class="metric-value" id="trading-correlation">-</span>
            </div>
        </div>
        
        <!-- Technical Analyzer Card -->
        <div class="model-card">
            <div class="model-title">
                📊 Technical Analyzer
                <span class="status-indicator" id="technical-status"></span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Overall Signal Accuracy:</span>
                <span class="metric-value" id="signal-accuracy">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">RSI Accuracy:</span>
                <span class="metric-value" id="rsi-accuracy">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">MACD Accuracy:</span>
                <span class="metric-value" id="macd-accuracy">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Bollinger Accuracy:</span>
                <span class="metric-value" id="bollinger-accuracy">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Momentum Accuracy:</span>
                <span class="metric-value" id="momentum-accuracy">-</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Breakout Accuracy:</span>
                <span class="metric-value" id="breakout-accuracy">-</span>
            </div>
        </div>
    </div>
    
    <!-- Performance Charts -->
    <div class="performance-chart">
        <h3>📈 Model Accuracy Trends</h3>
        <div class="chart-container">
            <canvas id="accuracyChart"></canvas>
        </div>
    </div>
    
    <div class="performance-chart">
        <h3>⚡ Model Performance Metrics</h3>
        <div class="chart-container">
            <canvas id="performanceChart"></canvas>
        </div>
    </div>
    
    <!-- System Health -->
    <div class="system-health">
        <h3>🖥️ System Health</h3>
        <div class="health-grid">
            <div class="health-metric">
                <div class="health-value" id="cpu-usage">-</div>
                <div>CPU Usage</div>
            </div>
            <div class="health-metric">
                <div class="health-value" id="memory-usage">-</div>
                <div>Memory Usage</div>
            </div>
            <div class="health-metric">
                <div class="health-value" id="active-models">-</div>
                <div>Active Models</div>
            </div>
            <div class="health-metric">
                <div class="health-value" id="uptime">-</div>
                <div>Uptime (hours)</div>
            </div>
        </div>
    </div>
    
    <!-- Alerts Section -->
    <div class="alert-section" id="alerts-section" style="display: none;">
        <h3>⚠️ Recent Alerts</h3>
        <div id="alerts-container"></div>
    </div>
    
    <div class="last-update">
        Last updated: <span id="last-update">-</span>
    </div>

    <script>
        let accuracyChart, performanceChart;
        
        // Initialize charts
        function initCharts() {
            const ctx1 = document.getElementById('accuracyChart').getContext('2d');
            accuracyChart = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'ML Predictor',
                            borderColor: '#00ff88',
                            backgroundColor: 'rgba(0, 255, 136, 0.1)',
                            data: []
                        },
                        {
                            label: 'Sentiment Analyzer',
                            borderColor: '#00aaff',
                            backgroundColor: 'rgba(0, 170, 255, 0.1)',
                            data: []
                        },
                        {
                            label: 'Technical Analyzer',
                            borderColor: '#ffaa00',
                            backgroundColor: 'rgba(255, 170, 0, 0.1)',
                            data: []
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: 'white'
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: 'white'
                            },
                            grid: {
                                color: 'rgba(255,255,255,0.1)'
                            }
                        },
                        y: {
                            ticks: {
                                color: 'white'
                            },
                            grid: {
                                color: 'rgba(255,255,255,0.1)'
                            },
                            min: 0,
                            max: 1
                        }
                    }
                }
            });
            
            const ctx2 = document.getElementById('performanceChart').getContext('2d');
            performanceChart = new Chart(ctx2, {
                type: 'bar',
                data: {
                    labels: ['ML Predictor', 'Sentiment Analyzer', 'Technical Analyzer'],
                    datasets: [
                        {
                            label: 'Inference Time (ms)',
                            backgroundColor: 'rgba(255, 100, 100, 0.8)',
                            data: [0, 0, 0]
                        },
                        {
                            label: 'Data Quality',
                            backgroundColor: 'rgba(100, 255, 100, 0.8)',
                            data: [0, 0, 0]
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: 'white'
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: 'white'
                            },
                            grid: {
                                color: 'rgba(255,255,255,0.1)'
                            }
                        },
                        y: {
                            ticks: {
                                color: 'white'
                            },
                            grid: {
                                color: 'rgba(255,255,255,0.1)'
                            }
                        }
                    }
                }
            });
        }
        
        // Refresh data
        async function refreshData() {
            try {
                const response = await fetch('/api/ai_metrics');
                const data = await response.json();
                
                updateMetrics(data);
                updateCharts(data);
                updateSystemHealth(data);
                updateAlerts(data);
                
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
        
        function updateMetrics(data) {
            // ML Predictor metrics
            if (data.ml_predictor) {
                const ml = data.ml_predictor;
                document.getElementById('lstm-accuracy').textContent = (ml.lstm_accuracy * 100).toFixed(1) + '%';
                document.getElementById('rl-accuracy').textContent = (ml.rl_accuracy * 100).toFixed(1) + '%';
                document.getElementById('combined-accuracy').textContent = (ml.combined_accuracy * 100).toFixed(1) + '%';
                document.getElementById('direction-accuracy').textContent = (ml.direction_accuracy * 100).toFixed(1) + '%';
                document.getElementById('inference-time').textContent = ml.lstm_inference_time.toFixed(0) + 'ms';
                document.getElementById('price-mae').textContent = (ml.price_prediction_mae * 100).toFixed(2) + '%';
                
                // Status indicator
                const mlStatus = document.getElementById('ml-status');
                mlStatus.className = 'status-indicator ' + (ml.combined_accuracy > 0.7 ? 'status-active' : 
                                   ml.combined_accuracy > 0.5 ? 'status-warning' : 'status-error');
            }
            
            // Sentiment Analyzer metrics
            if (data.sentiment_analyzer) {
                const sentiment = data.sentiment_analyzer;
                document.getElementById('sentiment-accuracy').textContent = (sentiment.overall_sentiment_accuracy * 100).toFixed(1) + '%';
                document.getElementById('twitter-accuracy').textContent = (sentiment.twitter_accuracy * 100).toFixed(1) + '%';
                document.getElementById('reddit-accuracy').textContent = (sentiment.reddit_accuracy * 100).toFixed(1) + '%';
                document.getElementById('mentions-processed').textContent = sentiment.total_mentions_processed;
                document.getElementById('data-quality').textContent = ((sentiment.twitter_data_quality + sentiment.reddit_data_quality) / 2 * 100).toFixed(1) + '%';
                document.getElementById('trading-correlation').textContent = (sentiment.sentiment_trading_correlation * 100).toFixed(1) + '%';
                
                // Status indicator
                const sentimentStatus = document.getElementById('sentiment-status');
                sentimentStatus.className = 'status-indicator ' + (sentiment.overall_sentiment_accuracy > 0.7 ? 'status-active' : 
                                          sentiment.overall_sentiment_accuracy > 0.5 ? 'status-warning' : 'status-error');
            }
            
            // Technical Analyzer metrics
            if (data.technical_analyzer) {
                const technical = data.technical_analyzer;
                document.getElementById('signal-accuracy').textContent = (technical.overall_signal_accuracy * 100).toFixed(1) + '%';
                document.getElementById('rsi-accuracy').textContent = (technical.rsi_signal_accuracy * 100).toFixed(1) + '%';
                document.getElementById('macd-accuracy').textContent = (technical.macd_signal_accuracy * 100).toFixed(1) + '%';
                document.getElementById('bollinger-accuracy').textContent = (technical.bollinger_signal_accuracy * 100).toFixed(1) + '%';
                document.getElementById('momentum-accuracy').textContent = (technical.momentum_accuracy * 100).toFixed(1) + '%';
                document.getElementById('breakout-accuracy').textContent = (technical.breakout_prediction_accuracy * 100).toFixed(1) + '%';
                
                // Status indicator
                const technicalStatus = document.getElementById('technical-status');
                technicalStatus.className = 'status-indicator ' + (technical.overall_signal_accuracy > 0.7 ? 'status-active' : 
                                           technical.overall_signal_accuracy > 0.5 ? 'status-warning' : 'status-error');
            }
        }
        
        function updateCharts(data) {
            const now = new Date().toLocaleTimeString();
            
            // Update accuracy chart
            if (accuracyChart && data.ml_predictor && data.sentiment_analyzer && data.technical_analyzer) {
                accuracyChart.data.labels.push(now);
                accuracyChart.data.datasets[0].data.push(data.ml_predictor.combined_accuracy);
                accuracyChart.data.datasets[1].data.push(data.sentiment_analyzer.overall_sentiment_accuracy);
                accuracyChart.data.datasets[2].data.push(data.technical_analyzer.overall_signal_accuracy);
                
                // Keep only last 20 data points
                if (accuracyChart.data.labels.length > 20) {
                    accuracyChart.data.labels.shift();
                    accuracyChart.data.datasets.forEach(dataset => dataset.data.shift());
                }
                
                accuracyChart.update('none');
            }
            
            // Update performance chart
            if (performanceChart && data.ml_predictor && data.sentiment_analyzer && data.technical_analyzer) {
                performanceChart.data.datasets[0].data = [
                    data.ml_predictor.lstm_inference_time,
                    100, // Placeholder for sentiment inference time
                    50   // Placeholder for technical inference time
                ];
                
                performanceChart.data.datasets[1].data = [
                    data.ml_predictor.combined_accuracy * 100,
                    (data.sentiment_analyzer.twitter_data_quality + data.sentiment_analyzer.reddit_data_quality) / 2 * 100,
                    data.technical_analyzer.overall_signal_accuracy * 100
                ];
                
                performanceChart.update('none');
            }
        }
        
        function updateSystemHealth(data) {
            if (data.system_health) {
                const health = data.system_health;
                document.getElementById('cpu-usage').textContent = health.cpu_usage.toFixed(1) + '%';
                document.getElementById('memory-usage').textContent = health.memory_usage.toFixed(1) + '%';
                document.getElementById('active-models').textContent = health.active_models;
                document.getElementById('uptime').textContent = health.uptime_hours.toFixed(1) + 'h';
            }
        }
        
        function updateAlerts(data) {
            // This would be populated with actual alerts from the monitoring system
            // For now, it's a placeholder
        }
        
        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            refreshData();
            
            // Auto-refresh every 30 seconds
            setInterval(refreshData, 30000);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main AI model dashboard"""
    return render_template_string(AI_DASHBOARD_TEMPLATE)

@app.route('/api/ai_metrics')
def get_ai_metrics():
    """Get current AI model metrics"""
    global monitor
    
    if monitor:
        try:
            metrics = monitor.get_current_metrics()
            return jsonify(metrics)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Monitor not initialized'}), 503

@app.route('/api/start_monitoring')
def start_monitoring():
    """Start the AI model monitoring"""
    global monitor, monitoring_thread
    
    try:
        if not monitor:
            monitor = AIModelMonitor()
        
        if not monitoring_thread or not monitoring_thread.is_alive():
            def run_monitor():
                asyncio.run(monitor.start_monitoring())
            
            monitoring_thread = threading.Thread(target=run_monitor, daemon=True)
            monitoring_thread.start()
        
        return jsonify({'status': 'started', 'message': 'AI model monitoring started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_monitoring')
def stop_monitoring():
    """Stop the AI model monitoring"""
    global monitor
    
    try:
        if monitor:
            monitor.stop_monitoring()
        return jsonify({'status': 'stopped', 'message': 'AI model monitoring stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_dashboard(host='127.0.0.1', port=5001, debug=False):
    """Run the AI model dashboard"""
    print(f"🤖 Starting AI Model Dashboard on http://{host}:{port}")
    print("📊 Features:")
    print("   - Real-time model performance metrics")
    print("   - Accuracy and inference time tracking")
    print("   - Model drift detection")
    print("   - System health monitoring")
    print("   - Interactive performance charts")
    print()
    
    # Start monitoring automatically
    global monitor, monitoring_thread
    monitor = AIModelMonitor()
    
    def run_monitor():
        asyncio.run(monitor.start_monitoring())
    
    monitoring_thread = threading.Thread(target=run_monitor, daemon=True)
    monitoring_thread.start()
    
    app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == '__main__':
    run_dashboard()