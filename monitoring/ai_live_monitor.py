"""
Smart Ape Mode - Live AI Model Monitor
=====================================

Real-time terminal interface for AI model monitoring with:
- Live model performance metrics
- Accuracy tracking and alerts
- Model health monitoring
- Drift detection
"""

import time
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import os
import sys

# Try to import colorama for colored output
try:
    from colorama import init, Fore, Back, Style
    init()
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Mock colorama classes
    class MockColor:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    Fore = Back = Style = MockColor()

# Try to import the AI model monitor
try:
    from ai_model_monitor import AIModelMonitor
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False
    print("Warning: ai_model_monitor.py not found. Please ensure it's in the same directory.")

class LiveAIMonitor:
    """Live terminal interface for AI model monitoring"""
    
    def __init__(self, refresh_interval: int = 5):
        self.refresh_interval = refresh_interval
        self.monitor = None
        self.last_metrics = {}
        self.alert_history = []
        self.start_time = datetime.now()
        
        if MONITOR_AVAILABLE:
            self.monitor = AIModelMonitor()
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_color(self, value: float, thresholds: Dict[str, float], reverse: bool = False) -> str:
        """Get color based on value and thresholds"""
        if not COLORS_AVAILABLE:
            return ""
        
        if reverse:
            # For metrics where lower is better (like error rates)
            if value <= thresholds.get('good', 0.7):
                return Fore.GREEN
            elif value <= thresholds.get('warning', 0.85):
                return Fore.YELLOW
            else:
                return Fore.RED
        else:
            # For metrics where higher is better (like accuracy)
            if value >= thresholds.get('good', 0.7):
                return Fore.GREEN
            elif value >= thresholds.get('warning', 0.55):
                return Fore.YELLOW
            else:
                return Fore.RED
    
    def format_metric(self, label: str, value: Any, unit: str = "", color: str = "", width: int = 25) -> str:
        """Format a metric for display"""
        if isinstance(value, float):
            formatted_value = f"{value:.3f}{unit}"
        else:
            formatted_value = f"{value}{unit}"
        
        reset = Style.RESET_ALL if COLORS_AVAILABLE else ""
        return f"{label:<{width}}: {color}{formatted_value}{reset}"
    
    def display_header(self):
        """Display header with title and system info"""
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🤖 SMART APE MODE - AI MODEL LIVE MONITOR{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        
        uptime = datetime.now() - self.start_time
        uptime_str = f"{int(uptime.total_seconds() // 3600):02d}:{int((uptime.total_seconds() % 3600) // 60):02d}:{int(uptime.total_seconds() % 60):02d}"
        
        print(f"🕐 Monitor Uptime: {uptime_str}")
        print(f"🔄 Refresh Rate: {self.refresh_interval}s")
        print(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def display_ml_predictor_metrics(self, metrics: Dict[str, Any]):
        """Display ML Predictor metrics"""
        print(f"{Fore.BLUE}🧠 ML PREDICTOR{Style.RESET_ALL}")
        print("-" * 40)
        
        if not metrics:
            print(f"{Fore.RED}❌ No data available{Style.RESET_ALL}")
            print()
            return
        
        # Accuracy metrics
        thresholds = {'good': 0.7, 'warning': 0.55}
        lstm_color = self.get_color(metrics.get('lstm_accuracy', 0), thresholds)
        rl_color = self.get_color(metrics.get('rl_accuracy', 0), thresholds)
        combined_color = self.get_color(metrics.get('combined_accuracy', 0), thresholds)
        direction_color = self.get_color(metrics.get('direction_accuracy', 0), thresholds)
        
        print(self.format_metric("LSTM Accuracy", metrics.get('lstm_accuracy', 0), "%", lstm_color))
        print(self.format_metric("RL Accuracy", metrics.get('rl_accuracy', 0), "%", rl_color))
        print(self.format_metric("Combined Accuracy", metrics.get('combined_accuracy', 0), "%", combined_color))
        print(self.format_metric("Direction Accuracy", metrics.get('direction_accuracy', 0), "%", direction_color))
        
        # Performance metrics
        inference_color = self.get_color(metrics.get('lstm_inference_time', 0), {'good': 500, 'warning': 1000}, reverse=True)
        mae_color = self.get_color(metrics.get('price_prediction_mae', 0), {'good': 0.03, 'warning': 0.05}, reverse=True)
        
        print(self.format_metric("Inference Time", metrics.get('lstm_inference_time', 0), "ms", inference_color))
        print(self.format_metric("Price MAE", metrics.get('price_prediction_mae', 0), "%", mae_color))
        print(self.format_metric("Signal Strength", metrics.get('signal_strength_avg', 0), "", ""))
        print()
    
    def display_sentiment_analyzer_metrics(self, metrics: Dict[str, Any]):
        """Display Sentiment Analyzer metrics"""
        print(f"{Fore.MAGENTA}💭 SENTIMENT ANALYZER{Style.RESET_ALL}")
        print("-" * 40)
        
        if not metrics:
            print(f"{Fore.RED}❌ No data available{Style.RESET_ALL}")
            print()
            return
        
        # Accuracy metrics
        thresholds = {'good': 0.7, 'warning': 0.55}
        overall_color = self.get_color(metrics.get('overall_sentiment_accuracy', 0), thresholds)
        twitter_color = self.get_color(metrics.get('twitter_accuracy', 0), thresholds)
        reddit_color = self.get_color(metrics.get('reddit_accuracy', 0), thresholds)
        
        print(self.format_metric("Overall Accuracy", metrics.get('overall_sentiment_accuracy', 0), "%", overall_color))
        print(self.format_metric("Twitter Accuracy", metrics.get('twitter_accuracy', 0), "%", twitter_color))
        print(self.format_metric("Reddit Accuracy", metrics.get('reddit_accuracy', 0), "%", reddit_color))
        
        # Data quality metrics
        data_quality = (metrics.get('twitter_data_quality', 0) + metrics.get('reddit_data_quality', 0)) / 2
        quality_color = self.get_color(data_quality, {'good': 0.8, 'warning': 0.6})
        correlation_color = self.get_color(metrics.get('sentiment_trading_correlation', 0), {'good': 0.6, 'warning': 0.4})
        
        print(self.format_metric("Data Quality", data_quality, "%", quality_color))
        print(self.format_metric("Mentions Processed", metrics.get('total_mentions_processed', 0), "", ""))
        print(self.format_metric("Trading Correlation", metrics.get('sentiment_trading_correlation', 0), "%", correlation_color))
        print()
    
    def display_technical_analyzer_metrics(self, metrics: Dict[str, Any]):
        """Display Technical Analyzer metrics"""
        print(f"{Fore.YELLOW}📊 TECHNICAL ANALYZER{Style.RESET_ALL}")
        print("-" * 40)
        
        if not metrics:
            print(f"{Fore.RED}❌ No data available{Style.RESET_ALL}")
            print()
            return
        
        # Signal accuracy metrics
        thresholds = {'good': 0.7, 'warning': 0.55}
        overall_color = self.get_color(metrics.get('overall_signal_accuracy', 0), thresholds)
        rsi_color = self.get_color(metrics.get('rsi_signal_accuracy', 0), thresholds)
        macd_color = self.get_color(metrics.get('macd_signal_accuracy', 0), thresholds)
        bollinger_color = self.get_color(metrics.get('bollinger_signal_accuracy', 0), thresholds)
        
        print(self.format_metric("Overall Signal Accuracy", metrics.get('overall_signal_accuracy', 0), "%", overall_color))
        print(self.format_metric("RSI Accuracy", metrics.get('rsi_signal_accuracy', 0), "%", rsi_color))
        print(self.format_metric("MACD Accuracy", metrics.get('macd_signal_accuracy', 0), "%", macd_color))
        print(self.format_metric("Bollinger Accuracy", metrics.get('bollinger_signal_accuracy', 0), "%", bollinger_color))
        
        # Advanced metrics
        momentum_color = self.get_color(metrics.get('momentum_accuracy', 0), thresholds)
        breakout_color = self.get_color(metrics.get('breakout_prediction_accuracy', 0), thresholds)
        
        print(self.format_metric("Momentum Accuracy", metrics.get('momentum_accuracy', 0), "%", momentum_color))
        print(self.format_metric("Breakout Accuracy", metrics.get('breakout_prediction_accuracy', 0), "%", breakout_color))
        print()
    
    def display_system_health(self, health: Dict[str, Any]):
        """Display system health metrics"""
        print(f"{Fore.GREEN}🖥️ SYSTEM HEALTH{Style.RESET_ALL}")
        print("-" * 40)
        
        if not health:
            print(f"{Fore.RED}❌ No data available{Style.RESET_ALL}")
            print()
            return
        
        # Resource usage
        cpu_color = self.get_color(health.get('cpu_usage', 0), {'good': 50, 'warning': 80}, reverse=True)
        memory_color = self.get_color(health.get('memory_usage', 0), {'good': 60, 'warning': 80}, reverse=True)
        
        print(self.format_metric("CPU Usage", health.get('cpu_usage', 0), "%", cpu_color))
        print(self.format_metric("Memory Usage", health.get('memory_usage', 0), "%", memory_color))
        print(self.format_metric("Active Models", health.get('active_models', 0), "", ""))
        print(self.format_metric("Uptime", health.get('uptime_hours', 0), "h", ""))
        print()
    
    def display_alerts(self):
        """Display recent alerts"""
        print(f"{Fore.RED}🚨 RECENT ALERTS{Style.RESET_ALL}")
        print("-" * 40)
        
        # Read recent alerts from log file
        alert_file = Path("logs/ai_model_alerts.log")
        recent_alerts = []
        
        if alert_file.exists():
            try:
                with open(alert_file, 'r') as f:
                    lines = f.readlines()
                    # Get last 5 alerts
                    for line in lines[-5:]:
                        try:
                            alert = json.loads(line.strip())
                            recent_alerts.append(alert)
                        except:
                            continue
            except Exception as e:
                print(f"Error reading alerts: {e}")
        
        if not recent_alerts:
            print(f"{Fore.GREEN}✅ No recent alerts{Style.RESET_ALL}")
        else:
            for alert in recent_alerts[-3:]:  # Show last 3
                timestamp = alert.get('timestamp', 'Unknown')
                message = alert.get('message', 'Unknown alert')
                alert_type = alert.get('type', 'info')
                
                color = Fore.RED if alert_type in ['performance', 'system'] else Fore.YELLOW
                print(f"{color}⚠️ {timestamp}: {message}{Style.RESET_ALL}")
        
        print()
    
    def display_footer(self):
        """Display footer with controls"""
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Controls: Ctrl+C to stop | Data refreshes every {self.refresh_interval}s{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Logs: logs/ai_*.log | Web Dashboard: http://localhost:5001{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current AI model metrics"""
        if self.monitor:
            try:
                return self.monitor.get_current_metrics()
            except Exception as e:
                return {'error': str(e)}
        else:
            # Return mock data for demonstration
            return {
                'ml_predictor': {
                    'lstm_accuracy': 0.73,
                    'rl_accuracy': 0.68,
                    'combined_accuracy': 0.71,
                    'direction_accuracy': 0.75,
                    'lstm_inference_time': 245,
                    'price_prediction_mae': 0.034,
                    'signal_strength_avg': 0.62
                },
                'sentiment_analyzer': {
                    'overall_sentiment_accuracy': 0.69,
                    'twitter_accuracy': 0.72,
                    'reddit_accuracy': 0.66,
                    'twitter_data_quality': 0.85,
                    'reddit_data_quality': 0.78,
                    'total_mentions_processed': 234,
                    'sentiment_trading_correlation': 0.58
                },
                'technical_analyzer': {
                    'overall_signal_accuracy': 0.74,
                    'rsi_signal_accuracy': 0.71,
                    'macd_signal_accuracy': 0.69,
                    'bollinger_signal_accuracy': 0.76,
                    'momentum_accuracy': 0.72,
                    'breakout_prediction_accuracy': 0.67
                },
                'system_health': {
                    'cpu_usage': 45.2,
                    'memory_usage': 62.8,
                    'active_models': 3,
                    'uptime_hours': 12.5
                }
            }
    
    async def run_monitor(self):
        """Run the live monitoring interface"""
        print(f"{Fore.GREEN}🤖 Starting AI Model Live Monitor...{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Press Ctrl+C to stop{Style.RESET_ALL}")
        print()
        time.sleep(2)
        
        while True:
            try:
                # Get current metrics
                metrics = await self.get_current_metrics()
                
                # Clear screen and display
                self.clear_screen()
                self.display_header()
                
                # Display model metrics
                self.display_ml_predictor_metrics(metrics.get('ml_predictor', {}))
                self.display_sentiment_analyzer_metrics(metrics.get('sentiment_analyzer', {}))
                self.display_technical_analyzer_metrics(metrics.get('technical_analyzer', {}))
                self.display_system_health(metrics.get('system_health', {}))
                self.display_alerts()
                self.display_footer()
                
                # Wait for next refresh
                await asyncio.sleep(self.refresh_interval)
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}🛑 Stopping AI Model Monitor...{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
                await asyncio.sleep(5)

def main():
    """Main function"""
    print("🤖 Smart Ape Mode - AI Model Live Monitor")
    print("=========================================")
    
    # Check if monitoring files exist
    if not MONITOR_AVAILABLE:
        print("⚠️ Warning: AI model monitor not available.")
        print("Running in demo mode with mock data.")
        print()
    
    # Get refresh interval from user
    try:
        interval = int(input("Enter refresh interval in seconds (default 5): ") or "5")
        if interval < 1:
            interval = 5
    except:
        interval = 5
    
    print(f"🔄 Refresh rate set to {interval} seconds")
    print()
    
    # Create and run monitor
    monitor = LiveAIMonitor(refresh_interval=interval)
    
    try:
        asyncio.run(monitor.run_monitor())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")

if __name__ == "__main__":
    main()