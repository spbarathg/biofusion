"""
Smart Ape Mode - Data Analysis Tool
==================================

Analyzes comprehensive logs to generate insights and recommendations:
- Trade performance analysis
- AI model effectiveness evaluation
- Market pattern identification
- Optimization recommendations
- Correlation analysis
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class SmartApeDataAnalyzer:
    """Comprehensive data analysis for Smart Ape Mode bot"""
    
    def __init__(self, log_dir: str = "comprehensive_logs"):
        self.log_dir = Path(log_dir)
        self.analysis_results = {}
        
        # Load all data
        self.trading_data = []
        self.ai_predictions = []
        self.market_data = []
        self.system_performance = []
        self.errors = []
        self.decisions = []
        
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all comprehensive log data"""
        
        print("📊 Loading comprehensive data for analysis...")
        
        # Load JSON Lines files
        for file_path in self.log_dir.glob("*.jsonl"):
            try:
                with open(file_path, 'r') as f:
                    data = [json.loads(line) for line in f if line.strip()]
                
                if 'trading_decisions' in file_path.name:
                    self.trading_data.extend(data)
                elif 'ai_predictions' in file_path.name:
                    self.ai_predictions.extend(data)
                elif 'market_data' in file_path.name:
                    self.market_data.extend(data)
                elif 'system_performance' in file_path.name:
                    self.system_performance.extend(data)
                elif 'errors' in file_path.name:
                    self.errors.extend(data)
                elif 'decisions' in file_path.name:
                    self.decisions.extend(data)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"✅ Loaded:")
        print(f"   - {len(self.trading_data)} trading decisions")
        print(f"   - {len(self.ai_predictions)} AI predictions")
        print(f"   - {len(self.market_data)} market data points")
        print(f"   - {len(self.system_performance)} system metrics")
        print(f"   - {len(self.errors)} error logs")
        print(f"   - {len(self.decisions)} decision logs")
        print()
    
    def analyze_trading_performance(self) -> Dict[str, Any]:
        """Comprehensive trading performance analysis"""
        
        print("🔍 Analyzing trading performance...")
        
        if not self.trading_data:
            return {"error": "No trading data available"}
        
        # Convert to DataFrame
        df = pd.DataFrame(self.trading_data)
        
        # Basic statistics
        total_trades = len(df)
        buy_trades = len(df[df['decision'] == 'buy'])
        sell_trades = len(df[df['decision'] == 'sell'])
        hold_decisions = len(df[df['decision'] == 'hold'])
        skip_decisions = len(df[df['decision'] == 'skip'])
        
        # Performance by decision type
        decision_analysis = {
            'total_decisions': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'hold_decisions': hold_decisions,
            'skip_decisions': skip_decisions,
            'action_rate': (buy_trades + sell_trades) / total_trades if total_trades > 0 else 0
        }
        
        # AI signal analysis
        ai_effectiveness = self._analyze_ai_effectiveness(df)
        
        # Risk analysis
        risk_analysis = self._analyze_risk_patterns(df)
        
        # Market condition analysis
        market_analysis = self._analyze_market_conditions(df)
        
        # Token performance
        token_analysis = self._analyze_token_performance(df)
        
        # Time-based patterns
        time_analysis = self._analyze_time_patterns(df)
        
        analysis = {
            'summary': decision_analysis,
            'ai_effectiveness': ai_effectiveness,
            'risk_analysis': risk_analysis,
            'market_analysis': market_analysis,
            'token_analysis': token_analysis,
            'time_analysis': time_analysis,
            'recommendations': self._generate_trading_recommendations(df)
        }
        
        self.analysis_results['trading_performance'] = analysis
        return analysis
    
    def _analyze_ai_effectiveness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze AI model effectiveness"""
        
        ai_analysis = {
            'ml_predictor': {},
            'sentiment_analyzer': {},
            'technical_analyzer': {},
            'combined_effectiveness': {}
        }
        
        # ML Predictor Analysis
        ml_confidences = []
        ml_decisions = []
        
        for _, row in df.iterrows():
            ml_signals = row.get('ai_signals', {}).get('ml_predictor', {})
            if ml_signals.get('confidence'):
                ml_confidences.append(ml_signals['confidence'])
                ml_decisions.append(row['decision'])
        
        if ml_confidences:
            ai_analysis['ml_predictor'] = {
                'avg_confidence': np.mean(ml_confidences),
                'confidence_std': np.std(ml_confidences),
                'high_confidence_trades': len([c for c in ml_confidences if c > 0.7]),
                'low_confidence_trades': len([c for c in ml_confidences if c < 0.5])
            }
        
        # Sentiment Analysis
        sentiment_scores = []
        sentiment_decisions = []
        
        for _, row in df.iterrows():
            sentiment_signals = row.get('ai_signals', {}).get('sentiment_analyzer', {})
            if sentiment_signals.get('overall_sentiment') is not None:
                sentiment_scores.append(sentiment_signals['overall_sentiment'])
                sentiment_decisions.append(row['decision'])
        
        if sentiment_scores:
            ai_analysis['sentiment_analyzer'] = {
                'avg_sentiment': np.mean(sentiment_scores),
                'sentiment_std': np.std(sentiment_scores),
                'positive_sentiment_trades': len([s for s in sentiment_scores if s > 0.2]),
                'negative_sentiment_trades': len([s for s in sentiment_scores if s < -0.2]),
                'neutral_sentiment_trades': len([s for s in sentiment_scores if -0.2 <= s <= 0.2])
            }
        
        # Technical Analysis
        technical_signals = []
        technical_decisions = []
        
        for _, row in df.iterrows():
            tech_signals = row.get('ai_signals', {}).get('technical_analyzer', {})
            if tech_signals.get('overall_signal') is not None:
                technical_signals.append(tech_signals['overall_signal'])
                technical_decisions.append(row['decision'])
        
        if technical_signals:
            ai_analysis['technical_analyzer'] = {
                'avg_signal': np.mean(technical_signals),
                'signal_std': np.std(technical_signals),
                'strong_buy_signals': len([s for s in technical_signals if s > 0.5]),
                'strong_sell_signals': len([s for s in technical_signals if s < -0.5]),
                'neutral_signals': len([s for s in technical_signals if -0.2 <= s <= 0.2])
            }
        
        return ai_analysis
    
    def _analyze_risk_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk assessment patterns"""
        
        position_sizes = []
        risk_scores = []
        max_losses = []
        
        for _, row in df.iterrows():
            risk_data = row.get('risk_assessment', {})
            if risk_data.get('position_size'):
                position_sizes.append(risk_data['position_size'])
            if risk_data.get('risk_score'):
                risk_scores.append(risk_data['risk_score'])
            if risk_data.get('max_loss'):
                max_losses.append(risk_data['max_loss'])
        
        risk_analysis = {}
        
        if position_sizes:
            risk_analysis['position_sizing'] = {
                'avg_position_size': np.mean(position_sizes),
                'max_position_size': np.max(position_sizes),
                'min_position_size': np.min(position_sizes),
                'position_size_std': np.std(position_sizes)
            }
        
        if risk_scores:
            risk_analysis['risk_scoring'] = {
                'avg_risk_score': np.mean(risk_scores),
                'high_risk_trades': len([r for r in risk_scores if r > 0.7]),
                'low_risk_trades': len([r for r in risk_scores if r < 0.3]),
                'risk_score_std': np.std(risk_scores)
            }
        
        return risk_analysis
    
    def _analyze_market_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market condition patterns"""
        
        market_analysis = {
            'price_movements': {},
            'volume_patterns': {},
            'liquidity_analysis': {}
        }
        
        # Price movement analysis
        price_changes = []
        volumes = []
        
        for _, row in df.iterrows():
            market_data = row.get('market_data', {})
            if market_data.get('price_change_24h'):
                price_changes.append(market_data['price_change_24h'])
            if market_data.get('volume_24h'):
                volumes.append(market_data['volume_24h'])
        
        if price_changes:
            market_analysis['price_movements'] = {
                'avg_price_change': np.mean(price_changes),
                'volatility': np.std(price_changes),
                'positive_movements': len([p for p in price_changes if p > 0]),
                'negative_movements': len([p for p in price_changes if p < 0])
            }
        
        if volumes:
            market_analysis['volume_patterns'] = {
                'avg_volume': np.mean(volumes),
                'volume_std': np.std(volumes),
                'high_volume_threshold': np.percentile(volumes, 75),
                'low_volume_threshold': np.percentile(volumes, 25)
            }
        
        return market_analysis
    
    def _analyze_token_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by token"""
        
        token_stats = defaultdict(lambda: {
            'trades': 0,
            'buy_decisions': 0,
            'sell_decisions': 0,
            'avg_confidence': [],
            'avg_sentiment': []
        })
        
        for _, row in df.iterrows():
            token = row.get('token_symbol')
            if token:
                token_stats[token]['trades'] += 1
                
                if row['decision'] == 'buy':
                    token_stats[token]['buy_decisions'] += 1
                elif row['decision'] == 'sell':
                    token_stats[token]['sell_decisions'] += 1
                
                # AI confidence
                ml_confidence = row.get('ai_signals', {}).get('ml_predictor', {}).get('confidence')
                if ml_confidence:
                    token_stats[token]['avg_confidence'].append(ml_confidence)
                
                # Sentiment
                sentiment = row.get('ai_signals', {}).get('sentiment_analyzer', {}).get('overall_sentiment')
                if sentiment is not None:
                    token_stats[token]['avg_sentiment'].append(sentiment)
        
        # Process statistics
        token_analysis = {}
        for token, stats in token_stats.items():
            token_analysis[token] = {
                'total_trades': stats['trades'],
                'buy_ratio': stats['buy_decisions'] / stats['trades'] if stats['trades'] > 0 else 0,
                'sell_ratio': stats['sell_decisions'] / stats['trades'] if stats['trades'] > 0 else 0,
                'avg_confidence': np.mean(stats['avg_confidence']) if stats['avg_confidence'] else 0,
                'avg_sentiment': np.mean(stats['avg_sentiment']) if stats['avg_sentiment'] else 0
            }
        
        return token_analysis
    
    def _analyze_time_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time-based trading patterns"""
        
        if df.empty:
            return {}
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Hourly patterns
        hourly_trades = df.groupby('hour').size().to_dict()
        hourly_buy_ratio = df[df['decision'] == 'buy'].groupby('hour').size() / df.groupby('hour').size()
        
        # Daily patterns
        daily_trades = df.groupby('day_of_week').size().to_dict()
        
        return {
            'hourly_patterns': {
                'trade_distribution': hourly_trades,
                'peak_hours': sorted(hourly_trades, key=hourly_trades.get, reverse=True)[:3],
                'quiet_hours': sorted(hourly_trades, key=hourly_trades.get)[:3]
            },
            'daily_patterns': {
                'trade_distribution': daily_trades,
                'most_active_days': sorted(daily_trades, key=daily_trades.get, reverse=True)[:3]
            }
        }
    
    def _generate_trading_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        # Analyze AI confidence patterns
        high_conf_trades = 0
        total_trades = len(df)
        
        for _, row in df.iterrows():
            ml_confidence = row.get('ai_signals', {}).get('ml_predictor', {}).get('confidence', 0)
            if ml_confidence > 0.7:
                high_conf_trades += 1
        
        if high_conf_trades / total_trades < 0.3:
            recommendations.append("Consider increasing minimum confidence threshold for ML predictions")
        
        # Analyze position sizing
        position_sizes = []
        for _, row in df.iterrows():
            pos_size = row.get('risk_assessment', {}).get('position_size')
            if pos_size:
                position_sizes.append(pos_size)
        
        if position_sizes and np.mean(position_sizes) > 0.1:
            recommendations.append("Consider reducing average position size for better risk management")
        
        # Analyze sentiment correlation
        sentiment_buy_correlation = 0
        sentiment_trades = 0
        
        for _, row in df.iterrows():
            sentiment = row.get('ai_signals', {}).get('sentiment_analyzer', {}).get('overall_sentiment')
            if sentiment is not None:
                sentiment_trades += 1
                if sentiment > 0.2 and row['decision'] == 'buy':
                    sentiment_buy_correlation += 1
        
        if sentiment_trades > 0 and sentiment_buy_correlation / sentiment_trades < 0.6:
            recommendations.append("Improve sentiment-based decision alignment")
        
        # Add more recommendations based on patterns
        if len(df) > 50:
            action_rate = len(df[df['decision'].isin(['buy', 'sell'])]) / len(df)
            if action_rate < 0.1:
                recommendations.append("Consider lowering thresholds to increase trading frequency")
            elif action_rate > 0.5:
                recommendations.append("Consider raising thresholds to be more selective")
        
        return recommendations
    
    def analyze_ai_model_performance(self) -> Dict[str, Any]:
        """Analyze AI model performance over time"""
        
        print("🤖 Analyzing AI model performance...")
        
        if not self.ai_predictions:
            return {"error": "No AI prediction data available"}
        
        df = pd.DataFrame(self.ai_predictions)
        
        # Performance by model
        model_performance = {}
        
        for model_name in df['model_name'].unique():
            model_data = df[df['model_name'] == model_name]
            
            confidences = model_data['model_performance'].apply(lambda x: x.get('confidence', 0))
            inference_times = model_data['model_performance'].apply(lambda x: x.get('inference_time_ms', 0))
            
            model_performance[model_name] = {
                'total_predictions': len(model_data),
                'avg_confidence': confidences.mean(),
                'confidence_trend': confidences.tolist()[-10:],  # Last 10 predictions
                'avg_inference_time': inference_times.mean(),
                'inference_time_trend': inference_times.tolist()[-10:]
            }
        
        self.analysis_results['ai_performance'] = model_performance
        return model_performance
    
    def analyze_system_health(self) -> Dict[str, Any]:
        """Analyze system performance and health"""
        
        print("🖥️ Analyzing system health...")
        
        if not self.system_performance:
            return {"error": "No system performance data available"}
        
        df = pd.DataFrame(self.system_performance)
        
        # Extract metrics
        cpu_usage = df['system_metrics'].apply(lambda x: x.get('cpu_usage', 0))
        memory_usage = df['system_metrics'].apply(lambda x: x.get('memory_usage', 0))
        
        system_analysis = {
            'performance_summary': {
                'avg_cpu_usage': cpu_usage.mean(),
                'max_cpu_usage': cpu_usage.max(),
                'avg_memory_usage': memory_usage.mean(),
                'max_memory_usage': memory_usage.max(),
                'cpu_spikes': len(cpu_usage[cpu_usage > 80]),
                'memory_spikes': len(memory_usage[memory_usage > 80])
            },
            'health_trends': {
                'cpu_trend': cpu_usage.tolist()[-20:],  # Last 20 measurements
                'memory_trend': memory_usage.tolist()[-20:]
            },
            'optimization_recommendations': []
        }
        
        # Generate system recommendations
        if cpu_usage.mean() > 70:
            system_analysis['optimization_recommendations'].append("High CPU usage detected - consider optimizing algorithms")
        
        if memory_usage.mean() > 80:
            system_analysis['optimization_recommendations'].append("High memory usage detected - implement memory cleanup")
        
        if len(cpu_usage[cpu_usage > 90]) > 3:
            system_analysis['optimization_recommendations'].append("Multiple CPU spikes detected - investigate performance bottlenecks")
        
        self.analysis_results['system_health'] = system_analysis
        return system_analysis
    
    def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns and frequency"""
        
        print("🚨 Analyzing error patterns...")
        
        if not self.errors:
            return {"message": "No errors detected - excellent!"}
        
        df = pd.DataFrame(self.errors)
        
        # Error analysis
        error_types = df['error_info'].apply(lambda x: x.get('type', 'unknown')).value_counts().to_dict()
        error_severity = df['error_info'].apply(lambda x: x.get('severity', 'unknown')).value_counts().to_dict()
        
        # Time-based error analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        hourly_errors = df.groupby('hour').size().to_dict()
        
        error_analysis = {
            'error_summary': {
                'total_errors': len(df),
                'error_types': error_types,
                'error_severity': error_severity,
                'error_rate_per_hour': len(df) / max(1, df['timestamp'].nunique())
            },
            'patterns': {
                'hourly_distribution': hourly_errors,
                'peak_error_hours': sorted(hourly_errors, key=hourly_errors.get, reverse=True)[:3]
            },
            'recommendations': []
        }
        
        # Generate error-based recommendations
        if len(df) > 10:
            error_analysis['recommendations'].append("High error count detected - review error handling")
        
        most_common_error = max(error_types, key=error_types.get) if error_types else None
        if most_common_error and error_types[most_common_error] > 5:
            error_analysis['recommendations'].append(f"Address recurring {most_common_error} errors")
        
        self.analysis_results['error_analysis'] = error_analysis
        return error_analysis
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report"""
        
        print("📋 Generating comprehensive analysis report...")
        
        # Run all analyses
        trading_perf = self.analyze_trading_performance()
        ai_perf = self.analyze_ai_model_performance()
        system_health = self.analyze_system_health()
        error_analysis = self.analyze_error_patterns()
        
        # Generate overall insights
        overall_insights = self._generate_overall_insights()
        
        # Create comprehensive report
        report = {
            'report_metadata': {
                'generation_time': datetime.now().isoformat(),
                'data_period': self._get_data_period(),
                'total_data_points': {
                    'trading_decisions': len(self.trading_data),
                    'ai_predictions': len(self.ai_predictions),
                    'system_metrics': len(self.system_performance),
                    'errors': len(self.errors)
                }
            },
            'trading_performance': trading_perf,
            'ai_model_performance': ai_perf,
            'system_health': system_health,
            'error_analysis': error_analysis,
            'overall_insights': overall_insights,
            'priority_recommendations': self._get_priority_recommendations()
        }
        
        # Save report
        report_file = self.log_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Comprehensive report saved to: {report_file}")
        
        return report
    
    def _generate_overall_insights(self) -> List[str]:
        """Generate overall insights from all data"""
        
        insights = []
        
        # Trading insights
        if self.trading_data:
            total_decisions = len(self.trading_data)
            buy_decisions = len([t for t in self.trading_data if t.get('decision') == 'buy'])
            
            if buy_decisions / total_decisions > 0.3:
                insights.append("High trading activity detected - bot is actively finding opportunities")
            else:
                insights.append("Conservative trading approach - bot is being selective")
        
        # AI insights
        if self.ai_predictions:
            avg_confidence = np.mean([p.get('model_performance', {}).get('confidence', 0) for p in self.ai_predictions])
            if avg_confidence > 0.7:
                insights.append("AI models showing high confidence in predictions")
            elif avg_confidence < 0.5:
                insights.append("AI models showing low confidence - may need retraining")
        
        # System insights
        if self.system_performance:
            avg_cpu = np.mean([s.get('system_metrics', {}).get('cpu_usage', 0) for s in self.system_performance])
            if avg_cpu > 80:
                insights.append("System under high load - consider optimization")
            else:
                insights.append("System running efficiently")
        
        # Error insights
        if len(self.errors) == 0:
            insights.append("No errors detected - system running smoothly")
        elif len(self.errors) > 10:
            insights.append("Multiple errors detected - review system stability")
        
        return insights
    
    def _get_priority_recommendations(self) -> List[Dict[str, str]]:
        """Get prioritized recommendations"""
        
        recommendations = []
        
        # High priority system issues
        if self.system_performance:
            avg_cpu = np.mean([s.get('system_metrics', {}).get('cpu_usage', 0) for s in self.system_performance])
            if avg_cpu > 85:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'System',
                    'recommendation': 'Optimize CPU usage - system running at high load'
                })
        
        # AI model issues
        if self.ai_predictions:
            avg_confidence = np.mean([p.get('model_performance', {}).get('confidence', 0) for p in self.ai_predictions])
            if avg_confidence < 0.5:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'AI Models',
                    'recommendation': 'Review and retrain AI models - low confidence detected'
                })
        
        # Trading performance
        if self.trading_data:
            action_rate = len([t for t in self.trading_data if t.get('decision') in ['buy', 'sell']]) / len(self.trading_data)
            if action_rate < 0.05:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Trading',
                    'recommendation': 'Consider lowering thresholds - very low trading activity'
                })
        
        # Error handling
        if len(self.errors) > 5:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Errors',
                'recommendation': 'Address recurring errors to improve stability'
            })
        
        return recommendations
    
    def _get_data_period(self) -> Dict[str, str]:
        """Get the data period covered"""
        
        all_timestamps = []
        
        for data_set in [self.trading_data, self.ai_predictions, self.system_performance]:
            for item in data_set:
                if 'timestamp' in item:
                    all_timestamps.append(item['timestamp'])
        
        if all_timestamps:
            return {
                'start': min(all_timestamps),
                'end': max(all_timestamps)
            }
        
        return {'start': 'unknown', 'end': 'unknown'}

if __name__ == "__main__":
    # Demo usage
    analyzer = SmartApeDataAnalyzer()
    
    if analyzer.trading_data or analyzer.ai_predictions:
        report = analyzer.generate_comprehensive_report()
        print("\n📊 Analysis Complete!")
        print(f"Generated comprehensive report with {len(report)} sections")
        
        # Show summary
        if 'overall_insights' in report:
            print("\n🔍 Key Insights:")
            for insight in report['overall_insights']:
                print(f"   • {insight}")
        
        if 'priority_recommendations' in report:
            print("\n🎯 Priority Recommendations:")
            for rec in report['priority_recommendations']:
                print(f"   • [{rec['priority']}] {rec['category']}: {rec['recommendation']}")
    else:
        print("⚠️ No data found for analysis. Run the bot first to generate data!")