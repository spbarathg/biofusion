"""
Smart Ape Mode - Comprehensive Logging Integration
=================================================

Integration script to add comprehensive logging to your existing bot:
- Easy integration with existing code
- Automatic data collection
- Analysis integration
- Export functions for feeding data to AI analysis
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# Import our comprehensive logging system
try:
    from comprehensive_logger import get_comprehensive_logger, log_trade, log_prediction, log_error_with_context, log_decision
    from data_analyzer import SmartApeDataAnalyzer
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    print("Warning: Comprehensive logging modules not found")

class SmartApeComprehensiveIntegration:
    """Integration class for comprehensive logging and analysis"""
    
    def __init__(self, bot_instance=None):
        self.bot_instance = bot_instance
        self.logger = get_comprehensive_logger() if LOGGING_AVAILABLE else None
        self.analyzer = None
        
        # Setup integration hooks
        self._setup_integration_hooks()
    
    def _setup_integration_hooks(self):
        """Setup automatic logging hooks for the bot"""
        
        if not self.logger:
            print("⚠️ Comprehensive logging not available")
            return
        
        print("🔗 Setting up comprehensive logging integration...")
        
        # Log integration start
        self.logger.log_decision_tree(
            decision_type="system_integration",
            inputs={"integration_time": datetime.now().isoformat()},
            decision_process=[{"step": "comprehensive_logging_enabled"}],
            final_decision="enabled",
            confidence=1.0,
            reasoning="Comprehensive logging system integrated with Smart Ape Mode bot"
        )
    
    # =================== TRADING INTEGRATION ===================
    
    def log_trading_analysis(self, 
                           token_symbol: str,
                           market_analysis: Dict[str, Any],
                           ai_predictions: Dict[str, Any],
                           risk_assessment: Dict[str, Any],
                           final_decision: str,
                           reasoning: Dict[str, Any]):
        """Log complete trading analysis with all context"""
        
        if not self.logger:
            return
        
        # Extract AI signals
        ai_signals = {
            'ml_prediction': ai_predictions.get('ml_prediction'),
            'ml_confidence': ai_predictions.get('ml_confidence'),
            'predicted_price': ai_predictions.get('predicted_price'),
            'predicted_direction': ai_predictions.get('direction'),
            'hold_duration': ai_predictions.get('hold_duration'),
            
            'sentiment_score': ai_predictions.get('sentiment_score'),
            'sentiment_confidence': ai_predictions.get('sentiment_confidence'),
            'social_buzz': ai_predictions.get('social_buzz'),
            'sentiment_momentum': ai_predictions.get('sentiment_momentum'),
            'mention_count': ai_predictions.get('mention_count'),
            
            'technical_signal': ai_predictions.get('technical_signal'),
            'rsi': ai_predictions.get('rsi'),
            'macd': ai_predictions.get('macd'),
            'bollinger_position': ai_predictions.get('bollinger_position'),
            'volume_spike': ai_predictions.get('volume_spike'),
            'breakout_probability': ai_predictions.get('breakout_probability')
        }
        
        # Log the trading decision
        self.logger.log_trading_decision(
            token_symbol=token_symbol,
            decision=final_decision,
            reasoning=reasoning,
            market_data=market_analysis,
            ai_signals=ai_signals,
            risk_assessment=risk_assessment
        )
    
    def log_trade_execution_with_outcome(self,
                                       trade_id: str,
                                       token_symbol: str,
                                       entry_details: Dict[str, Any],
                                       exit_details: Optional[Dict[str, Any]] = None,
                                       performance_metrics: Optional[Dict[str, Any]] = None):
        """Log trade execution and outcome"""
        
        if not self.logger:
            return
        
        # Log execution
        if entry_details:
            self.logger.log_trade_execution(
                trade_id=trade_id,
                execution_details=entry_details,
                transaction_hash=entry_details.get('tx_hash', ''),
                gas_used=entry_details.get('gas_used', 0),
                actual_price=entry_details.get('actual_price', 0),
                slippage=entry_details.get('slippage', 0),
                execution_time=entry_details.get('execution_time', 0)
            )
        
        # Log outcome if trade is complete
        if exit_details and performance_metrics:
            self.logger.log_trade_outcome(
                trade_id=trade_id,
                entry_price=performance_metrics.get('entry_price', 0),
                exit_price=performance_metrics.get('exit_price', 0),
                profit_loss=performance_metrics.get('profit_loss', 0),
                profit_loss_percentage=performance_metrics.get('profit_loss_pct', 0),
                hold_duration=timedelta(minutes=performance_metrics.get('hold_duration_minutes', 0)),
                exit_reason=performance_metrics.get('exit_reason', 'unknown'),
                ai_prediction_accuracy=performance_metrics.get('ai_accuracy', {})
            )
    
    # =================== AI MODEL INTEGRATION ===================
    
    def log_ai_model_prediction(self,
                              model_name: str,
                              token_symbol: str,
                              input_data: Dict[str, Any],
                              prediction_result: Dict[str, Any],
                              performance_metrics: Dict[str, Any]):
        """Log AI model predictions with validation"""
        
        if not self.logger:
            return
        
        self.logger.log_ai_prediction(
            model_name=model_name,
            token_symbol=token_symbol,
            prediction=prediction_result,
            input_features=input_data,
            confidence=performance_metrics.get('confidence', 0),
            inference_time=performance_metrics.get('inference_time_ms', 0)
        )
    
    def log_ai_model_health(self,
                          model_name: str,
                          accuracy_metrics: Dict[str, float],
                          performance_metrics: Dict[str, float],
                          drift_detection: Dict[str, float]):
        """Log AI model health and performance"""
        
        if not self.logger:
            return
        
        self.logger.log_ai_model_performance(
            model_name=model_name,
            accuracy_metrics=accuracy_metrics,
            performance_metrics=performance_metrics,
            drift_metrics=drift_detection
        )
    
    # =================== MARKET DATA INTEGRATION ===================
    
    def log_comprehensive_market_data(self,
                                    token_symbol: str,
                                    price_info: Dict[str, Any],
                                    volume_info: Dict[str, Any],
                                    liquidity_info: Dict[str, Any],
                                    social_info: Dict[str, Any],
                                    technical_indicators: Dict[str, Any]):
        """Log comprehensive market data"""
        
        if not self.logger:
            return
        
        self.logger.log_market_data(
            token_symbol=token_symbol,
            price_data=price_info,
            volume_data=volume_info,
            liquidity_data=liquidity_info,
            social_data=social_info,
            technical_indicators=technical_indicators
        )
    
    # =================== ERROR INTEGRATION ===================
    
    def log_bot_error(self,
                     error_type: str,
                     error_message: str,
                     context: Dict[str, Any],
                     severity: str = "error"):
        """Log bot errors with context"""
        
        if not self.logger:
            return
        
        self.logger.log_error(
            error_type=error_type,
            error_message=error_message,
            context=context,
            traceback_info="",  # Will be filled by the logger
            severity=severity
        )
    
    # =================== DECISION INTEGRATION ===================
    
    def log_bot_decision(self,
                        decision_type: str,
                        decision_inputs: Dict[str, Any],
                        decision_steps: List[Dict[str, Any]],
                        final_decision: Any,
                        confidence: float,
                        reasoning: str):
        """Log bot decision-making process"""
        
        if not self.logger:
            return
        
        self.logger.log_decision_tree(
            decision_type=decision_type,
            inputs=decision_inputs,
            decision_process=decision_steps,
            final_decision=final_decision,
            confidence=confidence,
            reasoning=reasoning
        )
    
    # =================== ANALYSIS FUNCTIONS ===================
    
    def run_performance_analysis(self) -> Dict[str, Any]:
        """Run comprehensive performance analysis"""
        
        if not LOGGING_AVAILABLE:
            return {"error": "Analysis tools not available"}
        
        print("📊 Running comprehensive performance analysis...")
        
        # Initialize analyzer
        self.analyzer = SmartApeDataAnalyzer()
        
        # Generate comprehensive report
        report = self.analyzer.generate_comprehensive_report()
        
        return report
    
    def export_data_for_ai_analysis(self, 
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None,
                                  format: str = "json") -> str:
        """Export data in format suitable for AI analysis"""
        
        if not self.logger:
            return "Logging not available"
        
        print("📤 Exporting data for AI analysis...")
        
        # Generate analysis dataset
        dataset = self.logger.generate_analysis_dataset(start_date, end_date)
        
        # Save in requested format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        if format.lower() == "json":
            export_file = Path(f"ai_analysis_export_{timestamp}.json")
            with open(export_file, 'w') as f:
                json.dump(dataset, f, indent=2, default=str)
        
        elif format.lower() == "csv":
            # Export as multiple CSV files
            export_dir = Path(f"ai_analysis_export_{timestamp}")
            export_dir.mkdir(exist_ok=True)
            
            # Convert each data type to CSV
            if dataset.get('trading_data'):
                trading_df = pd.DataFrame(dataset['trading_data'])
                trading_df.to_csv(export_dir / 'trading_data.csv', index=False)
            
            if dataset.get('ai_predictions'):
                ai_df = pd.DataFrame(dataset['ai_predictions'])
                ai_df.to_csv(export_dir / 'ai_predictions.csv', index=False)
            
            # Save metadata
            with open(export_dir / 'metadata.json', 'w') as f:
                json.dump(dataset['metadata'], f, indent=2, default=str)
            
            export_file = export_dir
        
        print(f"✅ Data exported to: {export_file}")
        return str(export_file)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get quick analysis summary"""
        
        if not self.logger:
            return {"error": "Logging not available"}
        
        # Get current data counts
        session_data = {
            'session_id': self.logger.session_id,
            'session_duration_hours': (datetime.now() - self.logger.session_start).total_seconds() / 3600,
            'data_counts': {
                'trades': self.logger.trade_counter,
                'predictions': self.logger.prediction_counter,
                'errors': self.logger.error_counter
            },
            'rates': {
                'trades_per_hour': self.logger.trade_counter / max(1, (datetime.now() - self.logger.session_start).total_seconds() / 3600),
                'error_rate': self.logger.error_counter / max(1, self.logger.trade_counter + self.logger.prediction_counter)
            }
        }
        
        return session_data
    
    # =================== INTEGRATION HELPERS ===================
    
    def create_analysis_prompt(self) -> str:
        """Create a prompt for AI analysis"""
        
        # Export current data
        export_file = self.export_data_for_ai_analysis()
        
        # Get summary
        summary = self.get_analysis_summary()
        
        prompt = f"""
I have comprehensive trading bot data that I'd like you to analyze for optimization opportunities.

**Data Export File:** {export_file}

**Session Summary:**
- Session ID: {summary.get('session_id', 'unknown')}
- Duration: {summary.get('session_duration_hours', 0):.2f} hours
- Total Trades: {summary.get('data_counts', {}).get('trades', 0)}
- AI Predictions: {summary.get('data_counts', {}).get('predictions', 0)}
- Errors: {summary.get('data_counts', {}).get('errors', 0)}
- Trading Rate: {summary.get('rates', {}).get('trades_per_hour', 0):.2f} trades/hour
- Error Rate: {summary.get('rates', {}).get('error_rate', 0):.3f}

**Please analyze this data and provide:**
1. Trading performance insights
2. AI model effectiveness evaluation
3. Risk management assessment
4. Market timing patterns
5. Error pattern analysis
6. Specific optimization recommendations
7. Parameter tuning suggestions
8. Code improvements

Focus on actionable insights that can improve profitability and reduce risk.
"""
        
        return prompt
    
    def stop_comprehensive_logging(self):
        """Stop the comprehensive logging system"""
        
        if self.logger:
            self.logger.stop_logging()
            print("🛑 Comprehensive logging stopped")
    
    # =================== CONTEXT MANAGERS ===================
    
    def log_operation_context(self, operation_name: str, context: Dict[str, Any] = None):
        """Context manager for logging operations"""
        
        if self.logger:
            return self.logger.log_operation(operation_name, context)
        else:
            # Return a dummy context manager
            from contextlib import nullcontext
            return nullcontext()

# =================== CONVENIENCE FUNCTIONS ===================

def setup_comprehensive_logging(bot_instance=None) -> SmartApeComprehensiveIntegration:
    """Setup comprehensive logging for Smart Ape Mode bot"""
    
    print("🚀 Setting up comprehensive logging system...")
    integration = SmartApeComprehensiveIntegration(bot_instance)
    
    if integration.logger:
        print("✅ Comprehensive logging active!")
        print(f"📁 Logs will be saved to: comprehensive_logs/")
        print(f"🆔 Session ID: {integration.logger.session_id}")
        print()
        print("📋 Available logging functions:")
        print("   - integration.log_trading_analysis()")
        print("   - integration.log_ai_model_prediction()")
        print("   - integration.log_comprehensive_market_data()")
        print("   - integration.log_bot_error()")
        print("   - integration.log_bot_decision()")
        print()
        print("📊 Analysis functions:")
        print("   - integration.run_performance_analysis()")
        print("   - integration.export_data_for_ai_analysis()")
        print("   - integration.create_analysis_prompt()")
        print()
    else:
        print("❌ Comprehensive logging setup failed")
    
    return integration

def quick_analysis() -> Dict[str, Any]:
    """Run quick analysis on existing data"""
    
    if not LOGGING_AVAILABLE:
        return {"error": "Analysis tools not available"}
    
    analyzer = SmartApeDataAnalyzer()
    return analyzer.generate_comprehensive_report()

def export_for_analysis(format: str = "json") -> str:
    """Quick export function"""
    
    integration = SmartApeComprehensiveIntegration()
    return integration.export_data_for_ai_analysis(format=format)

# =================== DEMO AND TESTING ===================

def demo_comprehensive_logging():
    """Demo of comprehensive logging system"""
    
    print("🎬 Demo: Comprehensive Logging System")
    print("=====================================")
    
    # Setup
    integration = setup_comprehensive_logging()
    
    if not integration.logger:
        print("❌ Demo failed - logging not available")
        return
    
    # Demo trading log
    integration.log_trading_analysis(
        token_symbol="DEMO",
        market_analysis={
            'current_price': 0.001,
            'volume_24h': 50000,
            'price_change_24h': 0.15,
            'market_cap': 1000000,
            'liquidity': 50000
        },
        ai_predictions={
            'ml_prediction': 'buy',
            'ml_confidence': 0.78,
            'predicted_price': 0.0012,
            'direction': 'up',
            'sentiment_score': 0.6,
            'technical_signal': 0.7
        },
        risk_assessment={
            'position_size': 0.05,
            'max_loss': 0.02,
            'risk_score': 0.3
        },
        final_decision="buy",
        reasoning={'signal_strength': 0.8, 'risk_acceptable': True}
    )
    
    # Demo AI prediction log
    integration.log_ai_model_prediction(
        model_name="ML_Predictor",
        token_symbol="DEMO",
        input_data={'price': 0.001, 'volume': 50000, 'sentiment': 0.6},
        prediction_result={'prediction': 'buy', 'price_target': 0.0012},
        performance_metrics={'confidence': 0.78, 'inference_time_ms': 245}
    )
    
    # Demo error log
    integration.log_bot_error(
        error_type="demo_error",
        error_message="This is a demo error for testing",
        context={'demo': True, 'test_error': True},
        severity="info"
    )
    
    print("\n✅ Demo logging complete!")
    print(f"🆔 Session ID: {integration.logger.session_id}")
    
    # Quick analysis
    summary = integration.get_analysis_summary()
    print(f"\n📊 Session Summary:")
    print(f"   - Trades logged: {summary['data_counts']['trades']}")
    print(f"   - Predictions logged: {summary['data_counts']['predictions']}")
    print(f"   - Errors logged: {summary['data_counts']['errors']}")
    
    # Create analysis prompt
    prompt = integration.create_analysis_prompt()
    print(f"\n📝 Analysis prompt created ({len(prompt)} characters)")
    print("Ready for AI analysis!")
    
    return integration

if __name__ == "__main__":
    # Run demo
    demo_integration = demo_comprehensive_logging()
    
    print("\n🎯 Next Steps:")
    print("1. Integrate this with your Smart Ape Mode bot")
    print("2. Run the bot to collect real data")
    print("3. Use create_analysis_prompt() to get data for AI analysis")
    print("4. Feed the exported data back to me for optimization recommendations")