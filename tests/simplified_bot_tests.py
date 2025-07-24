"""
SIMPLIFIED BOT TESTS - LEAN MATHEMATICAL CORE
============================================

Focused test suite for the simplified trading bot architecture.
Tests only the essential mathematical components.
"""

import pytest
import asyncio
import unittest
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from worker_ant_v1.trading.simplified_trading_bot import (
    SimplifiedTradingBot, 
    SimplifiedConfig, 
    TradingMetrics
)


class SimplifiedBotTestSuite:
    """Test suite for simplified trading bot"""
    
    def __init__(self):
        self.bot = None
        self.config = SimplifiedConfig(initial_capital_sol=1.0)
    
    async def run_all_tests(self):
        """Run all simplified bot tests"""
        print("🧪 Running Simplified Bot Test Suite...")
        
        test_methods = [
            self._test_configuration,
            self._test_mathematical_core,
            self._test_three_stage_pipeline,
            self._test_risk_management,
            self._test_metrics_tracking
        ]
        
        passed = 0
        failed = 0
        
        for test_method in test_methods:
            try:
                await test_method()
                print(f"✅ {test_method.__name__}")
                passed += 1
            except Exception as e:
                print(f"❌ {test_method.__name__}: {e}")
                failed += 1
        
        print(f"\n📊 Test Results: {passed} passed, {failed} failed")
        return failed == 0
    
    async def _test_configuration(self):
        """Test simplified configuration"""
        config = SimplifiedConfig(
            initial_capital_sol=2.0,
            acceptable_rel_threshold=0.15,
            hunt_threshold=0.65,
            kelly_fraction=0.3
        )
        
        assert config.initial_capital_sol == 2.0
        assert config.acceptable_rel_threshold == 0.15
        assert config.hunt_threshold == 0.65
        assert config.kelly_fraction == 0.3
        assert config.max_position_percent == 0.20  # Default
    
    async def _test_mathematical_core(self):
        """Test core mathematical functions"""
        bot = SimplifiedTradingBot(self.config)
        
        # Test Naive Bayes calculation
        signals = {
            'sentiment_score': 0.7,
            'volume_momentum': 1.5,
            'price_momentum': 0.1,
            'rsi_oversold': 1.0
        }
        
        win_probability = bot._calculate_naive_bayes_probability(signals)
        assert 0.0 <= win_probability <= 1.0, "Win probability should be between 0 and 1"
        
        # Test Kelly Criterion calculation
        position_size = bot._calculate_kelly_position_size(0.6, 1.0)
        assert position_size > 0, "Position size should be positive for profitable probability"
        assert position_size <= 0.2, "Position size should not exceed max position percent"
    
    async def _test_three_stage_pipeline(self):
        """Test the three-stage decision pipeline components"""
        bot = SimplifiedTradingBot(self.config)
        
        # Mock the core components
        bot.rug_detector = Mock()
        bot.devils_advocate = Mock()
        bot.sentiment_ai = Mock()
        bot.technical_analyzer = Mock()
        bot.trading_engine = Mock()
        
        # Mock rug detector response (low risk)
        mock_rug_result = Mock()
        mock_rug_result.detection_level.value = 'low'
        mock_rug_result.overall_risk = 0.2
        bot.rug_detector.analyze_token = AsyncMock(return_value=mock_rug_result)
        
        # Mock WCCA response (no veto)
        mock_wcca_result = {'veto': False, 'max_rel': 0.05}
        bot.devils_advocate.conduct_pre_mortem_analysis = AsyncMock(return_value=mock_wcca_result)
        
        # Mock sentiment response
        mock_sentiment = Mock()
        mock_sentiment.sentiment_score = 0.7
        mock_sentiment.confidence = 0.8
        mock_sentiment.social_buzz_score = 0.6
        bot.sentiment_ai.analyze_sentiment = AsyncMock(return_value=mock_sentiment)
        
        # Mock technical analysis
        mock_technical = Mock()
        mock_technical.rsi = 25  # Oversold
        bot.technical_analyzer.analyze_token = AsyncMock(return_value=mock_technical)
        
        # Mock trading engine
        bot.trading_engine.execute_buy_order = AsyncMock(return_value={
            'success': True,
            'execution_price': 0.5
        })
        
        # Test opportunity processing
        opportunity = {
            'token_address': 'test_token_123',
            'token_symbol': 'TEST',
            'token_name': 'Test Token',
            'volume_change_24h': 2.0,
            'price_change_24h': 0.1
        }
        
        await bot._process_opportunity_pipeline(opportunity)
        
        # Verify the three stages were called
        bot.rug_detector.analyze_token.assert_called_once()
        bot.devils_advocate.conduct_pre_mortem_analysis.assert_called_once()
        bot.sentiment_ai.analyze_sentiment.assert_called_once()
        bot.technical_analyzer.analyze_token.assert_called_once()
        bot.trading_engine.execute_buy_order.assert_called_once()
    
    async def _test_risk_management(self):
        """Test risk management features"""
        bot = SimplifiedTradingBot(self.config)
        
        # Test position should close conditions
        position = {
            'token_address': 'test_token',
            'token_symbol': 'TEST',
            'entry_price': 1.0,
            'stop_loss_price': 0.95,
            'target_profit': 1.15,
            'max_hold_until': datetime.now() + timedelta(hours=1)
        }
        
        # Mock current price below stop loss
        with patch.object(bot, 'trading_engine') as mock_engine:
            mock_engine.get_token_price = AsyncMock(return_value=0.90)
            should_close, reason = await bot._should_close_position(position)
            assert should_close == True
            assert reason == "stop_loss"
        
        # Mock current price above target
        with patch.object(bot, 'trading_engine') as mock_engine:
            mock_engine.get_token_price = AsyncMock(return_value=1.20)
            should_close, reason = await bot._should_close_position(position)
            assert should_close == True
            assert reason == "profit_target"
        
        # Mock expired hold time
        position['max_hold_until'] = datetime.now() - timedelta(hours=1)
        with patch.object(bot, 'trading_engine') as mock_engine:
            mock_engine.get_token_price = AsyncMock(return_value=1.05)
            should_close, reason = await bot._should_close_position(position)
            assert should_close == True
            assert reason == "max_hold_time"
    
    async def _test_metrics_tracking(self):
        """Test metrics tracking functionality"""
        metrics = TradingMetrics(current_capital_sol=1.0)
        
        # Test initial state
        assert metrics.trades_executed == 0
        assert metrics.successful_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.current_capital_sol == 1.0
        
        # Simulate successful trade
        metrics.trades_executed = 1
        metrics.successful_trades = 1
        metrics.total_profit_sol = 0.1
        
        # Calculate win rate
        win_rate = metrics.successful_trades / metrics.trades_executed
        assert win_rate == 1.0, "Win rate should be 100% for 1 successful trade out of 1"
        
        # Simulate losing trade
        metrics.trades_executed = 2
        metrics.total_profit_sol = 0.05  # Net profit after loss
        
        win_rate = metrics.successful_trades / metrics.trades_executed
        assert win_rate == 0.5, "Win rate should be 50% for 1 successful trade out of 2"


async def main():
    """Run the simplified bot test suite"""
    suite = SimplifiedBotTestSuite()
    success = await suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main()) 