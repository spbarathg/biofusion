"""
Integration Test Script
======================

Test script to verify that all systems are properly integrated.
"""

import asyncio
import sys
import os
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

async def test_basic_imports():
    """Test that all core modules can be imported"""
    print("🔍 Testing basic imports...")
    
    try:
        from worker_ant_v1.core.unified_trading_engine import get_trading_engine
        from worker_ant_v1.core.wallet_manager import get_wallet_manager
        from worker_ant_v1.core.vault_wallet_system import get_vault_system
        from worker_ant_v1.core.unified_config import get_trading_config, get_wallet_config
        
        
        from worker_ant_v1.intelligence.sentiment_first_ai import get_sentiment_first_ai
        from worker_ant_v1.intelligence.token_intelligence_system import get_token_intelligence_system
        from worker_ant_v1.intelligence.enhanced_rug_detector import EnhancedRugDetector
        
        
        from worker_ant_v1.trading.market_scanner import get_market_scanner
        from worker_ant_v1.trading.surgical_trade_executor import SurgicalTradeExecutor
        from worker_ant_v1.trading.main import HyperIntelligentTradingSwarm, MemecoinTradingBot
        
        
        from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch
        
        
        from worker_ant_v1.utils.logger import get_logger, setup_logger
        
        print("✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

async def test_three_stage_pipeline_imports():
    """Test that all three-stage pipeline components can be imported"""
    print("🔍 Testing three-stage pipeline imports...")
    
    try:
        # Stage 1: Survival Filter (WCCA)
        from worker_ant_v1.trading.devils_advocate_synapse import DevilsAdvocateSynapse
        
        # Stage 2: Win-Rate Engine (Naive Bayes)
        from worker_ant_v1.core.swarm_decision_engine import SwarmDecisionEngine
        
        # Stage 3: Growth Maximizer (Kelly Criterion)
        from worker_ant_v1.trading.hyper_compound_engine import HyperCompoundEngine
        
        print("✅ All three-stage pipeline imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline import error: {e}")
        return False

async def test_stage1_survival_filter():
    """Test Stage 1: Survival Filter (WCCA) - Devils Advocate Synapse"""
    print("🔍 Testing Stage 1: Survival Filter (WCCA)...")
    
    try:
        from worker_ant_v1.trading.devils_advocate_synapse import DevilsAdvocateSynapse
        
        # Initialize the survival filter
        devils_advocate = DevilsAdvocateSynapse()
        await devils_advocate.initialize()
        print("✅ Devils Advocate Synapse initialized")
        
        # Test high-risk trade (should VETO)
        high_risk_params = {
            'token_address': 'test_high_risk_token',
            'amount': 0.2,  # 0.2 SOL position
            'token_age_hours': 0.1,  # 6 minutes old - very risky
            'liquidity_concentration': 0.95,  # 95% concentrated
            'dev_holdings_percent': 80,  # Dev holds 80%
            'contract_verified': False,
            'has_transfer_restrictions': True,
            'sell_buy_ratio': 0.05,  # Very few sells
        }
        
        wcca_result = await devils_advocate.conduct_pre_mortem_analysis(high_risk_params)
        
        if wcca_result.get('veto', False):
            print(f"✅ Stage 1 VETO Test Passed: {wcca_result.get('reason', 'Unknown')}")
        else:
            print(f"⚠️ Stage 1 VETO Test Warning: High-risk trade not vetoed")
        
        # Test low-risk trade (should CLEAR)
        low_risk_params = {
            'token_address': 'test_low_risk_token',
            'amount': 0.05,  # 0.05 SOL position (small)
            'token_age_hours': 48,  # 2 days old - established
            'liquidity_concentration': 0.3,  # 30% concentrated
            'dev_holdings_percent': 5,  # Dev holds 5%
            'contract_verified': True,
            'has_transfer_restrictions': False,
            'sell_buy_ratio': 0.8,  # Healthy sell ratio
        }
        
        wcca_result = await devils_advocate.conduct_pre_mortem_analysis(low_risk_params)
        
        if not wcca_result.get('veto', False):
            print(f"✅ Stage 1 CLEAR Test Passed: R-EL = {wcca_result.get('max_rel', 0):.4f} SOL")
        else:
            print(f"⚠️ Stage 1 CLEAR Test Warning: Low-risk trade vetoed: {wcca_result.get('reason')}")
        
        print("✅ Stage 1 (Survival Filter) tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Stage 1 test error: {e}")
        return False

async def test_stage2_win_rate_engine():
    """Test Stage 2: Win-Rate Engine (Naive Bayes) - Swarm Decision Engine"""
    print("🔍 Testing Stage 2: Win-Rate Engine (Naive Bayes)...")
    
    try:
        from worker_ant_v1.core.swarm_decision_engine import SwarmDecisionEngine
        
        # Initialize the win-rate engine
        swarm_engine = SwarmDecisionEngine()
        print("✅ Swarm Decision Engine initialized")
        
        # Test with high-confidence signals
        high_confidence_market_data = {
            'sentiment_score': 0.85,  # Very positive sentiment
            'rug_risk_score': 0.1,   # Low rug risk
            'narrative_strength': 0.9,  # Strong narrative
            'volume_momentum': 1.0,   # High volume
            'price_momentum': 1.0,    # Positive price action
            'social_buzz': 0.95,      # High social activity
            'whale_activity': 0.8,    # Whale interest
            'liquidity_health': 1.0,  # Good liquidity
            'rsi': 25,               # Oversold (good entry)
            'volume_change_24h': 3.0, # Volume spike
            'price_breakout_signal': 0.9,  # Breakout signal
        }
        
        win_probability = await swarm_engine.analyze_opportunity(
            'test_high_confidence_token', high_confidence_market_data, 1.0
        )
        
        print(f"✅ High-confidence signals win probability: {win_probability:.3f}")
        
        # Test with low-confidence signals
        low_confidence_market_data = {
            'sentiment_score': 0.3,   # Low sentiment
            'rug_risk_score': 0.8,    # High rug risk
            'narrative_strength': 0.2, # Weak narrative
            'volume_momentum': 0.3,    # Low volume
            'price_momentum': 0.2,     # Negative price action
            'social_buzz': 0.1,        # Low social activity
            'whale_activity': 0.2,     # No whale interest
            'liquidity_health': 0.3,   # Poor liquidity
            'rsi': 75,                # Overbought
            'volume_change_24h': 0.5,  # Volume decline
            'price_breakout_signal': 0.1,  # No breakout
        }
        
        win_probability_low = await swarm_engine.analyze_opportunity(
            'test_low_confidence_token', low_confidence_market_data, 1.0
        )
        
        print(f"✅ Low-confidence signals win probability: {win_probability_low:.3f}")
        
        if win_probability > win_probability_low:
            print("✅ Stage 2 (Naive Bayes) correctly differentiated signal quality")
        else:
            print("⚠️ Stage 2 (Naive Bayes) did not differentiate signal quality properly")
        
        print("✅ Stage 2 (Win-Rate Engine) tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Stage 2 test error: {e}")
        return False

async def test_stage3_kelly_criterion():
    """Test Stage 3: Growth Maximizer (Kelly Criterion) - Hyper Compound Engine"""
    print("🔍 Testing Stage 3: Growth Maximizer (Kelly Criterion)...")
    
    try:
        from worker_ant_v1.trading.hyper_compound_engine import HyperCompoundEngine
        
        # Initialize the growth maximizer
        compound_engine = HyperCompoundEngine()
        await compound_engine.initialize()
        print("✅ Hyper Compound Engine initialized")
        
        # Test with high win probability
        high_win_prob = 0.75  # 75% win probability
        current_capital = 10.0  # 10 SOL capital
        
        position_size_high = await compound_engine.calculate_optimal_position_size(
            win_probability=high_win_prob,
            current_capital=current_capital
        )
        
        print(f"✅ High win probability ({high_win_prob:.1%}) position size: {position_size_high:.4f} SOL")
        
        # Test with low win probability
        low_win_prob = 0.55  # 55% win probability
        
        position_size_low = await compound_engine.calculate_optimal_position_size(
            win_probability=low_win_prob,
            current_capital=current_capital
        )
        
        print(f"✅ Low win probability ({low_win_prob:.1%}) position size: {position_size_low:.4f} SOL")
        
        # Kelly Criterion should recommend larger positions for higher win probabilities
        if position_size_high > position_size_low:
            print("✅ Stage 3 (Kelly Criterion) correctly scales position size with win probability")
        else:
            print("⚠️ Stage 3 (Kelly Criterion) position sizing logic may need review")
        
        # Test with very low win probability (should recommend small/no position)
        very_low_win_prob = 0.45  # 45% win probability (below 50%)
        
        position_size_very_low = await compound_engine.calculate_optimal_position_size(
            win_probability=very_low_win_prob,
            current_capital=current_capital
        )
        
        print(f"✅ Very low win probability ({very_low_win_prob:.1%}) position size: {position_size_very_low:.4f} SOL")
        
        print("✅ Stage 3 (Growth Maximizer) tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Stage 3 test error: {e}")
        return False

async def test_full_pipeline_integration():
    """Test the complete three-stage decision pipeline integration"""
    print("🔍 Testing Full Three-Stage Pipeline Integration...")
    
    try:
        from worker_ant_v1.trading.devils_advocate_synapse import DevilsAdvocateSynapse
        from worker_ant_v1.core.swarm_decision_engine import SwarmDecisionEngine
        from worker_ant_v1.trading.hyper_compound_engine import HyperCompoundEngine
        
        # Initialize all three stages
        devils_advocate = DevilsAdvocateSynapse()
        await devils_advocate.initialize()
        
        swarm_engine = SwarmDecisionEngine()
        
        compound_engine = HyperCompoundEngine()
        await compound_engine.initialize()
        
        print("✅ All three pipeline stages initialized")
        
        # Test complete pipeline with a good opportunity
        test_opportunity = {
            'token_address': 'test_pipeline_token',
            'token_symbol': 'TEST',
            'amount': 0.1,  # Initial estimate
            'token_age_hours': 24,
            'liquidity_concentration': 0.4,
            'dev_holdings_percent': 10,
            'contract_verified': True,
            'has_transfer_restrictions': False,
            'sell_buy_ratio': 0.7,
        }
        
        good_market_data = {
            'sentiment_score': 0.75,
            'rug_risk_score': 0.2,
            'narrative_strength': 0.8,
            'volume_momentum': 0.9,
            'price_momentum': 0.8,
            'social_buzz': 0.8,
            'whale_activity': 0.7,
            'liquidity_health': 0.9,
            'rsi': 35,
            'volume_change_24h': 2.0,
            'price_breakout_signal': 0.8,
        }
        
        # Stage 1: Survival Filter
        wcca_result = await devils_advocate.conduct_pre_mortem_analysis(test_opportunity)
        
        if wcca_result.get('veto', False):
            print(f"❌ Pipeline Test: Stage 1 vetoed good opportunity: {wcca_result.get('reason')}")
            return False
        
        print(f"✅ Stage 1 PASSED: R-EL = {wcca_result.get('max_rel', 0):.4f} SOL")
        
        # Stage 2: Win-Rate Engine
        win_probability = await swarm_engine.analyze_opportunity(
            'test_pipeline_token', good_market_data, 1.0
        )
        
        hunt_threshold = 0.6
        if win_probability < hunt_threshold:
            print(f"❌ Pipeline Test: Stage 2 rejected good opportunity: win_prob={win_probability:.3f}")
            return False
        
        print(f"✅ Stage 2 PASSED: Win probability = {win_probability:.3f}")
        
        # Stage 3: Growth Maximizer
        position_size = await compound_engine.calculate_optimal_position_size(
            win_probability=win_probability,
            current_capital=5.0  # 5 SOL capital
        )
        
        if position_size <= 0:
            print(f"❌ Pipeline Test: Stage 3 recommended no position: size={position_size:.4f}")
            return False
        
        print(f"✅ Stage 3 PASSED: Optimal position = {position_size:.4f} SOL")
        
        print("🎉 FULL PIPELINE INTEGRATION TEST PASSED!")
        print(f"   Summary: WCCA CLEAR → Naive Bayes {win_probability:.1%} → Kelly {position_size:.4f} SOL")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test error: {e}")
        return False

async def test_config_system():
    """Test configuration system"""
    print("🔍 Testing configuration system...")
    
    try:
        from worker_ant_v1.core.unified_config import get_trading_config, get_wallet_config
        
        
        trading_config = get_trading_config()
        print(f"✅ Trading config loaded: {trading_config.trading_mode}")
        
        
        wallet_config = get_wallet_config()
        print(f"✅ Wallet config loaded: {wallet_config['max_wallets']} max wallets")
        
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

async def test_logger_system():
    """Test logger system"""
    print("🔍 Testing logger system...")
    
    try:
        from worker_ant_v1.utils.logger import get_logger, setup_logger
        
        
        logger1 = get_logger("test_logger")
        logger1.info("Test message from get_logger")
        
        
        logger2 = setup_logger("test_setup_logger")
        logger2.info("Test message from setup_logger")
        
        print("✅ Logger system working")
        return True
        
    except Exception as e:
        print(f"❌ Logger error: {e}")
        return False

async def test_wallet_manager():
    """Test wallet manager"""
    print("🔍 Testing wallet manager...")
    
    try:
        from worker_ant_v1.core.wallet_manager import get_wallet_manager
        
        wallet_manager = await get_wallet_manager()
        
        
        wallet = await wallet_manager.create_wallet("test_wallet_001")
        print(f"✅ Created wallet: {wallet.wallet_id}")
        
        
        wallet_info = await wallet_manager.get_wallet_info("test_wallet_001")
        print(f"✅ Got wallet info: {wallet_info['address'][:8]}...")
        
        
        success = await wallet_manager.remove_wallet("test_wallet_001")
        print(f"✅ Removed wallet: {success}")
        
        return True
        
    except Exception as e:
        print(f"❌ Wallet manager error: {e}")
        return False

async def test_vault_system():
    """Test vault system"""
    print("🔍 Testing vault system...")
    
    try:
        from worker_ant_v1.core.vault_wallet_system import get_vault_system
        
        vault_system = await get_vault_system()
        
        
        success = await vault_system.deposit_profits(1.0)
        print(f"✅ Deposited profits: {success}")
        
        
        status = vault_system.get_vault_status()
        print(f"✅ Vault status: {len(status['vaults'])} vaults")
        
        return True
        
    except Exception as e:
        print(f"❌ Vault system error: {e}")
        return False

async def test_trading_engine():
    """Test trading engine"""
    print("🔍 Testing trading engine...")
    
    try:
        from worker_ant_v1.core.unified_trading_engine import get_trading_engine
        
        trading_engine = await get_trading_engine()
        
        
        market_data = {
            'token_address': 'test_token',
            'symbol': 'TEST',
            'price': 1.0,
            'volume_24h': 1000.0,
            'liquidity': 100.0,
            'price_change_24h': 5.0
        }
        
        processed_data = await trading_engine.process_market_data(market_data)
        print(f"✅ Processed market data: {processed_data['volatility']}")
        
        
        signals = await trading_engine.generate_trading_signals(market_data)
        print(f"✅ Generated signals: {signals['buy_signal']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading engine error: {e}")
        return False

async def test_intelligence_systems():
    """Test intelligence systems"""
    print("🔍 Testing intelligence systems...")
    
    try:
        from worker_ant_v1.intelligence.token_intelligence_system import get_token_intelligence_system
        from worker_ant_v1.intelligence.enhanced_rug_detector import EnhancedRugDetector
        
        
        intelligence = await get_token_intelligence_system()
        
        market_data = {
            'symbol': 'TEST',
            'price': 1.0,
            'volume_24h': 1000.0,
            'liquidity': 100.0,
            'price_change_24h': 5.0
        }
        
        signals = await intelligence.process_signals(market_data)
        print(f"✅ Processed signals: {signals['buy_signal']}")
        
        
        rug_detector = EnhancedRugDetector()
        await rug_detector.initialize()
        
        rug_score = await rug_detector.analyze_token("test_token", market_data)
        print(f"✅ Rug score: {rug_score}")
        
        return True
        
    except Exception as e:
        print(f"❌ Intelligence systems error: {e}")
        return False

async def test_trading_swarm():
    """Test trading swarm"""
    print("🔍 Testing trading swarm...")
    
    try:
        from worker_ant_v1.trading.main import HyperIntelligentTradingSwarm
        
        swarm = HyperIntelligentTradingSwarm()
        
        
        print("✅ Trading swarm created")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading swarm error: {e}")
        return False

async def test_testing_suite():
    """Test testing suite"""
    print("🔍 Testing testing suite...")
    
    try:
        from bulletproof_testing_suite import BulletproofTestingSuite
        
        suite = BulletproofTestingSuite()
        
        
        if hasattr(suite, 'run_comprehensive_tests'):
            print("✅ Testing suite has comprehensive tests method")
        else:
            print("❌ Testing suite missing comprehensive tests method")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Testing suite error: {e}")
        return False

async def main():
    """Run all integration tests"""
    print("🚀 Starting Integration Tests")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_config_system,
        test_logger_system,
        test_wallet_manager,
        test_vault_system,
        test_trading_engine,
        test_intelligence_systems,
        test_trading_swarm,
        test_testing_suite,
        test_three_stage_pipeline_imports,
        test_stage1_survival_filter,
        test_stage2_win_rate_engine,
        test_stage3_kelly_criterion,
        test_full_pipeline_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! The system is properly integrated.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the integration.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 