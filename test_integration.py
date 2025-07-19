#!/usr/bin/env python3
"""
Integration Test Script
======================

Test script to verify that all systems are properly integrated.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

async def test_basic_imports():
    """Test that all core modules can be imported"""
    print("🔍 Testing basic imports...")
    
    try:
        # Test core systems
        from worker_ant_v1.core.unified_trading_engine import get_trading_engine
        from worker_ant_v1.core.wallet_manager import get_wallet_manager
        from worker_ant_v1.core.vault_wallet_system import get_vault_system
        from worker_ant_v1.core.unified_config import get_trading_config, get_wallet_config
        
        # Test intelligence systems
        from worker_ant_v1.intelligence.sentiment_first_ai import get_sentiment_first_ai
        from worker_ant_v1.intelligence.token_intelligence_system import get_token_intelligence_system
        from worker_ant_v1.intelligence.enhanced_rug_detector import EnhancedRugDetector
        
        # Test trading systems
        from worker_ant_v1.trading.market_scanner import get_market_scanner
        from worker_ant_v1.trading.surgical_trade_executor import SurgicalTradeExecutor
        from worker_ant_v1.trading.main import HyperIntelligentTradingSwarm, MemecoinTradingBot
        
        # Test safety systems
        from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch
        
        # Test utils
        from worker_ant_v1.utils.logger import get_logger, setup_logger
        
        print("✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

async def test_config_system():
    """Test configuration system"""
    print("🔍 Testing configuration system...")
    
    try:
        from worker_ant_v1.core.unified_config import get_trading_config, get_wallet_config
        
        # Test trading config
        trading_config = get_trading_config()
        print(f"✅ Trading config loaded: {trading_config.trading_mode}")
        
        # Test wallet config
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
        
        # Test get_logger
        logger1 = get_logger("test_logger")
        logger1.info("Test message from get_logger")
        
        # Test setup_logger
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
        
        # Test wallet creation
        wallet = await wallet_manager.create_wallet("test_wallet_001")
        print(f"✅ Created wallet: {wallet.wallet_id}")
        
        # Test wallet info
        wallet_info = await wallet_manager.get_wallet_info("test_wallet_001")
        print(f"✅ Got wallet info: {wallet_info['address'][:8]}...")
        
        # Test wallet removal
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
        
        # Test profit deposit
        success = await vault_system.deposit_profits(1.0)
        print(f"✅ Deposited profits: {success}")
        
        # Test vault status
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
        
        # Test market data processing
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
        
        # Test signal generation
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
        
        # Test token intelligence
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
        
        # Test rug detector
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
        
        # Test initialization (without actually starting)
        print("✅ Trading swarm created")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading swarm error: {e}")
        return False

async def test_testing_suite():
    """Test testing suite"""
    print("🔍 Testing testing suite...")
    
    try:
        from worker_ant_v1.trading.bulletproof_testing_suite import BulletproofTestingSuite
        
        suite = BulletproofTestingSuite()
        
        # Test that the comprehensive tests method exists
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
        test_testing_suite
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