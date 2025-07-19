#!/usr/bin/env python3
"""
Simple Integration Test
======================

Basic test to check if imports work correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

def test_basic_imports():
    """Test that all core modules can be imported"""
    print("🔍 Testing basic imports...")
    
    try:
        # Test core systems
        from worker_ant_v1.core.unified_config import get_trading_config
        print("✅ Unified config imported")
        
        from worker_ant_v1.utils.logger import get_logger
        print("✅ Logger imported")
        
        from worker_ant_v1.intelligence.sentiment_analyzer import SentimentData
        print("✅ SentimentData imported")
        
        from worker_ant_v1.utils.solana_compat import Keypair, AsyncClient
        print("✅ Solana compatibility layer imported")
        
        print("✅ All basic imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_system():
    """Test configuration system"""
    print("🔍 Testing configuration system...")
    
    try:
        from worker_ant_v1.core.unified_config import get_trading_config
        
        # Test trading config
        trading_config = get_trading_config()
        print(f"✅ Trading config loaded: {trading_config.trading_mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting simple integration test...")
    
    tests = [
        test_basic_imports,
        test_config_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Integration is working.")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 