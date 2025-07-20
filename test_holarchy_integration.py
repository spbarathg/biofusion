"""
TEST HOLARCHY INTEGRATION - VERIFY ALL COMPONENTS WORK TOGETHER
=============================================================

This script tests the integration of all three phases:
1. Holarchy (Colony Commander, Swarm, Wallet hierarchy)
2. Adhocracy (Squad formation and management)
3. Blitzscaling (Aggressive growth mode)
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from entry_points.colony_commander import ColonyCommander
from worker_ant_v1.trading.squad_manager import SquadManager, SquadType
from worker_ant_v1.core.wallet_manager import UnifiedWalletManager
from worker_ant_v1.core.hyper_compound_engine import HyperCompoundEngine
from worker_ant_v1.trading.market_scanner import RealMarketScanner
from worker_ant_v1.utils.logger import get_logger


async def test_holarchy_structure():
    """Test the Holarchy structure (Colony -> Swarm -> Wallet)"""
    print("\n🏛️ Testing Holarchy Structure...")
    
    try:
        # Test Colony Commander initialization
        colony = ColonyCommander()
        print("✅ Colony Commander created")
        
        # Test swarm configurations
        print(f"📋 Swarm configs loaded: {len(colony.swarm_configs)}")
        for swarm_id, config in colony.swarm_configs.items():
            print(f"   - {swarm_id}: {config.strategy_type} ({config.initial_capital} SOL)")
        
        print("✅ Holarchy structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Holarchy test failed: {e}")
        return False


async def test_adhocracy_squads():
    """Test the Adhocracy squad formation"""
    print("\n🎯 Testing Adhocracy Squad Formation...")
    
    try:
        # Create wallet manager
        wallet_manager = UnifiedWalletManager()
        await wallet_manager.initialize()
        print("✅ Wallet manager initialized")
        
        # Create squad manager
        squad_manager = SquadManager()
        await squad_manager.initialize(wallet_manager)
        print("✅ Squad manager initialized")
        
        # Test squad formation for different opportunity types
        test_opportunities = [
            {
                'token_address': 'test_token_1',
                'token_age_hours': 0.5,  # Very new -> SNIPER squad
                'market_cap': 5000000,
                'volume_24h': 2000000,
                'is_trending': True,
                'volatility': 0.9
            },
            {
                'token_address': 'test_token_2',
                'token_age_hours': 48,  # Established -> WHALE_WATCH squad
                'market_cap': 50000000,
                'volume_24h': 10000000,
                'is_trending': False,
                'volatility': 0.3
            },
            {
                'token_address': 'test_token_3',
                'token_age_hours': 12,  # Medium age -> SCALPER squad
                'market_cap': 10000000,
                'volume_24h': 5000000,
                'is_trending': False,
                'volatility': 0.8
            }
        ]
        
        for i, opportunity in enumerate(test_opportunities):
            squad = await squad_manager.form_squad_for_opportunity(opportunity)
            if squad:
                print(f"✅ Squad {i+1}: {squad.squad_type.value} squad formed with {len(squad.wallet_ids)} wallets")
                print(f"   Target: {squad.mission_target_token}")
                print(f"   Rules: {squad.ruleset.max_position_size_sol} SOL max, {squad.ruleset.risk_level} risk")
            else:
                print(f"⚠️ Squad {i+1}: No squad formed (normal for some opportunities)")
        
        # Test squad disbanding
        if squad_manager.active_squads:
            squad_id = list(squad_manager.active_squads.keys())[0]
            await squad_manager.disband_squad(squad_id, "test_complete")
            print(f"✅ Squad {squad_id} disbanded successfully")
        
        print("✅ Adhocracy squad formation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Adhocracy test failed: {e}")
        return False


async def test_blitzscaling_mode():
    """Test the Blitzscaling mode activation"""
    print("\n🚀 Testing Blitzscaling Mode...")
    
    try:
        # Test HyperCompoundEngine blitzscaling
        compound_engine = HyperCompoundEngine()
        await compound_engine.set_blitzscaling_mode(True)
        print("✅ Compound engine blitzscaling activated")
        
        # Test compound amount calculation in blitzscaling mode
        normal_amount = await compound_engine._calculate_compound_amount(100.0)
        print(f"   Normal compound amount: {normal_amount:.2f} SOL")
        
        # Test WalletManager blitzscaling
        wallet_manager = UnifiedWalletManager()
        await wallet_manager.initialize()
        await wallet_manager.set_blitzscaling_mode(True)
        print("✅ Wallet manager blitzscaling activated")
        print(f"   Max wallets in blitzscaling: {wallet_manager.evolution_config['max_wallets']}")
        
        # Test MarketScanner blitzscaling
        market_scanner = RealMarketScanner()
        await market_scanner.initialize()
        await market_scanner.set_blitzscaling_mode(True)
        print("✅ Market scanner blitzscaling activated")
        
        # Test blitzscaling deactivation
        await compound_engine.set_blitzscaling_mode(False)
        await wallet_manager.set_blitzscaling_mode(False)
        await market_scanner.set_blitzscaling_mode(False)
        print("✅ Blitzscaling mode deactivated")
        
        print("✅ Blitzscaling mode test passed")
        return True
        
    except Exception as e:
        print(f"❌ Blitzscaling test failed: {e}")
        return False


async def test_wallet_self_assessment():
    """Test wallet self-assessment capabilities"""
    print("\n🧬 Testing Wallet Self-Assessment...")
    
    try:
        wallet_manager = UnifiedWalletManager()
        await wallet_manager.initialize()
        
        # Get a wallet for testing
        if wallet_manager.active_wallets:
            wallet_id = wallet_manager.active_wallets[0]
            wallet = wallet_manager.wallets[wallet_id]
            
            # Test personal risk assessment
            trade_params = {
                'risk_level': 'high',
                'position_size_sol': 15.0,
                'token_age_hours': 0.5,
                'is_trending': True
            }
            
            assessment = wallet.self_assess_trade(trade_params)
            print(f"✅ Wallet {wallet_id} self-assessment: {'ACCEPTED' if assessment else 'REJECTED'}")
            print(f"   Genetics: aggression={wallet.genetics.aggression:.2f}, patience={wallet.genetics.patience:.2f}")
            
            # Test squad ruleset override
            wallet.active_squad_ruleset = {
                'max_position_size_sol': 50.0,
                'risk_level': 'high'
            }
            
            squad_assessment = wallet.self_assess_trade(trade_params)
            print(f"✅ Squad ruleset assessment: {'ACCEPTED' if squad_assessment else 'REJECTED'}")
            
            # Reset squad ruleset
            wallet.active_squad_ruleset = None
            
        print("✅ Wallet self-assessment test passed")
        return True
        
    except Exception as e:
        print(f"❌ Wallet self-assessment test failed: {e}")
        return False


async def test_colony_commander_integration():
    """Test the complete Colony Commander integration"""
    print("\n🏛️ Testing Colony Commander Integration...")
    
    try:
        # Create colony commander
        colony = ColonyCommander()
        
        # Test colony status
        status = colony.get_colony_status()
        print("✅ Colony status retrieved:")
        print(f"   State: {status['state']}")
        print(f"   Swarms: {len(status['swarms'])}")
        print(f"   Blitzscaling: {status['metrics']['blitzscaling_active']}")
        
        # Test operational mode update
        colony.metrics.overall_win_rate = 0.75  # High win rate
        colony.metrics.total_trades = 100
        await colony._update_operational_mode()
        
        print(f"✅ Operational mode updated: Blitzscaling = {colony.blitzscaling_active}")
        
        print("✅ Colony Commander integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Colony Commander integration test failed: {e}")
        return False


async def main():
    """Run all integration tests"""
    print("🧪 HOLARCHIC, ADHOCRATIC, BLITZSCALING INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Holarchy Structure", test_holarchy_structure),
        ("Adhocracy Squads", test_adhocracy_squads),
        ("Blitzscaling Mode", test_blitzscaling_mode),
        ("Wallet Self-Assessment", test_wallet_self_assessment),
        ("Colony Commander Integration", test_colony_commander_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! The Holarchic, Adhocratic, Blitzscaling system is ready!")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 