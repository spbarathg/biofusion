"""
SMART APE NEURAL SWARM - SYSTEM INTEGRITY CHECK
===============================================

Comprehensive system integrity validation before launch.
Checks API keys, dependencies, modules, and runs dry boot tests.
"""

import os
import sys
import asyncio
import traceback
from typing import Dict, List, Tuple, Any
from datetime import datetime
from pathlib import Path

class SystemIntegrityChecker:
    """Comprehensive system integrity validation"""
    
    def __init__(self):
        self.results = {
            'api_keys': {'status': 'UNKNOWN', 'details': []},
            'dependencies': {'status': 'UNKNOWN', 'details': []},
            'modules': {'status': 'UNKNOWN', 'details': []},
            'configuration': {'status': 'UNKNOWN', 'details': []},
            'dry_boot': {'status': 'UNKNOWN', 'details': []},
            'trading_readiness': {'status': 'UNKNOWN', 'details': []}
        }
        
    def run_full_check(self) -> Dict[str, Any]:
        """Run complete system integrity check"""
        
        print("🔍 STARTING SYSTEM INTEGRITY CHECK")
        print("=" * 60)
        
        print("\n🔐 Phase 1: API Key Validation")
        self._check_api_keys()
        
        print("\n📦 Phase 2: Python Dependencies")
        self._check_dependencies()
        
        print("\n🧩 Phase 3: Module Loading Tests")
        self._check_modules()
        
        print("\n⚙️  Phase 4: Configuration Validation")
        self._check_configuration()
        
        print("\n💰 Phase 5: Trading System Validation")
        self._check_trading_system()
        
        print("\n🧪 Phase 6: Dry Boot Tests")
        asyncio.run(self._dry_boot_test())
        
        return self._generate_report()
    
    def _check_api_keys(self):
        """Validate API key configuration"""
        
        env_file = '.env.production'
        if not os.path.exists(env_file):
            self.results['api_keys']['status'] = 'CRITICAL_FAIL'
            self.results['api_keys']['details'].append(f"❌ Missing {env_file}")
            return
            
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if key and value and '\x00' not in value:
                        os.environ[key] = value
        
        critical_keys = {
            'HELIUS_API_KEY': 'Helius blockchain data (CRITICAL)',
            'SOLANA_TRACKER_API_KEY': 'Solana token tracking (CRITICAL)',
            'JUPITER_API_KEY': 'Jupiter DEX integration (CRITICAL)',
            'RAYDIUM_API_KEY': 'Raydium DEX integration (CRITICAL)'
        }
        
        optional_keys = {
            'QUICKNODE_RPC_URL': 'Premium RPC endpoint (OPTIONAL)',
            'DEXSCREENER_API_KEY': 'DEX data aggregation (OPTIONAL)',
            'BIRDEYE_API_KEY': 'Premium analytics (OPTIONAL)'
        }
        
        missing_critical = []
        found_keys = []
        
        for key, description in critical_keys.items():
            value = os.getenv(key, '').strip()
            if not value or value.startswith('REPLACE_WITH') or value.startswith('your_'):
                missing_critical.append(f"❌ {key}: {description}")
            else:
                masked_value = value[:8] + '...' + value[-4:] if len(value) > 12 else 'sk-...'
                found_keys.append(f"✅ {key}: {masked_value}")
        
        for key, description in optional_keys.items():
            value = os.getenv(key, '').strip()
            if value and not value.startswith('your_'):
                masked_value = value[:8] + '...' + value[-4:] if len(value) > 12 else 'sk-...'
                found_keys.append(f"✅ {key}: {masked_value}")
            else:
                found_keys.append(f"⚠️  {key}: Not configured")
        
        if missing_critical:
            self.results['api_keys']['status'] = 'CRITICAL_FAIL'
            self.results['api_keys']['details'].extend(missing_critical)
            print("\n❌ CRITICAL API KEYS MISSING:")
            for item in missing_critical:
                print(f"   {item}")
        else:
            self.results['api_keys']['status'] = 'PASS'
            print("✅ All critical API keys configured")
        
        self.results['api_keys']['details'].extend(found_keys)
        for item in found_keys:
            print(f"   {item}")
    
    def _check_dependencies(self):
        """Check Python package dependencies"""
        
        required_packages = [
            'aiohttp',
            'pydantic',
            'web3',
            'numpy',
            'pandas',
            'scikit-learn',
            'prometheus_client',
            'cryptography'
        ]
        
        missing_packages = []
        installed_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                installed_packages.append(f"✅ {package}")
            except ImportError:
                missing_packages.append(f"❌ {package}")
        
        if missing_packages:
            self.results['dependencies']['status'] = 'FAIL'
            print("❌ Missing dependencies:")
            for item in missing_packages:
                print(f"   {item}")
        else:
            self.results['dependencies']['status'] = 'PASS'
            print("✅ All dependencies installed")
        
        self.results['dependencies']['details'] = installed_packages + missing_packages
    
    def _check_modules(self):
        """Test loading of core modules"""
        
        core_modules = [
            'worker_ant_v1.core.unified_config',
            'worker_ant_v1.core.wallet_manager',
            'worker_ant_v1.core.unified_trading_engine',
            'worker_ant_v1.trading.order_buyer',
            'worker_ant_v1.trading.order_seller',
            'worker_ant_v1.intelligence.token_intelligence_system',
            'worker_ant_v1.safety.enhanced_rug_detector',
            'worker_ant_v1.utils.real_solana_integration'
        ]
        
        failed_modules = []
        loaded_modules = []
        
        for module_name in core_modules:
            try:
                __import__(module_name)
                loaded_modules.append(f"✅ {module_name}")
            except Exception as e:
                failed_modules.append(f"❌ {module_name}: {str(e)}")
        
        if failed_modules:
            self.results['modules']['status'] = 'FAIL'
            print(f"❌ {len(failed_modules)} modules failed to load")
            for item in failed_modules:
                print(f"   {item}")
        else:
            self.results['modules']['status'] = 'PASS'
            print(f"✅ All {len(core_modules)} core modules loaded successfully")
        
        self.results['modules']['details'] = loaded_modules + failed_modules
    
    def _check_configuration(self):
        """Validate configuration settings"""
        
        try:
            from worker_ant_v1.core.unified_config import UnifiedConfigManager
            
            config = UnifiedConfigManager()
            
            trading_config = config.get_trading_config()
            security_config = config.get_security_config()
            api_config = config.get_api_config()
            
            checks = []
            
            if trading_config.initial_capital_sol < 0.01:
                checks.append("❌ Initial capital too low")
            else:
                checks.append("✅ Initial capital configured")
            
            if trading_config.max_position_size_percent > 50:
                checks.append("⚠️  Position size very aggressive (>50%)")
            else:
                checks.append("✅ Position sizing reasonable")
            
            if security_config.enable_kill_switch:
                checks.append("✅ Kill switch enabled")
            else:
                checks.append("⚠️  Kill switch disabled")
            
            missing_apis = api_config.validate_required_apis()
            if missing_apis:
                checks.append(f"❌ Missing APIs: {', '.join(missing_apis)}")
            else:
                checks.append("✅ All required APIs configured")
            
            self.results['configuration']['status'] = 'PASS'
            self.results['configuration']['details'] = checks
            
            print("✅ Configuration validation passed")
            for check in checks:
                print(f"   {check}")
                
        except Exception as e:
            self.results['configuration']['status'] = 'FAIL'
            self.results['configuration']['details'] = [f"❌ Configuration error: {str(e)}"]
            print(f"❌ Configuration validation failed: {e}")
    
    def _check_trading_system(self):
        """Validate trading system readiness"""
        
        try:
            from worker_ant_v1.core.unified_trading_engine import UnifiedTradingEngine
            from worker_ant_v1.core.wallet_manager import get_wallet_manager
            
            engine = UnifiedTradingEngine()
            wallet_manager = get_wallet_manager()
            
            checks = []
            
            # Check wallet setup
            wallet_count = len(wallet_manager.get_all_wallets())
            if wallet_count < 1:
                checks.append("❌ No trading wallets configured")
            else:
                checks.append(f"✅ {wallet_count} trading wallets ready")
            
            # Check trading engine
            if engine.is_ready():
                checks.append("✅ Trading engine initialized")
            else:
                checks.append("❌ Trading engine not ready")
            
            # Check safety systems
            if engine.safety_systems_active():
                checks.append("✅ Safety systems active")
            else:
                checks.append("❌ Safety systems inactive")
            
            if any('❌' in check for check in checks):
                self.results['trading_readiness']['status'] = 'FAIL'
            else:
                self.results['trading_readiness']['status'] = 'PASS'
            
            self.results['trading_readiness']['details'] = checks
            
            for check in checks:
                print(f"   {check}")
                
        except Exception as e:
            self.results['trading_readiness']['status'] = 'FAIL'
            self.results['trading_readiness']['details'] = [f"❌ Trading system error: {str(e)}"]
            print(f"❌ Trading system validation failed: {e}")
    
    async def _dry_boot_test(self):
        """Run dry boot test of core systems"""
        
        try:
            from worker_ant_v1.core.unified_trading_engine import UnifiedTradingEngine
            
            print("Running dry boot test (no real trades)...")
            
            engine = UnifiedTradingEngine()
            await engine.initialize(dry_run=True)
            
            checks = []
            
            # Test market data fetching
            if await engine.test_market_data():
                checks.append("✅ Market data systems operational")
            else:
                checks.append("❌ Market data systems failed")
            
            # Test order simulation
            if await engine.test_order_simulation():
                checks.append("✅ Order systems operational")
            else:
                checks.append("❌ Order systems failed")
            
            # Test safety systems
            if await engine.test_safety_systems():
                checks.append("✅ Safety systems operational")
            else:
                checks.append("❌ Safety systems failed")
            
            if any('❌' in check for check in checks):
                self.results['dry_boot']['status'] = 'FAIL'
            else:
                self.results['dry_boot']['status'] = 'PASS'
            
            self.results['dry_boot']['details'] = checks
            
            for check in checks:
                print(f"   {check}")
            
            await engine.shutdown()
            
        except Exception as e:
            self.results['dry_boot']['status'] = 'FAIL'
            self.results['dry_boot']['details'] = [f"❌ Dry boot error: {str(e)}"]
            print(f"❌ Dry boot test failed: {e}")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate final integrity check report"""
        
        print("\n📊 SYSTEM INTEGRITY REPORT")
        print("=" * 60)
        
        all_passed = True
        critical_failures = []
        warnings = []
        
        for system, result in self.results.items():
            status = result['status']
            details = result['details']
            
            if status == 'CRITICAL_FAIL':
                all_passed = False
                critical_failures.extend(details)
            elif status == 'FAIL':
                all_passed = False
                warnings.extend(details)
        
        if critical_failures:
            print("\n🚨 CRITICAL FAILURES - SYSTEM CANNOT START:")
            for failure in critical_failures:
                print(f"   {failure}")
        
        if warnings:
            print("\n⚠️  WARNINGS - Review Before Launch:")
            for warning in warnings:
                print(f"   {warning}")
        
        if all_passed:
            print("\n✅ ALL SYSTEMS READY FOR LAUNCH")
        else:
            print("\n❌ SYSTEM NOT READY - Fix errors before launch")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'all_systems_ready': all_passed,
            'results': self.results,
            'critical_failures': critical_failures,
            'warnings': warnings
        }

def main():
    """Run complete system integrity check"""
    checker = SystemIntegrityChecker()
    results = checker.run_full_check()
    return 0 if results['all_systems_ready'] else 1

if __name__ == "__main__":
    sys.exit(main()) 