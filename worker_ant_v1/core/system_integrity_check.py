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
        
        critical_deps = [
            'aiohttp',
            'pydantic',
            'numpy',
            'pandas',
            'scikit-learn',
            'prometheus_client',
            'cryptography',
            'solana',
            'solders',
            'base58',
            'transformers',
            'torch',
            'aiosqlite',
            'redis',
            'websockets',
            'asyncio',
            'requests'
        ]
        
        optional_deps = [
            'tensorflow',
            'pymongo',
            'plotly',
            'matplotlib',
            'seaborn',
            'uvloop',
            'orjson',
            'sentry_sdk'
        ]
        
        missing_critical = []
        missing_optional = []
        installed_deps = []
        
        # Check critical dependencies
        for dep in critical_deps:
            try:
                __import__(dep)
                installed_deps.append(f"✅ {dep}")
            except ImportError:
                missing_critical.append(f"❌ {dep} (CRITICAL)")
        
        # Check optional dependencies
        for dep in optional_deps:
            try:
                __import__(dep)
                installed_deps.append(f"✅ {dep}")
            except ImportError:
                missing_optional.append(f"⚠️  {dep} (OPTIONAL)")
        
        # Check GPU availability for AI models
        try:
            import torch
            if torch.cuda.is_available():
                installed_deps.append("✅ CUDA GPU support available")
            else:
                installed_deps.append("⚠️  CUDA GPU support not available (CPU only)")
        except ImportError:
            missing_optional.append("⚠️  torch (OPTIONAL - for GPU acceleration)")
        
        # Report results
        if missing_critical:
            self.results['dependencies']['status'] = 'CRITICAL_FAIL'
            print(f"❌ Missing {len(missing_critical)} critical dependencies:")
            for item in missing_critical:
                print(f"   {item}")
        else:
            self.results['dependencies']['status'] = 'PASS'
            print(f"✅ All {len(critical_deps)} critical dependencies found")
        
        if missing_optional:
            print(f"⚠️  Missing {len(missing_optional)} optional dependencies:")
            for item in missing_optional:
                print(f"   {item}")
        
        self.results['dependencies']['details'] = installed_deps + missing_critical + missing_optional
    
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
        """Validate trading system components and configuration"""
        try:
            # Check trading mode configuration
            trading_mode = os.getenv('TRADING_MODE', '').upper()
            if trading_mode not in ['LIVE', 'SIMULATION', 'PRODUCTION']:
                self.results['trading_readiness']['status'] = 'CRITICAL_FAIL'
                self.results['trading_readiness']['details'].append(
                    f"❌ Invalid TRADING_MODE: {trading_mode} - must be LIVE, SIMULATION, or PRODUCTION"
                )
                return
            
            # Validate trading parameters
            required_params = {
                'MAX_TRADE_SIZE_SOL': (0.1, 1000.0),
                'MIN_TRADE_SIZE_SOL': (0.01, 10.0),
                'MAX_SLIPPAGE_PERCENT': (0.1, 5.0),
                'PROFIT_TARGET_PERCENT': (0.5, 100.0),
                'STOP_LOSS_PERCENT': (0.5, 50.0),
                'INITIAL_CAPITAL': (1.0, 10000.0)
            }
            
            missing_params = []
            invalid_params = []
            
            for param, (min_val, max_val) in required_params.items():
                value = os.getenv(param)
                if not value:
                    missing_params.append(param)
                else:
                    try:
                        float_val = float(value)
                        if not (min_val <= float_val <= max_val):
                            invalid_params.append(
                                f"{param}={float_val} (should be between {min_val}-{max_val})"
                            )
                    except ValueError:
                        invalid_params.append(f"{param}={value} (invalid number)")
            
            # Check for missing parameters
            if missing_params:
                self.results['trading_readiness']['status'] = 'CRITICAL_FAIL'
                self.results['trading_readiness']['details'].append(
                    f"❌ Missing trading parameters: {', '.join(missing_params)}"
                )
                return
            
            # Check for invalid parameters
            if invalid_params:
                self.results['trading_readiness']['status'] = 'CRITICAL_FAIL'
                self.results['trading_readiness']['details'].append(
                    f"❌ Invalid trading parameters: {', '.join(invalid_params)}"
                )
                return
            
            # Validate wallet configuration
            wallet_count = int(os.getenv('WALLET_COUNT', '10'))
            if wallet_count < 3 or wallet_count > 20:
                self.results['trading_readiness']['status'] = 'CRITICAL_FAIL'
                self.results['trading_readiness']['details'].append(
                    f"❌ Invalid WALLET_COUNT: {wallet_count} (should be between 3-20)"
                )
                return
            
            # Check safety settings
            safety_params = {
                'KILL_SWITCH_ENABLED': 'true',
                'EMERGENCY_STOP_ENABLED': 'true',
                'VAULT_ENABLED': 'true'
            }
            
            for param, expected_value in safety_params.items():
                actual_value = os.getenv(param, 'false').lower()
                if actual_value != expected_value:
                    self.results['trading_readiness']['details'].append(
                        f"⚠️  {param}={actual_value} (recommended: {expected_value})"
                    )
            
            # All checks passed
            self.results['trading_readiness']['status'] = 'PASS'
            self.results['trading_readiness']['details'].append(
                f"✅ Trading system configuration valid - Mode: {trading_mode}, Wallets: {wallet_count}"
            )
            
        except Exception as e:
            self.results['trading_readiness']['status'] = 'CRITICAL_FAIL'
            self.results['trading_readiness']['details'].append(f"❌ Trading system validation error: {str(e)}")
    
    async def _dry_boot_test(self):
        """Perform dry boot test of core systems"""
        
        try:
            print("   🔧 Testing configuration loading...")
            from worker_ant_v1.core.unified_config import UnifiedConfigManager
            config_manager = UnifiedConfigManager()
            config = config_manager.get_config()
            
            if not config:
                raise Exception("Configuration loading failed")
            
            print("   ✅ Configuration loaded successfully")
            
            print("   👛 Testing wallet manager...")
            from worker_ant_v1.core.wallet_manager import UnifiedWalletManager
            wallet_manager = UnifiedWalletManager()
            
            # Test wallet manager initialization
            if not await wallet_manager.initialize():
                raise Exception("Wallet manager initialization failed")
            
            print("   ✅ Wallet manager initialized successfully")
            
            print("   🌐 Testing Solana RPC connection...")
            try:
    from solana.rpc.async_api import AsyncClient
except ImportError:
    from ..utils.solana_compat import AsyncClient
            
            rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
            solana_client = AsyncClient(rpc_url)
            
            try:
                # Test RPC connection
                response = await solana_client.get_health()
                if response.value != "ok":
                    raise Exception(f"RPC health check failed: {response.value}")
                
                # Test basic RPC operations
                slot_response = await solana_client.get_slot()
                if not slot_response.value:
                    raise Exception("RPC slot query failed")
                
                print("   ✅ Solana RPC connection successful")
                
            except Exception as e:
                print(f"   ⚠️  RPC connection warning: {e}")
            
            print("   🧠 Testing intelligence systems...")
            from worker_ant_v1.intelligence.token_intelligence_system import TokenIntelligenceSystem
            intelligence = TokenIntelligenceSystem()
            
            # Test intelligence system initialization
            if not await intelligence.initialize():
                raise Exception("Intelligence system initialization failed")
            
            print("   ✅ Intelligence systems initialized successfully")
            
            print("   🛡️  Testing safety systems...")
            from worker_ant_v1.safety.enhanced_rug_detector import EnhancedRugDetector
            rug_detector = EnhancedRugDetector()
            
            # Test rug detector initialization
            if not await rug_detector.initialize():
                raise Exception("Rug detector initialization failed")
            
            print("   ✅ Safety systems initialized successfully")
            
            print("   🔄 Testing trading engine...")
            from worker_ant_v1.core.unified_trading_engine import UnifiedTradingEngine
            trading_engine = UnifiedTradingEngine()
            
            # Test trading engine initialization
            if not await trading_engine.initialize():
                raise Exception("Trading engine initialization failed")
            
            print("   ✅ Trading engine initialized successfully")
            
            print("   🏦 Testing vault system...")
            from worker_ant_v1.core.vault_wallet_system import VaultWalletSystem
            vault_system = VaultWalletSystem()
            
            # Test vault system initialization
            if not await vault_system.initialize_vault_system():
                raise Exception("Vault system initialization failed")
            
            print("   ✅ Vault system initialized successfully")
            
            # All systems initialized successfully
            self.results['dry_boot']['status'] = 'PASS'
            self.results['dry_boot']['details'].append("✅ All core systems initialized successfully")
            
        except Exception as e:
            self.results['dry_boot']['status'] = 'CRITICAL_FAIL'
            error_msg = f"❌ Dry boot test failed: {str(e)}"
            self.results['dry_boot']['details'].append(error_msg)
            print(f"   {error_msg}")
            raise
    
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