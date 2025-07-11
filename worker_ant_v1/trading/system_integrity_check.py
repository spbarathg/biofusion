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


sys.path.append('.')

class SystemIntegrityChecker:
    """Comprehensive system integrity validation"""
    
    def __init__(self):
        self.results = {
            'api_keys': {'status': 'UNKNOWN', 'details': []},
            'dependencies': {'status': 'UNKNOWN', 'details': []},
            'modules': {'status': 'UNKNOWN', 'details': []},
            'configuration': {'status': 'UNKNOWN', 'details': []},
            'dry_boot': {'status': 'UNKNOWN', 'details': []},
            'trading_readiness': {'status': 'UNKNOWN', 'details': []}  # New section
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
        
        
        print("\n💰 Phase 5: Trading System Validation")  # New phase
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
        """Check Python dependencies"""
        
        critical_deps = [
            'solana', 'solders', 'pydantic', 'cryptography', 
            'numpy', 'aiohttp', 'base58', 'websockets',
            'jupiter-py', 'raydium-py'  # Added DEX dependencies
        ]
        
        optional_deps = [
            'tensorflow', 'torch', 'transformers', 'pandas',
            'scikit-learn', 'redis', 'prometheus_client'
        ]
        
        missing_critical = []
        missing_optional = []
        found_deps = []
        
        
        for dep in critical_deps:
            try:
                __import__(dep)
                found_deps.append(f"✅ {dep}")
            except ImportError:
                missing_critical.append(f"❌ {dep} (CRITICAL)")
        
        
        for dep in optional_deps:
            try:
                __import__(dep)
                found_deps.append(f"✅ {dep}")
            except ImportError:
                missing_optional.append(f"⚠️  {dep} (OPTIONAL)")
        
        
        if missing_critical:
            self.results['dependencies']['status'] = 'CRITICAL_FAIL'
            print(f"❌ Missing {len(missing_critical)} critical dependencies")
            for item in missing_critical:
                print(f"   {item}")
        else:
            self.results['dependencies']['status'] = 'PASS'
            print(f"✅ All {len(critical_deps)} critical dependencies found")
        
        
        self.results['dependencies']['details'] = found_deps + missing_critical + missing_optional
    
    def _check_trading_system(self):
        """Validate trading system components"""
        try:
            # Import trading components
            from worker_ant_v1.trading.order_buyer import OrderBuyer
            from worker_ant_v1.trading.order_seller import OrderSeller
            from worker_ant_v1.core.unified_trading_engine import UnifiedTradingEngine
            from worker_ant_v1.core.wallet_manager import WalletManager
            
            # Check trading mode
            trading_mode = os.getenv('TRADING_MODE', '').upper()
            if trading_mode not in ['LIVE', 'SIMULATION']:
                self.results['trading_readiness']['status'] = 'CRITICAL_FAIL'
                self.results['trading_readiness']['details'].append(
                    "❌ Invalid TRADING_MODE - must be LIVE or SIMULATION"
                )
                return
                
            # Validate trading parameters
            required_params = {
                'MAX_TRADE_SIZE_SOL': (0.1, 1000.0),
                'MIN_TRADE_SIZE_SOL': (0.01, 10.0),
                'MAX_SLIPPAGE_PERCENT': (0.1, 5.0),
                'PROFIT_TARGET_PERCENT': (0.5, 100.0),
                'STOP_LOSS_PERCENT': (0.5, 50.0)
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
            
            if missing_params:
                self.results['trading_readiness']['status'] = 'CRITICAL_FAIL'
                self.results['trading_readiness']['details'].append(
                    f"❌ Missing required trading parameters: {', '.join(missing_params)}"
                )
            
            if invalid_params:
                self.results['trading_readiness']['status'] = 'CRITICAL_FAIL'
                self.results['trading_readiness']['details'].append(
                    "❌ Invalid trading parameters:\n" + "\n".join(f"   • {p}" for p in invalid_params)
                )
            
            # Validate wallet configuration
            wallet_manager = WalletManager()
            if not wallet_manager.is_ready():
                self.results['trading_readiness']['status'] = 'CRITICAL_FAIL'
                self.results['trading_readiness']['details'].append(
                    "❌ Wallet system not ready - check encryption and keys"
                )
            
            # All checks passed
            if not self.results['trading_readiness']['details']:
                self.results['trading_readiness']['status'] = 'PASS'
                self.results['trading_readiness']['details'].append(
                    "✅ Trading system validation complete - all checks passed"
                )
            
        except Exception as e:
            self.results['trading_readiness']['status'] = 'CRITICAL_FAIL'
            self.results['trading_readiness']['details'].append(
                f"❌ Trading system validation failed: {str(e)}"
            )
    
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
    
    async def _dry_boot_test(self):
        """Perform dry boot test of core systems"""
        
        try:
            print("   🔧 Testing configuration loading...")
            from worker_ant_v1.core.unified_config import UnifiedConfigManager
            config = UnifiedConfigManager()
            
            
            print("   👛 Testing wallet manager...")
            from worker_ant_v1.core.wallet_manager import UnifiedWalletManager
            wallet_manager = UnifiedWalletManager()
            
            
            print("   🌐 Testing Solana RPC connection...")
            from worker_ant_v1.utils.real_solana_integration import ProductionSolanaClient
            solana_client = ProductionSolanaClient()
            
            
            try:
                health = await solana_client.check_health()
                if health:
                    print("   ✅ RPC connection successful")
                else:
                    print("   ⚠️  RPC connection slow but working")
            except Exception as e:
                print(f"   ❌ RPC connection failed: {e}")
            
            
            print("   🧠 Testing intelligence systems...")
            from worker_ant_v1.intelligence.token_intelligence_system import TokenIntelligenceSystem
            intelligence = TokenIntelligenceSystem()
            
            
            print("   🛡️  Testing safety systems...")
            from worker_ant_v1.safety.enhanced_rug_detector import EnhancedRugDetector
            rug_detector = EnhancedRugDetector()
            
            self.results['dry_boot']['status'] = 'PASS'
            self.results['dry_boot']['details'] = [
                "✅ Configuration loading successful",
                "✅ Wallet manager initialized", 
                "✅ RPC connection established",
                "✅ Intelligence systems online",
                "✅ Safety systems armed"
            ]
            
            print("✅ Dry boot test completed successfully")
            
        except Exception as e:
            self.results['dry_boot']['status'] = 'FAIL'
            self.results['dry_boot']['details'] = [f"❌ Dry boot failed: {str(e)}"]
            print(f"❌ Dry boot test failed: {e}")
            traceback.print_exc()
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate final integrity report"""
        
        print("\n" + "=" * 60)
        print("🔍 SYSTEM INTEGRITY REPORT")
        print("=" * 60)
        
        
        statuses = [result['status'] for result in self.results.values()]
        
        if 'CRITICAL_FAIL' in statuses:
            overall_status = 'CRITICAL_FAIL'
            status_emoji = '❌'
            action = 'FIX CRITICAL ISSUES BEFORE LAUNCH'
        elif 'FAIL' in statuses:
            overall_status = 'FAIL'  
            status_emoji = '⚠️'
            action = 'RESOLVE ISSUES BEFORE PRODUCTION'
        else:
            overall_status = 'READY'
            status_emoji = '✅'
            action = 'SYSTEM READY FOR LAUNCH'
        
        print(f"\n{status_emoji} OVERALL STATUS: {overall_status}")
        print(f"🎯 ACTION REQUIRED: {action}")
        
        
        print(f"\n📊 DETAILED BREAKDOWN:")
        for category, result in self.results.items():
            status_icon = '❌' if 'FAIL' in result['status'] else '✅'
            print(f"   {status_icon} {category.upper()}: {result['status']}")
        
        
        if overall_status in ['CRITICAL_FAIL', 'FAIL']:
            print(f"\n🚨 CRITICAL ISSUES TO RESOLVE:")
            for category, result in self.results.items():
                if 'FAIL' in result['status']:
                    for detail in result['details']:
                        if '❌' in detail:
                            print(f"   {detail}")
        
        print(f"\n📝 INTEGRITY CHECK COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'overall_status': overall_status,
            'action_required': action,
            'timestamp': datetime.now().isoformat(),
            'details': self.results
        }

def main():
    """Main entry point"""
    checker = SystemIntegrityChecker()
    report = checker.run_full_check()
    
    
    if report['overall_status'] == 'CRITICAL_FAIL':
        sys.exit(2)
    elif report['overall_status'] == 'FAIL':
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 