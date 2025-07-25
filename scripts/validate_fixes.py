#!/usr/bin/env python3
"""
COMPREHENSIVE FIX VALIDATION SCRIPT
==================================

Validates all critical fixes implemented from the security audit.
This script verifies that:
1. Unified schemas work correctly
2. Configuration templates are aligned
3. Security vulnerabilities are fixed
4. Enhanced systems function properly
5. Dependencies are correct

Usage:
    python scripts/validate_fixes.py
"""

import asyncio
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class FixValidator:
    """Comprehensive fix validation"""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        result = {
            'test': test_name,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        
        print(f"{status} | {test_name}")
        if details and not passed:
            print(f"      {details}")
    
    async def run_all_validations(self):
        """Run all validation tests"""
        print("🔍 COMPREHENSIVE FIX VALIDATION")
        print("=" * 50)
        
        # Critical Fix Validations
        await self.validate_unified_schemas()
        await self.validate_security_fixes()
        await self.validate_config_alignment()
        await self.validate_wcca_enhancement()
        await self.validate_enhanced_systems()
        await self.validate_dependencies()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {self.passed / (self.passed + self.failed) * 100:.1f}%")
        
        if self.failed == 0:
            print("\n🎉 ALL FIXES VALIDATED SUCCESSFULLY!")
            return True
        else:
            print(f"\n⚠️ {self.failed} VALIDATION FAILURES - Review and fix")
            return False
    
    async def validate_unified_schemas(self):
        """Validate unified schema implementation"""
        print("\n🔧 Testing Unified Schema Fixes...")
        
        try:
            # Test schema imports
            from worker_ant_v1.core.schemas import TradeRecord, SystemEvent, ValidationLevel
            self.log_result("Schema Import", True, "All schemas import correctly")
            
            # Test TradeRecord creation
            trade_record = TradeRecord(
                timestamp=datetime.now(),
                trade_id="test_123",
                token_address="test_token",
                token_symbol="TEST"
            )
            self.log_result("TradeRecord Creation", True, f"Created with ID: {trade_record.trade_id}")
            
            # Test schema consistency between logger and database
            from worker_ant_v1.utils.logger import TradeRecord as LoggerTradeRecord
            from worker_ant_v1.core.database import TradeRecord as DBTradeRecord
            
            # They should be the same class now
            schema_consistent = LoggerTradeRecord is DBTradeRecord
            self.log_result("Schema Consistency", schema_consistent, 
                          "Logger and Database use same TradeRecord class" if schema_consistent 
                          else "Schema mismatch between logger and database")
            
        except Exception as e:
            self.log_result("Unified Schema", False, f"Error: {str(e)}")
    
    async def validate_security_fixes(self):
        """Validate security vulnerability fixes"""
        print("\n🔒 Testing Security Fixes...")
        
        try:
            # Check that simplified template doesn't contain private keys
            simplified_template = Path("config/simplified.env.template")
            if simplified_template.exists():
                content = simplified_template.read_text()
                has_private_key = "WALLET_PRIVATE_KEY" in content
                self.log_result("Private Key Removal", not has_private_key,
                              "Private key removed from simplified template" if not has_private_key
                              else "Private key still present in template")
                
                # Check for security notices
                has_security_notice = "Never put private keys in config files" in content
                self.log_result("Security Notice", has_security_notice,
                              "Security warning added to template")
                
                # Check for missing API keys
                required_keys = ["HELIUS_API_KEY", "SOLANA_TRACKER_API_KEY", "RAYDIUM_API_KEY"]
                missing_keys = [key for key in required_keys if key not in content]
                self.log_result("API Key Completeness", len(missing_keys) == 0,
                              f"Missing keys: {missing_keys}" if missing_keys else "All required API keys present")
            else:
                self.log_result("Template Exists", False, "Simplified template not found")
                
        except Exception as e:
            self.log_result("Security Fixes", False, f"Error: {str(e)}")
    
    async def validate_config_alignment(self):
        """Validate configuration template alignment"""
        print("\n⚙️ Testing Configuration Alignment...")
        
        try:
            # Check both templates exist
            main_template = Path("config/env.template")
            simplified_template = Path("config/simplified.env.template")
            
            both_exist = main_template.exists() and simplified_template.exists()
            self.log_result("Templates Exist", both_exist, "Both configuration templates found")
            
            if both_exist:
                main_content = main_template.read_text()
                simplified_content = simplified_template.read_text()
                
                # Check for consistency in critical fields
                critical_fields = ["BIRDEYE_API_KEY", "JUPITER_API_KEY", "HELIUS_API_KEY"]
                consistency_issues = []
                
                for field in critical_fields:
                    in_main = field in main_content
                    in_simplified = field in simplified_content
                    if in_main != in_simplified:
                        consistency_issues.append(f"{field} inconsistency")
                
                self.log_result("Template Consistency", len(consistency_issues) == 0,
                              f"Issues: {consistency_issues}" if consistency_issues 
                              else "Critical fields consistent between templates")
                
        except Exception as e:
            self.log_result("Config Alignment", False, f"Error: {str(e)}")
    
    async def validate_wcca_enhancement(self):
        """Validate WCCA R-EL calculation enhancement"""
        print("\n🛡️ Testing WCCA Enhancement...")
        
        try:
            from worker_ant_v1.trading.devils_advocate_synapse import DevilsAdvocateSynapse, FailurePattern
            
            # Create WCCA instance
            wcca = DevilsAdvocateSynapse()
            await wcca.initialize()
            
            # Test enhanced R-EL calculation
            test_params = {
                'token_address': 'test_token_123',
                'amount': 0.1,
                'token_age_hours': 24,
                'liquidity_concentration': 0.5,
                'dev_holdings_percent': 10.0,
                'contract_verified': True,
                'has_transfer_restrictions': False,
                'sell_buy_ratio': 1.0
            }
            
            result = await wcca.conduct_pre_mortem_analysis(test_params)
            
            # Check that enhanced analysis includes multiple patterns
            has_patterns = 'patterns_analyzed' in result and result['patterns_analyzed'] > 2
            self.log_result("Enhanced WCCA", has_patterns,
                          f"Analyzes {result.get('patterns_analyzed', 0)} failure patterns" if has_patterns
                          else "WCCA not analyzing multiple patterns")
            
            # Check for R-EL breakdown
            has_breakdown = 'rel_breakdown' in result
            self.log_result("R-EL Breakdown", has_breakdown,
                          "Provides detailed R-EL breakdown" if has_breakdown
                          else "No R-EL breakdown provided")
            
        except Exception as e:
            self.log_result("WCCA Enhancement", False, f"Error: {str(e)}")
    
    async def validate_enhanced_systems(self):
        """Validate enhanced system implementations"""
        print("\n🚀 Testing Enhanced Systems...")
        
        try:
            # Test enhanced Naive Bayes (structure only, not full functionality)
            from worker_ant_v1.trading.simplified_trading_bot import SimplifiedTradingBot
            
            bot = SimplifiedTradingBot()
            
            # Check enhanced signal probabilities structure
            signal_probs = bot.signal_probabilities
            enhanced_fields = ['signal_history', 'total_trades', 'successful_trades', 'last_update']
            has_enhanced_fields = all(field in signal_probs for field in enhanced_fields)
            
            self.log_result("Enhanced Naive Bayes", has_enhanced_fields,
                          "Has dynamic learning fields" if has_enhanced_fields
                          else "Missing dynamic learning structure")
            
            # Check profit processing enhancement (method exists)
            from worker_ant_v1.core.unified_trading_engine import UnifiedTradingEngine
            engine = UnifiedTradingEngine()
            
            has_fallback = hasattr(engine, '_store_profit_fallback')
            has_recovery = hasattr(engine, '_schedule_profit_recovery')
            
            self.log_result("Profit Processing Enhancement", has_fallback and has_recovery,
                          "Has fallback storage and recovery" if has_fallback and has_recovery
                          else "Missing profit protection mechanisms")
            
        except Exception as e:
            self.log_result("Enhanced Systems", False, f"Error: {str(e)}")
    
    async def validate_dependencies(self):
        """Validate dependency fixes"""
        print("\n📦 Testing Dependencies...")
        
        try:
            # Check pyproject.toml
            pyproject_file = Path("pyproject.toml")
            if pyproject_file.exists():
                content = pyproject_file.read_text()
                
                # Check that problematic SDK dependencies are removed
                problematic_deps = ["jupiter-python-sdk", "birdeye-python", "dexscreener-python"]
                has_problematic = any(dep in content for dep in problematic_deps)
                
                self.log_result("Dependency Cleanup", not has_problematic,
                              "Removed problematic SDK dependencies" if not has_problematic
                              else "Still contains problematic dependencies")
                
                # Check for essential dependencies
                essential_deps = ["aiohttp", "solana", "structlog", "asyncpg"]
                missing_essential = [dep for dep in essential_deps if dep not in content]
                
                self.log_result("Essential Dependencies", len(missing_essential) == 0,
                              f"Missing: {missing_essential}" if missing_essential
                              else "All essential dependencies present")
            else:
                self.log_result("Pyproject Exists", False, "pyproject.toml not found")
            
            # Check CI pipeline fix
            ci_file = Path(".github/workflows/ci.yml")
            if ci_file.exists():
                content = ci_file.read_text()
                
                # Check for correct validation path
                correct_path = "worker_ant_v1/safety/validate_production_secrets.py" in content
                pip_install_e = "pip install -e ." in content
                
                self.log_result("CI Pipeline Fix", correct_path and pip_install_e,
                              "CI uses correct paths and installation method" if correct_path and pip_install_e
                              else "CI pipeline still has incorrect configurations")
            else:
                self.log_result("CI File Exists", False, "CI workflow file not found")
                
        except Exception as e:
            self.log_result("Dependencies", False, f"Error: {str(e)}")
    
    def save_results(self):
        """Save validation results to file"""
        try:
            results_file = Path("logs/fix_validation_results.json")
            results_file.parent.mkdir(exist_ok=True)
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.results),
                'passed': self.passed,
                'failed': self.failed,
                'success_rate': self.passed / (self.passed + self.failed) * 100,
                'results': self.results
            }
            
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n📄 Results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


async def main():
    """Main validation function"""
    validator = FixValidator()
    
    try:
        success = await validator.run_all_validations()
        validator.save_results()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        print(traceback.format_exc())
        sys.exit(2)


# Entry point removed - execute via entry_points/run_bot.py --mode test
# This maintains the Entry Point Doctrine as defined in CONTRIBUTING.md 