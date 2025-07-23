"""
UNIFIED SYSTEM VALIDATOR
========================

Comprehensive system validation that consolidates configuration validation,
system integrity checking, and dependency validation into a single system.
"""

import os
import sys
import asyncio
import traceback
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path
import importlib
import aiohttp

from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.utils.constants import (
    ValidationLevel, TradingMode, SecurityLevel, APIEndpoints, TokenMints,
    DefaultValues, ErrorMessages, SuccessMessages, ValidationRules
)


# ValidationLevel enum is now imported from constants


class ValidationResult:
    """Result of a validation check"""
    
    def __init__(self):
        self.issues: List[Dict[str, Any]] = []
        self.passed = True
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
    
    def add_issue(self, level: ValidationLevel, message: str, details: Optional[str] = None):
        """Add a validation issue"""
        issue = {
            "level": level.value,
            "message": message,
            "details": details,
            "timestamp": datetime.now(),
        }
        self.issues.append(issue)
        
        if level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR]:
            self.passed = False
    
    def add_critical(self, message: str, details: Optional[str] = None):
        """Add a critical issue"""
        self.add_issue(ValidationLevel.CRITICAL, message, details)
    
    def add_error(self, message: str, details: Optional[str] = None):
        """Add an error issue"""
        self.add_issue(ValidationLevel.ERROR, message, details)
    
    def add_warning(self, message: str, details: Optional[str] = None):
        """Add a warning issue"""
        self.add_issue(ValidationLevel.WARNING, message, details)
    
    def add_info(self, message: str, details: Optional[str] = None):
        """Add an info message"""
        self.add_issue(ValidationLevel.INFO, message, details)
    
    def finalize(self):
        """Finalize the validation result"""
        self.end_time = datetime.now()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        critical_count = len([i for i in self.issues if i["level"] == ValidationLevel.CRITICAL.value])
        error_count = len([i for i in self.issues if i["level"] == ValidationLevel.ERROR.value])
        warning_count = len([i for i in self.issues if i["level"] == ValidationLevel.WARNING.value])
        info_count = len([i for i in self.issues if i["level"] == ValidationLevel.INFO.value])
        
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        return {
            "passed": self.passed,
            "duration_seconds": duration,
            "total_issues": len(self.issues),
            "critical_issues": critical_count,
            "error_issues": error_count,
            "warning_issues": warning_count,
            "info_messages": info_count,
            "issues": self.issues,
        }


class UnifiedSystemValidator:
    """Unified system validator for comprehensive validation"""
    
    def __init__(self):
        self.logger = get_logger("SystemValidator")
        self.result = ValidationResult()
        
        # Validation phases
        self.phases = [
            "environment",
            "dependencies",
            "configuration",
            "api_keys",
            "modules",
            "trading_system",
            "network_connectivity",
            "dry_boot",
        ]
        
        self.logger.info("🔍 Unified System Validator initialized")
    
    async def run_full_validation(self) -> ValidationResult:
        """Run complete system validation"""
        self.logger.info("🚀 Starting comprehensive system validation")
        self.result = ValidationResult()
        
        try:
            # Phase 1: Environment validation
            await self._validate_environment()
            
            # Phase 2: Dependencies validation
            await self._validate_dependencies()
            
            # Phase 3: Configuration validation
            await self._validate_configuration()
            
            # Phase 4: API keys validation
            await self._validate_api_keys()
            
            # Phase 5: Module loading validation
            await self._validate_modules()
            
            # Phase 6: Trading system validation
            await self._validate_trading_system()
            
            # Phase 7: Network connectivity validation
            await self._validate_network_connectivity()
            
            # Phase 8: Dry boot validation
            await self._validate_dry_boot()
            
        except Exception as e:
            self.result.add_critical(f"Validation process failed: {str(e)}", traceback.format_exc())
        
        finally:
            self.result.finalize()
            self._log_validation_summary()
        
        return self.result
    
    async def _validate_environment(self):
        """Validate environment setup"""
        self.logger.info("🔧 Validating environment setup...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 9):
            self.result.add_critical(
                f"Python version {python_version.major}.{python_version.minor} is not supported. "
                "Python 3.9 or higher is required."
            )
        else:
            self.result.add_info(f"Python version {python_version.major}.{python_version.minor}.{python_version.micro} is supported")
        
        # Check working directory
        cwd = Path.cwd()
        if "antbot" not in cwd.name.lower():
            self.result.add_warning(
                f"Working directory '{cwd.name}' doesn't appear to be the antbot project directory"
            )
        
        # Check environment file
        env_file = Path(".env.production")
        if not env_file.exists():
            self.result.add_critical("Missing .env.production file")
        else:
            self.result.add_info("Environment file found")
        
        # Check logs directory
        logs_dir = Path("logs")
        if not logs_dir.exists():
            logs_dir.mkdir(exist_ok=True)
            self.result.add_info("Created logs directory")
        else:
            self.result.add_info("Logs directory exists")
    
    async def _validate_dependencies(self):
        """Validate Python dependencies"""
        self.logger.info("📦 Validating dependencies...")
        
        # Critical dependencies
        critical_deps = {
            "aiohttp": "Async HTTP client",
            "numpy": "Numerical computing",
            "pandas": "Data analysis",
            "solana": "Solana blockchain integration",
            "solders": "Solana data structures",
            "base58": "Base58 encoding",
            "cryptography": "Encryption and security",
            "aiosqlite": "Async SQLite database",
            "structlog": "Structured logging",
        }
        
        # Optional dependencies
        optional_deps = {
            "torch": "PyTorch for AI models",
            "transformers": "Hugging Face transformers",
            "scikit-learn": "Machine learning",
            "redis": "Redis caching",
            "prometheus_client": "Metrics and monitoring",
            "uvloop": "Fast event loop",
            "orjson": "Fast JSON parsing",
        }
        
        # Check critical dependencies
        missing_critical = []
        for dep, description in critical_deps.items():
            try:
                importlib.import_module(dep)
                self.result.add_info(f"✅ {dep}: {description}")
            except ImportError:
                missing_critical.append(dep)
                self.result.add_critical(f"❌ {dep}: {description} - MISSING")
        
        if missing_critical:
            self.result.add_critical(
                f"Missing {len(missing_critical)} critical dependencies: {', '.join(missing_critical)}"
            )
        
        # Check optional dependencies
        for dep, description in optional_deps.items():
            try:
                importlib.import_module(dep)
                self.result.add_info(f"✅ {dep}: {description}")
            except ImportError:
                self.result.add_warning(f"⚠️ {dep}: {description} - OPTIONAL")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                self.result.add_info("✅ CUDA GPU support available")
            else:
                self.result.add_info("ℹ️ CUDA GPU support not available (CPU only)")
        except ImportError:
            self.result.add_warning("⚠️ PyTorch not available - GPU acceleration disabled")
    
    async def _validate_configuration(self):
        """Validate configuration settings"""
        self.logger.info("⚙️ Validating configuration...")
        
        # Load environment variables
        env_file = Path(".env.production")
        if env_file.exists():
            with open(env_file, "r") as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        if key and value:
                            os.environ[key] = value
        
        # Validate trading mode
        trading_mode = os.getenv("TRADING_MODE", "").upper()
        valid_modes = [mode.value.upper() for mode in TradingMode]
        if trading_mode not in valid_modes:
            self.result.add_critical(
                f"Invalid TRADING_MODE: {trading_mode}. Must be one of: {', '.join(valid_modes)}"
            )
        else:
            self.result.add_info(f"Trading mode: {trading_mode}")
        
        # Validate numeric parameters using validation rules
        trading_rules = ValidationRules.get_trading_rules()
        safety_rules = ValidationRules.get_safety_rules()
        
        # Combine all validation rules
        all_rules = {**trading_rules, **safety_rules}
        all_rules.update({
            "WALLET_COUNT": (3, 20, "Number of trading wallets"),
        })
        
        for param, (min_val, max_val) in all_rules.items():
            description = param.replace("_", " ").title()
            value_str = os.getenv(param)
            if value_str:
                try:
                    value = float(value_str)
                    if not (min_val <= value <= max_val):
                        self.result.add_warning(
                            f"{param}={value} is outside recommended range ({min_val}-{max_val}): {description}"
                        )
                    else:
                        self.result.add_info(f"✅ {param}={value}: {description}")
                except ValueError:
                    self.result.add_error(f"Invalid numeric value for {param}: {value_str}")
            else:
                self.result.add_warning(f"Missing configuration parameter: {param}")
        
        # Validate safety settings
        safety_params = {
            "KILL_SWITCH_ENABLED": "true",
            "EMERGENCY_STOP_ENABLED": "true",
            "VAULT_ENABLED": "true",
        }
        
        for param, recommended_value in safety_params.items():
            actual_value = os.getenv(param, "false").lower()
            if actual_value != recommended_value:
                self.result.add_warning(
                    f"{param}={actual_value} (recommended: {recommended_value})"
                )
            else:
                self.result.add_info(f"✅ {param}={actual_value}")
    
    async def _validate_api_keys(self):
        """Validate API key configuration"""
        self.logger.info("🔑 Validating API keys...")
        
        # Critical API keys
        critical_keys = {
            "HELIUS_API_KEY": "Helius blockchain data",
            "SOLANA_TRACKER_API_KEY": "Solana token tracking",
            "JUPITER_API_KEY": "Jupiter DEX integration",
        }
        
        # Optional API keys
        optional_keys = {
            "BIRDEYE_API_KEY": "Birdeye analytics",
            "DEXSCREENER_API_KEY": "DEX data aggregation",
            "QUICKNODE_RPC_URL": "Premium RPC endpoint",
        }
        
        # Check critical keys
        missing_critical = []
        for key, description in critical_keys.items():
            value = os.getenv(key, "").strip()
            if not value or value.startswith("REPLACE_WITH") or value.startswith("your_"):
                missing_critical.append(key)
                self.result.add_critical(f"❌ {key}: {description} - MISSING OR INVALID")
            else:
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                self.result.add_info(f"✅ {key}: {masked_value} ({description})")
        
        if missing_critical:
            self.result.add_critical(
                f"Missing {len(missing_critical)} critical API keys: {', '.join(missing_critical)}"
            )
        
        # Check optional keys
        for key, description in optional_keys.items():
            value = os.getenv(key, "").strip()
            if value and not value.startswith("REPLACE_WITH"):
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                self.result.add_info(f"✅ {key}: {masked_value} ({description})")
            else:
                self.result.add_warning(f"⚠️ {key}: {description} - OPTIONAL")
    
    async def _validate_modules(self):
        """Validate core module loading"""
        self.logger.info("🧩 Validating module loading...")
        
        core_modules = [
            "worker_ant_v1.core.unified_config",
            "worker_ant_v1.core.wallet_manager",
            "worker_ant_v1.core.unified_trading_engine",
            "worker_ant_v1.trading.order_buyer",
            "worker_ant_v1.trading.order_seller",
            "worker_ant_v1.intelligence.token_intelligence_system",
            "worker_ant_v1.safety.enhanced_rug_detector",
            "worker_ant_v1.utils.market_data_fetcher",
        ]
        
        failed_modules = []
        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
                self.result.add_info(f"✅ {module_name}")
            except Exception as e:
                failed_modules.append(module_name)
                self.result.add_error(f"❌ {module_name}: {str(e)}")
        
        if failed_modules:
            self.result.add_error(f"Failed to load {len(failed_modules)} modules")
        else:
            self.result.add_info(f"All {len(core_modules)} core modules loaded successfully")
    
    async def _validate_trading_system(self):
        """Validate trading system configuration"""
        self.logger.info("💰 Validating trading system...")
        
        try:
            # Import trading components
            from worker_ant_v1.core.unified_config import UnifiedConfigManager
            from worker_ant_v1.core.wallet_manager import UnifiedWalletManager
            
            # Test configuration loading
            config_manager = UnifiedConfigManager()
            config = config_manager.get_config()
            if config:
                self.result.add_info("✅ Configuration manager loaded successfully")
            else:
                self.result.add_error("❌ Configuration manager failed to load")
            
            # Test wallet manager
            wallet_manager = UnifiedWalletManager()
            if await wallet_manager.initialize():
                self.result.add_info("✅ Wallet manager initialized successfully")
            else:
                self.result.add_error("❌ Wallet manager initialization failed")
            
            # Validate wallet configuration
            wallet_count = int(os.getenv("WALLET_COUNT", "10"))
            if wallet_count < 3 or wallet_count > 20:
                self.result.add_warning(
                    f"Wallet count {wallet_count} is outside recommended range (3-20)"
                )
            else:
                self.result.add_info(f"✅ Wallet count: {wallet_count}")
            
        except Exception as e:
            self.result.add_error(f"Trading system validation failed: {str(e)}")
    
    async def _validate_network_connectivity(self):
        """Validate network connectivity to external services"""
        self.logger.info("🌐 Validating network connectivity...")
        
        connectivity_tests = [
            ("Jupiter API", f"{APIEndpoints.JUPITER_PRICE_URL}?ids=SOL"),
            ("Birdeye API", f"{APIEndpoints.BIRDEYE_BASE_URL}/public/price?address={TokenMints.SOL}"),
            ("DexScreener API", f"{APIEndpoints.DEXSCREENER_TOKEN_URL}/{TokenMints.SOL}"),
        ]
        
        async with aiohttp.ClientSession() as session:
            for service_name, url in connectivity_tests:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            self.result.add_info(f"✅ {service_name}: Connected")
                        else:
                            self.result.add_warning(f"⚠️ {service_name}: HTTP {response.status}")
                except asyncio.TimeoutError:
                    self.result.add_warning(f"⚠️ {service_name}: Timeout")
                except Exception as e:
                    self.result.add_warning(f"⚠️ {service_name}: {str(e)}")
    
    async def _validate_dry_boot(self):
        """Perform dry boot validation of core systems"""
        self.logger.info("🧪 Performing dry boot validation...")
        
        try:
            # Test configuration loading
            from worker_ant_v1.core.unified_config import UnifiedConfigManager
            config_manager = UnifiedConfigManager()
            config = config_manager.get_config()
            if config:
                self.result.add_info("✅ Configuration loaded successfully")
            else:
                self.result.add_error("❌ Configuration loading failed")
            
            # Test wallet manager initialization
            from worker_ant_v1.core.wallet_manager import UnifiedWalletManager
            wallet_manager = UnifiedWalletManager()
            if await wallet_manager.initialize():
                self.result.add_info("✅ Wallet manager initialized successfully")
            else:
                self.result.add_error("❌ Wallet manager initialization failed")
            
            # Test market data fetcher
            from worker_ant_v1.utils.market_data_fetcher import get_market_data_fetcher
            market_fetcher = await get_market_data_fetcher()
            if market_fetcher:
                self.result.add_info("✅ Market data fetcher initialized successfully")
            else:
                self.result.add_error("❌ Market data fetcher initialization failed")
            
            # Test Solana RPC connection
            try:
                from solana.rpc.async_api import AsyncClient
                client = AsyncClient(APIEndpoints.SOLANA_MAINNET_RPC)
                response = await client.get_health()
                if response.value == "ok":
                    self.result.add_info("✅ Solana RPC connection successful")
                else:
                    self.result.add_warning("⚠️ Solana RPC connection slow")
                await client.close()
            except Exception as e:
                self.result.add_warning(f"⚠️ Solana RPC connection failed: {str(e)}")
            
            # Test intelligence systems
            try:
                from worker_ant_v1.intelligence.token_intelligence_system import TokenIntelligenceSystem
                intelligence = TokenIntelligenceSystem()
                self.result.add_info("✅ Intelligence system loaded successfully")
            except Exception as e:
                self.result.add_warning(f"⚠️ Intelligence system loading failed: {str(e)}")
            
            # Test safety systems
            try:
                from worker_ant_v1.intelligence.enhanced_rug_detector import EnhancedRugDetector
                rug_detector = EnhancedRugDetector()
                self.result.add_info("✅ Safety systems loaded successfully")
            except Exception as e:
                self.result.add_warning(f"⚠️ Safety systems loading failed: {str(e)}")
            
        except Exception as e:
            self.result.add_error(f"Dry boot validation failed: {str(e)}")
    
    def _log_validation_summary(self):
        """Log validation summary"""
        summary = self.result.get_summary()
        
        self.logger.info("📊 Validation Summary:")
        self.logger.info(f"   Status: {'✅ PASSED' if summary['passed'] else '❌ FAILED'}")
        self.logger.info(f"   Duration: {summary['duration_seconds']:.2f} seconds")
        self.logger.info(f"   Total Issues: {summary['total_issues']}")
        self.logger.info(f"   Critical: {summary['critical_issues']}")
        self.logger.info(f"   Errors: {summary['error_issues']}")
        self.logger.info(f"   Warnings: {summary['warning_issues']}")
        self.logger.info(f"   Info: {summary['info_messages']}")
        
        if not summary['passed']:
            self.logger.error("❌ System validation failed - please fix critical issues before proceeding")
        else:
            self.logger.info("✅ System validation passed - ready for operation")


# Global validator instance
_system_validator: Optional[UnifiedSystemValidator] = None


async def get_system_validator() -> UnifiedSystemValidator:
    """Get or create global system validator instance"""
    global _system_validator
    
    if _system_validator is None:
        _system_validator = UnifiedSystemValidator()
    
    return _system_validator


async def validate_production_config() -> bool:
    """Validate production configuration (backward compatibility)"""
    validator = await get_system_validator()
    result = await validator.run_full_validation()
    return result.passed


def validate_production_config_sync() -> bool:
    """Synchronous version for backward compatibility"""
    try:
        return asyncio.run(validate_production_config())
    except Exception:
        return False 