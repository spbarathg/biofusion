"""
PRODUCTION CONFIGURATION VALIDATOR
=================================

Comprehensive validation system for production deployment configuration.
Ensures all required environment variables are set with valid values.
"""

import os
import re
import logging
import base58
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from urllib.parse import urlparse

class ValidationSeverity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"

class ValidationResult:
    """Configuration validation result"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.is_valid = True
        
    def add_error(self, message: str):
        self.errors.append(message)
        self.is_valid = False
        
    def add_warning(self, message: str):
        self.warnings.append(message)
        
    def add_info(self, message: str):
        self.info.append(message)

class ProductionConfigValidator:
    """Comprehensive production configuration validator"""
    
    def __init__(self):
        self.logger = logging.getLogger("ConfigValidator")
        
        # Required environment variables for production
        self.required_vars = {
            'TRADING_MODE': self._validate_trading_mode,
            'WALLET_ENCRYPTION_PASSWORD': self._validate_wallet_password,
            'SOLANA_RPC_URL': self._validate_rpc_url,
        }
        
        # Optional but recommended variables - SOCIAL MEDIA REMOVED
        self.recommended_vars = {
            'PRIVATE_RPC_URL': self._validate_rpc_url,
        }
        
        # Security validation rules
        self.security_rules = {
            'WALLET_ENCRYPTION_PASSWORD': self._validate_password_strength,
            'WALLET_PASSWORD': self._validate_password_strength,
        }
        
    def validate_production_config(self) -> ValidationResult:
        """Validate production configuration comprehensively"""
        
        result = ValidationResult()
        
        self.logger.info("🔍 Starting production configuration validation...")
        
        # 1. Validate required variables
        self._validate_required_variables(result)
        
        # 2. Validate security settings
        self._validate_security_settings(result)
        
        # 3. Validate network settings
        self._validate_network_settings(result)
        
        # 4. Validate trading parameters
        self._validate_trading_parameters(result)
        
        # 5. Validate alert system
        self._validate_alert_system(result)
        
        # 6. Validate wallet configuration
        self._validate_wallet_configuration(result)
        
        # 7. Production-specific checks
        self._validate_production_specific(result)
        
        # Log results
        self._log_validation_results(result)
        
        return result
    
    def _validate_required_variables(self, result: ValidationResult):
        """Validate all required environment variables"""
        
        for var_name, validator in self.required_vars.items():
            value = os.getenv(var_name)
            
            if not value:
                result.add_error(f"Missing required environment variable: {var_name}")
                continue
                
            # Run specific validator
            if not validator(value):
                result.add_error(f"Invalid value for {var_name}: {value}")
                
        # Check wallet configuration
        encrypted_key = os.getenv('ENCRYPTED_WALLET_KEY')
        auto_create = os.getenv('AUTO_CREATE_WALLET', 'false').lower() == 'true'
        
        if not encrypted_key and not auto_create:
            result.add_error("Either ENCRYPTED_WALLET_KEY or AUTO_CREATE_WALLET=true must be set")
            
        if encrypted_key and not os.getenv('WALLET_PASSWORD'):
            result.add_error("WALLET_PASSWORD required when using ENCRYPTED_WALLET_KEY")
    
    def _validate_security_settings(self, result: ValidationResult):
        """Validate security-related settings"""
        
        # Check security level
        security_level = os.getenv('SECURITY_LEVEL', 'high')
        if security_level.lower() not in ['high', 'maximum']:
            result.add_warning(f"Security level '{security_level}' is not recommended for production. Use 'high' or 'maximum'")
            
        # Validate password strength
        for var_name, validator in self.security_rules.items():
            value = os.getenv(var_name)
            if value and not validator(value):
                result.add_error(f"Weak password for {var_name}. Use a strong password with 12+ characters, mixed case, numbers, and symbols")
                
        # Check kill switch settings
        if os.getenv('ENABLE_KILL_SWITCH', 'true').lower() != 'true':
            result.add_error("ENABLE_KILL_SWITCH must be 'true' for production deployment")
            
        if os.getenv('EMERGENCY_STOP_ENABLED', 'true').lower() != 'true':
            result.add_error("EMERGENCY_STOP_ENABLED must be 'true' for production deployment")
    
    def _validate_network_settings(self, result: ValidationResult):
        """Validate network and RPC settings"""
        
        # Validate primary RPC
        rpc_url = os.getenv('SOLANA_RPC_URL')
        if rpc_url and not self._validate_rpc_url(rpc_url):
            result.add_error(f"Invalid Solana RPC URL: {rpc_url}")
            
        # Check backup RPCs
        backup_urls = [
            os.getenv('SOLANA_RPC_BACKUP_1'),
            os.getenv('SOLANA_RPC_BACKUP_2')
        ]
        
        valid_backups = sum(1 for url in backup_urls if url and self._validate_rpc_url(url))
        if valid_backups < 1:
            result.add_warning("Consider setting backup RPC URLs for better reliability")
            
        # Check private RPC recommendation
        if not os.getenv('PRIVATE_RPC_URL'):
            result.add_info("Consider using a private RPC endpoint for better performance and reliability")
    
    def _validate_trading_parameters(self, result: ValidationResult):
        """Validate trading configuration parameters"""
        
        # Validate trading mode
        trading_mode = os.getenv('TRADING_MODE', '').upper()
        if trading_mode not in ['LIVE', 'SIMULATION']:
            result.add_error(f"TRADING_MODE must be 'LIVE' or 'SIMULATION', got: {trading_mode}")
            
        if trading_mode == 'LIVE':
            result.add_info("Trading mode set to LIVE - ensure you're ready for real trading!")
            
        # Validate numeric parameters
        numeric_params = {
            'INITIAL_CAPITAL': (1.0, 100000.0),
            'MAX_DAILY_LOSS_SOL': (0.1, 100.0),
            'PROFIT_TARGET_PERCENT': (0.5, 100.0),
            'STOP_LOSS_PERCENT': (0.5, 50.0),
            'MAX_TRADES_PER_HOUR': (1, 100),
        }
        
        for param, (min_val, max_val) in numeric_params.items():
            value_str = os.getenv(param)
            if value_str:
                try:
                    value = float(value_str)
                    if not (min_val <= value <= max_val):
                        result.add_warning(f"{param} value {value} is outside recommended range ({min_val}-{max_val})")
                except ValueError:
                    result.add_error(f"Invalid numeric value for {param}: {value_str}")
    
    def _validate_alert_system(self, result: ValidationResult):
        """Validate alert system configuration - PURE ON-CHAIN ONLY"""
        
        alert_configured = False
        
        # SOCIAL MEDIA ALERTS REMOVED - Only file and webhook alerts
        
        # Check webhook alerts
        webhook_urls = os.getenv('WEBHOOK_URLS')
        if webhook_urls:
            alert_configured = True
            result.add_info("Webhook alerts configured")
            
        # File logging is always available
        alert_configured = True
        result.add_info("File-based alerts always available")
            
        if not webhook_urls:
            result.add_info("Consider configuring webhook URLs for external monitoring")
    
    def _validate_wallet_configuration(self, result: ValidationResult):
        """Validate wallet security configuration"""
        
        # Check wallet exposure limits
        max_exposure = os.getenv('MAX_WALLET_EXPOSURE_SOL')
        if max_exposure:
            try:
                exposure = float(max_exposure)
                if exposure > 50.0:
                    result.add_warning(f"High wallet exposure limit: {exposure} SOL. Consider reducing for better security")
            except ValueError:
                result.add_error(f"Invalid MAX_WALLET_EXPOSURE_SOL value: {max_exposure}")
                
        # Check wallet rotation
        if os.getenv('WALLET_ROTATION', 'true').lower() != 'true':
            result.add_warning("Wallet rotation disabled. Consider enabling for better security")
    
    def _validate_production_specific(self, result: ValidationResult):
        """Validate production-specific settings"""
        
        # Ensure we're not in debug mode
        if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
            result.add_warning("DEBUG_MODE is enabled. Disable for production deployment")
            
        # Check environment setting
        environment = os.getenv('ENVIRONMENT', 'production')
        if environment.lower() != 'production':
            result.add_warning(f"ENVIRONMENT is set to '{environment}'. Set to 'production' for production deployment")
            
        # Check development mode
        if os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true':
            result.add_warning("DEVELOPMENT_MODE is enabled. Disable for production deployment")
    
    def _validate_trading_mode(self, value: str) -> bool:
        """Validate trading mode"""
        return value.upper() in ['LIVE', 'SIMULATION', 'PAPER']
    
    def _validate_wallet_password(self, value: str) -> bool:
        """Validate wallet encryption password"""
        return len(value) >= 8 and value != 'CHANGE_ME_SUPER_SECURE_PASSWORD_123!'
    
    def _validate_rpc_url(self, value: str) -> bool:
        """Validate Solana RPC URL"""
        try:
            parsed = urlparse(value)
            return parsed.scheme in ['http', 'https'] and parsed.netloc
        except:
            return False
    
    # SOCIAL MEDIA VALIDATION FUNCTIONS REMOVED - PURE ON-CHAIN + AI ONLY
        pattern = r'^\d{8,10}:[A-Za-z0-9_-]{35}$'
        return re.match(pattern, value) is not None
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < 12:
            return False
            
        # Check for mixed case, numbers, and symbols
        has_lower = re.search(r'[a-z]', password)
        has_upper = re.search(r'[A-Z]', password)
        has_digit = re.search(r'\d', password)
        has_symbol = re.search(r'[!@#$%^&*(),.?":{}|<>]', password)
        
        return all([has_lower, has_upper, has_digit, has_symbol])
    
    def _log_validation_results(self, result: ValidationResult):
        """Log validation results"""
        
        if result.is_valid:
            self.logger.info("✅ Configuration validation passed")
        else:
            self.logger.error("❌ Configuration validation failed")
            
        for error in result.errors:
            self.logger.error(f"🚨 ERROR: {error}")
            
        for warning in result.warnings:
            self.logger.warning(f"⚠️ WARNING: {warning}")
            
        for info in result.info:
            self.logger.info(f"ℹ️ INFO: {info}")

def validate_production_config() -> bool:
    """Quick validation function for production configuration"""
    validator = ProductionConfigValidator()
    result = validator.validate_production_config()
    return result.is_valid

if __name__ == "__main__":
    # Test configuration validation
    is_valid = validate_production_config()
    exit(0 if is_valid else 1) 