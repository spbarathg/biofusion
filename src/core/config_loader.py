import os
import yaml
from typing import Dict, Any, Optional

class ConfigLoader:
    """
    Centralized configuration loader for Ant Bot.
    Loads and provides access to configuration values from settings.yaml.
    """
    
    _instance = None
    _config = None
    
    def __new__(cls, config_path: str = "config/settings.yaml"):
        """Singleton pattern to ensure only one config loader instance exists."""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._config_path = config_path
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """Load the configuration file."""
        try:
            with open(self._config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            self._config = {}
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self._config.copy() if self._config else {}
    
    def get_colony_config(self) -> Dict[str, Any]:
        """Get colony management configuration."""
        return self._config.get('colony', {}) if self._config else {}
    
    def get_capital_config(self) -> Dict[str, Any]:
        """Get capital management configuration."""
        return self._config.get('capital', {}) if self._config else {}
    
    def get_worker_config(self) -> Dict[str, Any]:
        """Get worker configuration."""
        return self._config.get('worker', {}) if self._config else {}
    
    def get_princess_config(self) -> Dict[str, Any]:
        """Get princess configuration."""
        return self._config.get('princess', {}) if self._config else {}
    
    def get_drone_config(self) -> Dict[str, Any]:
        """Get drone configuration."""
        return self._config.get('drone', {}) if self._config else {}
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration."""
        return self._config.get('risk', {}) if self._config else {}
    
    def get_network_config(self) -> Dict[str, Any]:
        """Get network settings configuration."""
        return self._config.get('network', {}) if self._config else {}
    
    def get_dex_config(self) -> Dict[str, Any]:
        """Get DEX preferences configuration."""
        return self._config.get('dex', {}) if self._config else {}
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring and logging configuration."""
        return self._config.get('monitoring', {}) if self._config else {}
    
    def get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment configuration."""
        return self._config.get('deployment', {}) if self._config else {}
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self._config.get('security', {}) if self._config else {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value by key.
        
        Args:
            key: Dot notation key path (e.g., 'colony.initial_capital')
            default: Default value if key not found
            
        Returns:
            Configuration value or default if not found
        """
        if not self._config:
            return default
        
        parts = key.split('.')
        value = self._config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def reload(self) -> None:
        """Reload the configuration file."""
        self._load_config()


# Create a singleton instance for import
config = ConfigLoader()

# Convenience function to get config values
def get_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value by key.
    
    Args:
        key: Dot notation key path (e.g., 'colony.initial_capital')
        default: Default value if key not found
        
    Returns:
        Configuration value or default if not found
    """
    return config.get(key, default) 