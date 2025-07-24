"""
Core trading system components
"""

# Core configuration management
from worker_ant_v1.core.unified_config import UnifiedConfigManager

# Safe imports with error handling
def safe_import():
    """Safely import core components"""
    components = {}
    
    try:
        from worker_ant_v1.core.unified_config import get_config_manager
        components['get_config_manager'] = get_config_manager
    except ImportError:
        components['get_config_manager'] = None
        
    try:
        from worker_ant_v1.core.wallet_manager import UnifiedWalletManager, get_wallet_manager
        components['UnifiedWalletManager'] = UnifiedWalletManager
        components['get_wallet_manager'] = get_wallet_manager
    except ImportError:
        components['UnifiedWalletManager'] = None
        components['get_wallet_manager'] = None
        
    try:
        from worker_ant_v1.core.unified_trading_engine import UnifiedTradingEngine, get_trading_engine
        components['UnifiedTradingEngine'] = UnifiedTradingEngine
        components['get_trading_engine'] = get_trading_engine
    except ImportError:
        components['UnifiedTradingEngine'] = None
        components['get_trading_engine'] = None

    try:
        from worker_ant_v1.safety.vault_wallet_system import VaultWalletSystem
        components['VaultWalletSystem'] = VaultWalletSystem
    except ImportError:
        components['VaultWalletSystem'] = None


        
    return components

# Initialize components
_components = safe_import()
get_config_manager = _components.get('get_config_manager')
UnifiedWalletManager = _components.get('UnifiedWalletManager')
get_wallet_manager = _components.get('get_wallet_manager')
UnifiedTradingEngine = _components.get('UnifiedTradingEngine')
get_trading_engine = _components.get('get_trading_engine')
VaultWalletSystem = _components.get('VaultWalletSystem')

__all__ = [
    'UnifiedConfigManager',
    'get_config_manager',
    'UnifiedWalletManager',
    'get_wallet_manager',
    'UnifiedTradingEngine',
    'get_trading_engine',
    'VaultWalletSystem'
]
