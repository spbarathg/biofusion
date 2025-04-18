"""
Centralized path management for AntBot
"""
import os
from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
WALLETS_DIR = DATA_DIR / "wallets"
BACKUPS_DIR = DATA_DIR / "backups"

# Make sure directories exist
LOGS_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)
WALLETS_DIR.mkdir(exist_ok=True, parents=True)
BACKUPS_DIR.mkdir(exist_ok=True, parents=True)

# Common paths
CONFIG_PATH = CONFIG_DIR / "settings.yaml"
QUEEN_CONFIG_PATH = CONFIG_DIR / "queen.yaml"
ENCRYPTION_KEY_PATH = DATA_DIR / ".encryption_key"

def get_path(relative_path):
    """Get absolute path relative to project root"""
    if not relative_path:
        raise ValueError("Path cannot be empty")
    
    # Handle absolute paths
    if os.path.isabs(relative_path):
        return Path(relative_path)
    
    # Handle paths with ./ at the beginning
    if relative_path.startswith('./'):
        relative_path = relative_path[2:]
        
    return ROOT_DIR / relative_path 