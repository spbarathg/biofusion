import os
import sys
import time
import asyncio
import json
from pathlib import Path

# Ensure our lib directories are in the path
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))
sys.path.append(os.path.abspath("./bindings"))

try:
    from src.core.wallet_manager import WalletManager
    from bindings.worker_bridge import WorkerBridge
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)

# Setup paths
CONFIG_DIR = Path("config")
if not CONFIG_DIR.exists():
    CONFIG_DIR.mkdir(parents=True)

# Configure settings for high frequency memecoin trading
MEMECOIN_CONFIG = {
    "risk": {
        "max_per_trade": 50.0,  # Never use more than 5% of capital per trade
        "min_profit_threshold": 0.005,  # 0.5% minimum profit
        "slippage_tolerance": 0.01,  # 1% slippage allowed
        "max_open_trades": 10,  # Allow up to 10 simultaneous trades
        "trade_timeout": 60  # Close trade if not executed within 60 seconds
    },
    "dex": {
        "preferred": ["jupiter", "raydium", "orca"],
        "memecoin_list": [
            "BONK", "WIF", "MYRO", "BOPE", "POPCAT", 
            "SLERF", "FLOKI", "MEMU", "JUP", "BOOK"
        ],
    },
    "trading": {
        "frequency_over_magnitude": True,  # Prioritize frequency
        "trade_size_strategy": "adaptive",  # Adapt trade size based on opportunity
        "max_trades_per_hour": 100,  # High frequency
        "max_trade_size": 100.0,  # Maximum SOL per trade
        "default_trade_size_percentage": 0.02  # Default trade size 2% of capital
    }
}

# Create or update the memecoin trading config
def setup_config():
    config_file = CONFIG_DIR / "memecoin_settings.json"
    
    # Save the configuration
    with open(config_file, "w") as f:
        json.dump(MEMECOIN_CONFIG, f, indent=2)
    
    print(f"✅ Memecoin trading configuration saved to {config_file}")
    return config_file

async def setup_wallets():
    # Initialize wallet manager
    wallet_manager = WalletManager()
    
    # List existing wallets
    print("Checking existing wallets...")
    wallets = wallet_manager.list_wallets()
    
    if not wallets:
        print("No wallets found. Creating new queen wallet...")
        queen_id = wallet_manager.create_wallet("Memecoin Queen", "queen")
        print(f"✅ Created queen wallet: {queen_id}")
    else:
        queen_wallets = [w for w in wallets if w["type"] == "queen"]
        if queen_wallets:
            print(f"✅ Found existing queen wallet: {queen_wallets[0]['id']}")
            queen_id = queen_wallets[0]['id']
        else:
            queen_id = wallet_manager.create_wallet("Memecoin Queen", "queen")
            print(f"✅ Created queen wallet: {queen_id}")
    
    # Create worker wallets if needed
    worker_wallets = [w for w in wallet_manager.list_wallets() if w["type"] == "worker"]
    
    if len(worker_wallets) < 3:
        print("Creating worker wallets...")
        for i in range(3 - len(worker_wallets)):
            worker_id = wallet_manager.create_wallet(f"Memecoin Worker {i+1}", "worker")
            print(f"✅ Created worker wallet: {worker_id}")
    else:
        print(f"✅ Found {len(worker_wallets)} existing worker wallets")
    
    return wallet_manager

async def setup_trading(wallet_manager, config_path):
    # Initialize worker bridge
    worker_bridge = WorkerBridge(config_path=str(config_path))
    
    # Get the queen wallet
    queen_wallets = [w for w in wallet_manager.list_wallets() if w["type"] == "queen"]
    queen_id = queen_wallets[0]['id']
    queen_public_key = queen_wallets[0]['public_key']
    
    print(f"Setting up trading with queen wallet: {queen_id}")
    
    # Check queen wallet balance
    balance = await wallet_manager.get_balance(queen_id)
    
    if balance < 0.1:
        print(f"⚠️ Warning: Queen wallet has low balance: {balance} SOL")
        print("Please fund your wallet before starting trading")
        return False
    
    print(f"✅ Queen wallet balance: {balance} SOL")
    
    # Get worker wallets
    worker_wallets = [w for w in wallet_manager.list_wallets() if w["type"] == "worker"]
    
    # Set initial capital for workers
    total_workers = len(worker_wallets)
    capital_per_worker = balance * 0.8 / total_workers  # Use 80% of balance for trading
    
    print(f"Starting {total_workers} workers with {capital_per_worker} SOL each")
    
    # Start worker ants
    started_workers = []
    for worker in worker_wallets:
        worker_id = worker['id']
        worker_pubkey = worker['public_key']
        
        success = await worker_bridge.start_worker(
            worker_id=worker_id,
            wallet_address=worker_pubkey,
            capital=capital_per_worker
        )
        
        if success:
            started_workers.append(worker_id)
            print(f"✅ Started worker {worker_id}")
        else:
            print(f"❌ Failed to start worker {worker_id}")
    
    print(f"✅ Started {len(started_workers)} workers for memecoin trading")
    
    return started_workers

async def main():
    print("🐜 Setting up AntBot for High-Frequency Memecoin Trading")
    print("=" * 60)
    
    # Setup configuration
    config_path = setup_config()
    
    # Setup wallets
    wallet_manager = await setup_wallets()
    
    # Setup trading
    workers = await setup_trading(wallet_manager, config_path)
    
    if workers:
        print("\n" + "=" * 60)
        print("🚀 AntBot is ready for high-frequency memecoin trading!")
        print(f"🐜 {len(workers)} worker ants are searching for opportunities")
        print("\nMonitor your trading dashboard to track performance")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️ AntBot setup incomplete. Please check the errors above.")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main()) 