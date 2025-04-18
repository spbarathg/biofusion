#!/usr/bin/env python3
"""
AntBot - Entry point script to start the bot

This script provides a convenient way to start either the Queen or the Dashboard.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Ensure the script can be run from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import core components
from src.utils.logging.logger import setup_logging
from src.core.agents.queen import Queen
import src.dashboard.app as dashboard_app

# Set up logging
logger = setup_logging("antbot", "antbot.log")

async def run_queen(args):
    """Run the Queen agent"""
    queen = Queen(args.config)
    
    if args.init_capital:
        await queen.initialize_colony(args.init_capital)
    
    if args.state:
        state = await queen.get_colony_state()
        import json
        print(json.dumps(state, indent=2))
        return
    
    if args.backup:
        backup_path = await queen.backup_wallets()
        print(f"Wallets backed up to: {backup_path}")
        return
    
    if args.stop:
        await queen.stop_colony()
        return
    
    # Default behavior: initialize and run the colony
    if not (args.init_capital or args.state or args.backup or args.stop):
        balance = 10.0  # Default balance
        await queen.initialize_colony(balance)
        try:
            logger.info(f"AntBot Queen started with {balance} SOL. Press Ctrl+C to stop.")
            # Run forever until interrupted
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            logger.info("Stopping AntBot Queen...")
            await queen.stop_colony()
            logger.info("AntBot Queen stopped.")

def run_dashboard(args):
    """Run the dashboard"""
    try:
        import streamlit
        # Check if we're already in a Streamlit runtime
        if os.environ.get("STREAMLIT_RUNTIME") == "1":
            dashboard_app.render_streamlit_dashboard()
        else:
            # Launch Streamlit
            dashboard_file = Path(__file__).parent / "src" / "dashboard" / "app.py"
            os.system(f"streamlit run {dashboard_file}")
    except ImportError:
        print("Streamlit not installed. Please install with: pip install -e '.[dashboard]'")
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AntBot Trading System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Queen command
    queen_parser = subparsers.add_parser("queen", help="Run the Queen agent")
    queen_parser.add_argument("--config", type=str, default="config/settings.yaml", 
                              help="Path to configuration file")
    queen_parser.add_argument("--init-capital", type=float, help="Initial capital in SOL")
    queen_parser.add_argument("--state", action="store_true", help="Show colony state")
    queen_parser.add_argument("--backup", action="store_true", help="Backup wallets")
    queen_parser.add_argument("--stop", action="store_true", help="Stop the colony")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Run the dashboard")
    dashboard_parser.add_argument("--config", type=str, default="config/settings.yaml",
                                help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.command == "queen":
        asyncio.run(run_queen(args))
    elif args.command == "dashboard":
        run_dashboard(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 