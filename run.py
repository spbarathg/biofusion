#!/usr/bin/env python3
"""
Worker Ant V1 - Quick Start Script
=================================

Simple launcher for the Worker Ant V1 MVP trading bot.
"""

import sys
import os
import subprocess

def main():
    """Launch Worker Ant V1"""
    
    print("🐜 Starting Worker Ant V1 - MVP Trading Bot")
    print("=" * 50)
    
    # Change to worker_ant_v1 directory and run
    worker_dir = os.path.join(os.path.dirname(__file__), 'worker_ant_v1')
    
    if not os.path.exists(worker_dir):
        print("❌ Error: worker_ant_v1 directory not found!")
        print("Make sure you're running this from the project root.")
        sys.exit(1)
    
    try:
        # Run the worker ant
        subprocess.run([sys.executable, 'run.py'], cwd=worker_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Worker Ant V1: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Worker Ant V1 stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main() 