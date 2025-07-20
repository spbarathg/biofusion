"""
API KEY CONFIGURATION HELPER
============================
Interactive setup for missing API keys
"""

import os
import sys
from pathlib import Path

def configure_missing_api_keys():
    """Interactive configuration of missing API keys"""
    print("🔑 **API KEY CONFIGURATION ASSISTANT**")
    print("=" * 60)
    print("This script will help you configure the missing API keys.")
    print("Press Enter to skip optional keys, or 'quit' to exit.\n")
    
    
    env_file = str(Path(__file__).parent.parent / ".env.production")
    env_lines = []
    
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            env_lines = f.readlines()
    
    
    api_keys = {
        'HELIUS_API_KEY': {
            'description': 'Core Solana API (REQUIRED)',
            'help': 'Get from: https://helius.dev',
            'required': True,
            'placeholder': 'your_helius_api_key_here'
        },
        'SOLANA_TRACKER_API_KEY': {
            'description': 'Token tracking API (REQUIRED)',
            'help': 'Get from: https://solanatracker.io',
            'required': True,
            'placeholder': 'your_solana_tracker_api_key'
        },
        'JUPITER_API_KEY': {
            'description': 'Jupiter DEX API (REQUIRED)',
            'help': 'Get from: https://jup.ag/api',
            'required': True,
            'placeholder': 'your_jupiter_api_key'
        },
        'RAYDIUM_API_KEY': {
            'description': 'Raydium DEX API (REQUIRED)',
            'help': 'Get from: https://raydium.io/api',
            'required': True,
            'placeholder': 'your_raydium_api_key'
        },
        'DEXSCREENER_API_KEY': {
            'description': 'DEX data API (Optional - Performance boost)',
            'help': 'Get from: https://dexscreener.com/api',
            'required': False,
            'placeholder': 'your_dexscreener_api_key'
        },
        'BIRDEYE_API_KEY': {
            'description': 'Market analytics API (Optional - Better signals)',
            'help': 'Get from: https://birdeye.so',
            'required': False,
            'placeholder': 'your_birdeye_api_key'
        },
        'COINGECKO_API_KEY': {
            'description': 'Price feeds API (Optional - Backup data)',
            'help': 'Get from: https://coingecko.com/api',
            'required': False,
            'placeholder': 'your_coingecko_api_key'
        }
    }
    
    
    current_values = {}
    for line in env_lines:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            current_values[key] = value
    
    
    updated_keys = {}
    
    for key, info in api_keys.items():
        current_value = current_values.get(key, '')
        
        print(f"\n🔧 **{key}**")
        print(f"   Purpose: {info['description']}")
        print(f"   Get it: {info['help']}")
        
        if current_value and not current_value.startswith('your_'):
            print(f"   Current: {current_value[:8]}..." if len(current_value) > 8 else current_value)
            response = input(f"   Keep current value? (y/n): ").strip().lower()
            if response != 'n':
                continue
        
        if info['required']:
            while True:
                new_value = input(f"   Enter {key} (REQUIRED): ").strip()
                if new_value.lower() == 'quit':
                    print("Configuration cancelled.")
                    return False
                if new_value and not new_value.startswith('your_'):
                    updated_keys[key] = new_value
                    print(f"   ✅ {key} configured")
                    break
                elif info['required']:
                    print(f"   ❌ {key} is required for system operation")
                else:
                    print(f"   ⚠️  Skipping optional key: {key}")
                    break
        else:
            new_value = input(f"   Enter {key} (optional, Enter to skip): ").strip()
            if new_value.lower() == 'quit':
                print("Configuration cancelled.")
                return False
            if new_value and not new_value.startswith('your_'):
                updated_keys[key] = new_value
                print(f"   ✅ {key} configured")
            else:
                print(f"   ⏭️  Skipped optional key: {key}")
    
    
    if updated_keys:
        print(f"\n💾 **Updating {env_file}**")
        
        
        for i, line in enumerate(env_lines):
            line_stripped = line.strip()
            if '=' in line_stripped and not line_stripped.startswith('#'):
                key, _ = line_stripped.split('=', 1)
                if key in updated_keys:
                    env_lines[i] = f"{key}={updated_keys[key]}\n"
                    del updated_keys[key]
        
        
        for key, value in updated_keys.items():
            env_lines.append(f"{key}={value}\n")
        
        
        with open(env_file, 'w') as f:
            f.writelines(env_lines)
        
        print(f"✅ Updated {len(updated_keys)} API keys in {env_file}")
    else:
        print("No changes made to environment file.")
    
    
    print(f"\n📊 **CONFIGURATION STATUS**")
    missing_required = []
    
    for key, info in api_keys.items():
        if info['required']:
            current_value = current_values.get(key, updated_keys.get(key, ''))
            if not current_value or current_value.startswith('your_'):
                missing_required.append(key)
                print(f"❌ {key}: STILL MISSING")
            else:
                print(f"✅ {key}: CONFIGURED")
    
    if missing_required:
        print(f"\n🚨 **CRITICAL: {len(missing_required)} required API keys still missing**")
        print("System cannot launch until these are configured:")
        for key in missing_required:
            print(f"   • {key}")
        return False
    else:
        print(f"\n🚀 **ALL REQUIRED API KEYS CONFIGURED**")
        print("System is ready for launch!")
        return True

def main():
    """Main configuration flow"""
    print("🔥 **CRYPTO SWARM API KEY SETUP**")
    print("🎯 Configure missing API keys for system launch")
    print("=" * 60)
    
    if not os.path.exists('.env.production'):
        print("❌ .env.production file not found!")
        print("Creating from template...")
        
        
        if os.path.exists(str(Path(__file__).parent.parent / 'config/env.template')):
            import shutil
            shutil.copy(str(Path(__file__).parent.parent / 'config/env.template'), env_file)
            print("✅ Created .env.production from template")
        else:
            with open(env_file, 'w') as f:
                f.write("# CRYPTO SWARM CONFIGURATION\n")
                f.write("TRADING_MODE=simulation\n")
                f.write("SECURITY_LEVEL=high\n")
                f.write("ENABLE_KILL_SWITCH=true\n")
                f.write("EMERGENCY_STOP_ENABLED=true\n")
                f.write("\n# API KEYS\n")
                f.write("HELIUS_API_KEY=your_helius_api_key_here\n")
                f.write("SOLANA_TRACKER_API_KEY=your_solana_tracker_api_key\n")
            print("✅ Created minimal .env.production file")
    
    success = configure_missing_api_keys()
    
    if success:
        print(f"\n🎯 **NEXT STEPS:**")
        print("1. Run: python integrity_check.py")
        print("2. Verify all systems pass")
        print("3. Launch with: python entry_points/run_bot.py")
    else:
        print(f"\n❌ **Configuration incomplete. Fix missing keys and try again.**")
    
    return success

if __name__ == "__main__":
    result = main()
    exit(0 if result else 1) 