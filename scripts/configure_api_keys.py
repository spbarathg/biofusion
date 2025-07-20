"""
API KEY CONFIGURATION HELPER
============================
Interactive setup for missing API keys with Vault support

Modern version pushes secrets to HashiCorp Vault for production security.
Falls back to file-based configuration for development environments.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from worker_ant_v1.core.secrets_manager import get_secrets_manager, SecretsConfig, SecretProvider

def _get_api_keys_config() -> Dict[str, Dict[str, Any]]:
    """Get API keys configuration"""
    return {
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
        'QUICKNODE_RPC_URL': {
            'description': 'High-performance RPC endpoint (Optional)',
            'help': 'Get from: https://quicknode.com',
            'required': False,
            'placeholder': 'https://your-endpoint.solana-mainnet.quiknode.pro'
        }
    }


async def _check_vault_availability() -> bool:
    """Check if Vault is available and properly configured"""
    try:
        secrets_manager = await get_secrets_manager()
        health = await secrets_manager.health_check()
        return health["primary_healthy"]
    except Exception:
        return False


async def _get_current_secrets_from_vault(secrets_manager) -> Dict[str, str]:
    """Get current secrets from vault"""
    current_values = {}
    api_keys = _get_api_keys_config()
    
    for key in api_keys.keys():
        try:
            value = await secrets_manager.get_secret(key.lower())
            current_values[key] = value
        except Exception:
            current_values[key] = ''
    
    return current_values


def _get_current_secrets_from_file() -> Dict[str, str]:
    """Get current secrets from environment file"""
    env_file = str(Path(__file__).parent.parent / ".env.production")
    current_values = {}
    
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    current_values[key] = value
    
    return current_values


async def _interactive_secret_collection(
    api_keys: Dict[str, Dict[str, Any]], 
    current_values: Dict[str, str]
) -> Dict[str, str]:
    """Interactively collect API keys from user"""
    updated_keys = {}
    
    for key, info in api_keys.items():
        current_value = current_values.get(key, '')
        
        print(f"\n🔧 **{key}**")
        print(f"   Purpose: {info['description']}")
        print(f"   Get it: {info['help']}")
        
        if current_value and not current_value.startswith('your_'):
            masked_value = current_value[:8] + "..." if len(current_value) > 8 else current_value
            print(f"   Current: {masked_value}")
            response = input(f"   Keep current value? (y/n): ").strip().lower()
            if response != 'n':
                continue
        
        if info['required']:
            while True:
                new_value = input(f"   Enter {key} (REQUIRED): ").strip()
                if new_value.lower() == 'quit':
                    print("Configuration cancelled.")
                    return {}
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
                return {}
            if new_value and not new_value.startswith('your_'):
                updated_keys[key] = new_value
                print(f"   ✅ {key} configured")
            else:
                print(f"   ⏭️  Skipped optional key: {key}")
    
    return updated_keys


async def _store_secrets_in_vault(updated_keys: Dict[str, str]) -> bool:
    """Store secrets in HashiCorp Vault"""
    try:
        secrets_manager = await get_secrets_manager()
        success_count = 0
        
        print(f"\n🔐 **Storing secrets in Vault...**")
        
        for key, value in updated_keys.items():
            try:
                success = await secrets_manager.put_secret(key.lower(), value)
                if success:
                    success_count += 1
                    print(f"   ✅ {key} stored in Vault")
                else:
                    print(f"   ❌ Failed to store {key} in Vault")
            except Exception as e:
                print(f"   ❌ Error storing {key}: {e}")
        
        print(f"\n✅ Successfully stored {success_count}/{len(updated_keys)} secrets in Vault")
        return success_count == len(updated_keys)
        
    except Exception as e:
        print(f"❌ Vault storage failed: {e}")
        return False


def _store_secrets_in_file(updated_keys: Dict[str, str]) -> bool:
    """Store secrets in environment file (fallback)"""
    env_file = str(Path(__file__).parent.parent / ".env.production")
    env_lines = []
    
    # Read existing file
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            env_lines = f.readlines()
    
    # Update existing lines
    remaining_keys = updated_keys.copy()
    for i, line in enumerate(env_lines):
        line_stripped = line.strip()
        if '=' in line_stripped and not line_stripped.startswith('#'):
            key, _ = line_stripped.split('=', 1)
            if key in remaining_keys:
                env_lines[i] = f"{key}={remaining_keys[key]}\n"
                del remaining_keys[key]
    
    # Add new keys
    for key, value in remaining_keys.items():
        env_lines.append(f"{key}={value}\n")
    
    # Write file
    try:
        with open(env_file, 'w') as f:
            f.writelines(env_lines)
        print(f"✅ Updated {len(updated_keys)} API keys in {env_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to write to {env_file}: {e}")
        return False


async def _validate_configuration(use_vault: bool) -> bool:
    """Validate that all required secrets are properly configured"""
    api_keys = _get_api_keys_config()
    
    if use_vault:
        try:
            secrets_manager = await get_secrets_manager()
            current_values = await _get_current_secrets_from_vault(secrets_manager)
        except Exception:
            current_values = {}
    else:
        current_values = _get_current_secrets_from_file()
    
    print(f"\n📊 **CONFIGURATION STATUS**")
    missing_required = []
    
    for key, info in api_keys.items():
        if info['required']:
            current_value = current_values.get(key, '')
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
        if use_vault:
            print("Secrets stored securely in HashiCorp Vault!")
        else:
            print("Secrets stored in environment file.")
        print("System is ready for launch!")
        return True


async def configure_api_keys_modern() -> bool:
    """Modern async API key configuration with Vault support"""
    print("🔑 **MODERN API KEY CONFIGURATION ASSISTANT**")
    print("=" * 60)
    print("This script will help you configure API keys securely.")
    print("Press Enter to skip optional keys, or 'quit' to exit.\n")
    
    # Check vault availability
    vault_available = await _check_vault_availability()
    use_vault = False
    
    if vault_available:
        print("🔐 **HashiCorp Vault Available**")
        choice = input("Store secrets in Vault for production security? (y/n): ").strip().lower()
        use_vault = choice == 'y'
        
        if use_vault:
            print("✅ Using Vault for secure secret storage")
        else:
            print("📄 Using environment file storage")
    else:
        print("⚠️  Vault not available - using environment file storage")
        print("For production deployment, configure Vault for enhanced security.")
    
    # Get current configuration
    api_keys = _get_api_keys_config()
    
    if use_vault:
        try:
            secrets_manager = await get_secrets_manager()
            current_values = await _get_current_secrets_from_vault(secrets_manager)
        except Exception as e:
            print(f"❌ Error accessing Vault: {e}")
            print("Falling back to environment file...")
            use_vault = False
            current_values = _get_current_secrets_from_file()
    else:
        current_values = _get_current_secrets_from_file()
    
    # Interactive collection
    updated_keys = await _interactive_secret_collection(api_keys, current_values)
    
    if not updated_keys:
        print("No secrets to update.")
        return await _validate_configuration(use_vault)
    
    # Store secrets
    storage_success = False
    if use_vault:
        storage_success = await _store_secrets_in_vault(updated_keys)
        if not storage_success:
            print("⚠️  Vault storage failed - falling back to file storage")
            storage_success = _store_secrets_in_file(updated_keys)
    else:
        storage_success = _store_secrets_in_file(updated_keys)
    
    if not storage_success:
        print("❌ Failed to store secrets")
        return False
    
    # Final validation
    return await _validate_configuration(use_vault)


def configure_missing_api_keys():
    """Legacy synchronous API key configuration (backward compatibility)"""
    print("🔑 **API KEY CONFIGURATION ASSISTANT (Legacy Mode)**")
    print("=" * 60)
    print("This script will help you configure the missing API keys.")
    print("For modern Vault support, run with --modern flag.")
    print("Press Enter to skip optional keys, or 'quit' to exit.\n")
    
    # Use file-based configuration only
    env_file = str(Path(__file__).parent.parent / ".env.production")
    current_values = _get_current_secrets_from_file()
    api_keys = _get_api_keys_config()
    
    # Run synchronous collection (blocking)
    try:
        updated_keys = asyncio.run(_interactive_secret_collection(api_keys, current_values))
    except KeyboardInterrupt:
        print("Configuration cancelled.")
        return False
    
    if not updated_keys:
        print("No secrets to update.")
        return asyncio.run(_validate_configuration(False))
    
    # Store in file
    storage_success = _store_secrets_in_file(updated_keys)
    
    if not storage_success:
        print("❌ Failed to store secrets")
        return False
    
    # Final validation
    return asyncio.run(_validate_configuration(False))

def _ensure_env_file_exists():
    """Ensure .env.production file exists"""
    env_file = str(Path(__file__).parent.parent / '.env.production')
    
    if not os.path.exists(env_file):
        print("❌ .env.production file not found!")
        print("Creating from template...")
        
        template_file = str(Path(__file__).parent.parent / 'config/env.template')
        if os.path.exists(template_file):
            import shutil
            shutil.copy(template_file, env_file)
            print("✅ Created .env.production from template")
        else:
            # Create minimal env file
            with open(env_file, 'w') as f:
                f.write("# ANTBOT CONFIGURATION\n")
                f.write("TRADING_MODE=simulation\n")
                f.write("SECURITY_LEVEL=high\n")
                f.write("ENABLE_KILL_SWITCH=true\n")
                f.write("EMERGENCY_STOP_ENABLED=true\n")
                f.write("\n# API KEYS (will be stored in Vault for production)\n")
                f.write("HELIUS_API_KEY=your_helius_api_key_here\n")
                f.write("SOLANA_TRACKER_API_KEY=your_solana_tracker_api_key\n")
                f.write("JUPITER_API_KEY=your_jupiter_api_key\n")
                f.write("RAYDIUM_API_KEY=your_raydium_api_key\n")
            print("✅ Created minimal .env.production file")


async def main_async():
    """Main async configuration flow with modern Vault support"""
    print("🔥 **ANTBOT API KEY SETUP (MODERN)**")
    print("🎯 Configure API keys securely for system launch")
    print("=" * 60)
    
    _ensure_env_file_exists()
    
    success = await configure_api_keys_modern()
    
    if success:
        print(f"\n🎯 **NEXT STEPS:**")
        print("1. Run: python scripts/manage_db.py upgrade head")
        print("2. Run: python entry_points/run_bot.py --mode production")
        print("3. Monitor system health in Grafana dashboard")
    else:
        print(f"\n❌ **Configuration incomplete. Fix missing keys and try again.**")
    
    return success


def main():
    """Main configuration flow (legacy compatibility)"""
    # Check for modern flag
    if "--modern" in sys.argv:
        return asyncio.run(main_async())
    
    print("🔥 **ANTBOT API KEY SETUP (LEGACY)**")
    print("🎯 Configure API keys for system launch")
    print("=" * 60)
    print("💡 Tip: Use --modern flag for Vault support")
    
    _ensure_env_file_exists()
    
    success = configure_missing_api_keys()
    
    if success:
        print(f"\n🎯 **NEXT STEPS:**")
        print("1. Run: python scripts/manage_db.py upgrade head")
        print("2. Run: python entry_points/run_bot.py --mode production")
        print("3. For production security, consider using --modern flag with Vault")
    else:
        print(f"\n❌ **Configuration incomplete. Fix missing keys and try again.**")
    
    return success


if __name__ == "__main__":
    try:
        result = main()
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n🛑 Configuration cancelled by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Configuration failed: {e}")
        exit(1) 