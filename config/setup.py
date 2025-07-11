"""
SMART APE BOT SETUP SCRIPT
=========================

Quick setup script to configure the bot for first-time users.
Handles all the tedious configuration steps automatically.
"""

import os
import sys
import shutil
import secrets
import string
from pathlib import Path

class BotSetup:
    """Interactive bot setup"""
    
    def __init__(self):
        self.config_path = ".env.production"
        self.template_path = "env.template"
        
    def run_setup(self):
        """Run the complete setup process"""
        print("\n🦍 SMART APE BOT SETUP")
        print("=" * 40)
        print("Let's configure your trading bot!\n")
        
        
        if os.path.exists(self.config_path):
            response = input(f"Configuration file {self.config_path} already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Setup cancelled.")
                return
        
        
        if not self._copy_template():
            return
        
        
        self._configure_trading_mode()
        self._configure_security()
        self._configure_capital()
        self._configure_api_keys()
        self._configure_alerts()
        
        print("\n✅ Setup complete!")
        print("\nNext steps:")
        print("1. Review your configuration: nano .env.production")
        print("2. Test in simulation: python run_bot.py --mode simulation")
        print("3. Deploy production: python run_bot.py --mode production")
        print("\n🚀 Ready to trade!")
    
    def _copy_template(self) -> bool:
        """Copy environment template"""
        if not os.path.exists(self.template_path):
            print(f"❌ Template file {self.template_path} not found!")
            return False
        
        try:
            shutil.copy(self.template_path, self.config_path)
            print(f"✅ Created {self.config_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to copy template: {e}")
            return False
    
    def _configure_trading_mode(self):
        """Configure trading mode"""
        print("\n📊 TRADING MODE")
        print("1. simulation - Safe paper trading (recommended for testing)")
        print("2. live - Real money trading (requires SOL)")
        
        while True:
            choice = input("Select mode (1-2): ").strip()
            if choice == "1":
                mode = "simulation"
                break
            elif choice == "2":
                mode = "live"
                print("⚠️  WARNING: Live mode uses real money!")
                confirm = input("Are you sure? Type 'YES' to confirm: ")
                if confirm == "YES":
                    break
                else:
                    continue
            else:
                print("Please enter 1 or 2")
        
        self._update_config("TRADING_MODE", mode)
        print(f"✅ Trading mode set to: {mode}")
    
    def _configure_security(self):
        """Configure security settings"""
        print("\n🔐 SECURITY CONFIGURATION")
        
        
        wallet_password = self._generate_password()
        encryption_password = self._generate_password()
        
        self._update_config("WALLET_ENCRYPTION_PASSWORD", wallet_password)
        self._update_config("WALLET_PASSWORD", encryption_password)
        
        print("✅ Generated secure passwords")
        print("⚠️  IMPORTANT: Save these passwords securely!")
        print(f"Wallet Password: {wallet_password}")
        print(f"Encryption Password: {encryption_password}")
    
    def _configure_capital(self):
        """Configure capital settings"""
        print("\n💰 CAPITAL CONFIGURATION")
        
        while True:
            try:
                capital = float(input("Initial capital in SOL (default: 10.0): ") or "10.0")
                if capital > 0:
                    break
                else:
                    print("Capital must be positive")
            except ValueError:
                print("Please enter a valid number")
        
        self._update_config("INITIAL_CAPITAL_SOL", str(capital))
        print(f"✅ Initial capital set to: {capital} SOL")
    
    def _configure_api_keys(self):
        """Configure API keys"""
        print("\n🔑 API CONFIGURATION")
        print("Required APIs for optimal performance:")
        
        
        helius_key = input("Helius API key (get from helius.dev): ").strip()
        if helius_key:
            self._update_config("HELIUS_API_KEY", helius_key)
            print("✅ Helius API configured")
        
        
        tracker_key = input("Solana Tracker API key (optional): ").strip()
        if tracker_key:
            self._update_config("SOLANA_TRACKER_API_KEY", tracker_key)
            print("✅ Solana Tracker API configured")
    
    def _configure_alerts(self):
        """Configure alert system"""
        print("\n📢 ALERT CONFIGURATION")
        print("Configure at least one alert method:")
        
        
        discord_url = input("Discord webhook URL (optional): ").strip()
        if discord_url:
            self._update_config("DISCORD_WEBHOOK_URL", discord_url)
            print("✅ Discord alerts configured")
        
        
        telegram_token = input("Telegram bot token (optional): ").strip()
        if telegram_token:
            telegram_chat = input("Telegram chat ID: ").strip()
            self._update_config("TELEGRAM_BOT_TOKEN", telegram_token)
            self._update_config("TELEGRAM_CHAT_ID", telegram_chat)
            print("✅ Telegram alerts configured")
    
    def _generate_password(self, length: int = 16) -> str:
        """Generate a secure password"""
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(chars) for _ in range(length))
    
    def _update_config(self, key: str, value: str):
        """Update configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                lines = f.readlines()
            
            
            updated = False
            for i, line in enumerate(lines):
                if line.startswith(f"{key}="):
                    lines[i] = f"{key}={value}\n"
                    updated = True
                    break
            
            
            with open(self.config_path, 'w') as f:
                f.writelines(lines)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not update {key}: {e}")

def main():
    """Main setup function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        setup = BotSetup()
        if not os.path.exists(".env.production"):
            shutil.copy("env.template", ".env.production")
        print("✅ Quick setup complete! Edit .env.production manually.")
    else:
    else:
        setup = BotSetup()
        setup.run_setup()

if __name__ == "__main__":
    main() 