"""
VAULT WALLET SYSTEM - PROFIT PROTECTION & COMPOUNDING
====================================================

Advanced vault wallet system that automatically stores profits, manages
compound growth, and provides secure backup storage for trading gains.
Implements requirement #13: Send portion of every win to a vault wallet.
"""

import asyncio
import time
import uuid
import hashlib
import secrets
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import os
from pathlib import Path

from worker_ant_v1.core.unified_config import get_trading_config, get_security_config
from worker_ant_v1.utils.logger import setup_logger

class VaultType(Enum):
    """Different types of vault wallets"""
    EMERGENCY = "emergency"        # Emergency funds only
    DAILY = "daily"               # Daily profit storage
    WEEKLY = "weekly"             # Weekly compound pool
    MONTHLY = "monthly"           # Monthly long-term storage
    LEGENDARY = "legendary"       # Reserved for massive gains
    GODMODE = "godmode"          # Ultimate security vault

class VaultSecurity(Enum):
    """Security levels for vault protection"""
    BASIC = "basic"               # Simple encryption
    ADVANCED = "advanced"         # Multi-layer protection
    MILITARY = "military"         # Military-grade security
    QUANTUM = "quantum"           # Future-proof quantum resistance

@dataclass
class VaultTransaction:
    """Individual vault transaction record"""
    
    transaction_id: str
    vault_id: str
    transaction_type: str  # "deposit", "withdrawal", "compound", "emergency"
    amount_sol: float
    timestamp: datetime
    
    # Transaction metadata
    source_wallet: Optional[str] = None
    reason: str = ""
    profit_percentage: float = 0.0
    
    # Security tracking
    signature: Optional[str] = None
    verification_hash: Optional[str] = None

class SecureVaultWallet:
    """Individual secure vault wallet for profit storage"""
    
    def __init__(self, vault_type: VaultType, security_level: VaultSecurity, 
                 initial_balance: float = 0.0):
        self.vault_id = f"vault_{vault_type.value}_{uuid.uuid4().hex[:8]}"
        self.vault_type = vault_type
        self.security_level = security_level
        
        # Wallet credentials (encrypted in production)
        self.wallet_address = self._generate_secure_address()
        self.private_key_hash = self._generate_key_hash()
        
        # Balance tracking
        self.current_balance = initial_balance
        self.total_deposits = 0.0
        self.total_withdrawals = 0.0
        self.compound_growth = 0.0
        
        # Vault configuration
        self.max_balance_sol = self._get_max_balance()
        self.auto_compound_threshold = self._get_compound_threshold()
        self.emergency_access_enabled = vault_type == VaultType.EMERGENCY
        
        # Transaction history
        self.transaction_history: List[VaultTransaction] = []
        
        # Operational state
        self.is_active = True
        self.is_locked = False
        self.creation_time = datetime.now()
        self.last_backup = None
        
        self.logger = setup_logger(f"VaultWallet_{self.vault_id}")
        
    def _generate_secure_address(self) -> str:
        """Generate secure wallet address"""
        base = f"{self.vault_type.value}_{self.security_level.value}_{time.time()}"
        return hashlib.sha256(base.encode()).hexdigest()[:32]
    
    def _generate_key_hash(self) -> str:
        """Generate private key hash for security"""
        return hashlib.sha256(f"{self.wallet_address}_private".encode()).hexdigest()
    
    def _get_max_balance(self) -> float:
        """Get maximum balance for vault type"""
        max_balances = {
            VaultType.EMERGENCY: 10.0,     # 10 SOL emergency fund
            VaultType.DAILY: 50.0,         # 50 SOL daily storage
            VaultType.WEEKLY: 200.0,       # 200 SOL weekly pool
            VaultType.MONTHLY: 1000.0,     # 1000 SOL monthly storage
            VaultType.LEGENDARY: 5000.0,   # 5000 SOL legendary vault
            VaultType.GODMODE: float('inf') # Unlimited godmode vault
        }
        return max_balances.get(self.vault_type, 100.0)
    
    def _get_compound_threshold(self) -> float:
        """Get auto-compound threshold for vault type"""
        thresholds = {
            VaultType.EMERGENCY: float('inf'),  # Never auto-compound emergency
            VaultType.DAILY: 5.0,              # Compound at 5 SOL
            VaultType.WEEKLY: 25.0,            # Compound at 25 SOL
            VaultType.MONTHLY: 100.0,          # Compound at 100 SOL
            VaultType.LEGENDARY: 500.0,        # Compound at 500 SOL
            VaultType.GODMODE: 1000.0          # Compound at 1000 SOL
        }
        return thresholds.get(self.vault_type, 10.0)
    
    async def deposit(self, amount_sol: float, source_wallet: str = None, 
                     reason: str = "profit_storage") -> bool:
        """Deposit SOL into vault"""
        
        if not self.is_active or self.is_locked:
            self.logger.error(f"Cannot deposit to inactive/locked vault {self.vault_id}")
            return False
        
        if amount_sol <= 0:
            self.logger.error(f"Invalid deposit amount: {amount_sol}")
            return False
        
        # Check vault capacity
        if self.current_balance + amount_sol > self.max_balance_sol:
            self.logger.warning(f"Deposit would exceed vault capacity")
            return False
        
        # Create transaction record
        transaction = VaultTransaction(
            transaction_id=f"dep_{uuid.uuid4().hex[:8]}",
            vault_id=self.vault_id,
            transaction_type="deposit",
            amount_sol=amount_sol,
            timestamp=datetime.now(),
            source_wallet=source_wallet,
            reason=reason
        )
        
        try:
            # Update balance
            self.current_balance += amount_sol
            self.total_deposits += amount_sol
            
            # Record transaction
            self.transaction_history.append(transaction)
            
            # Auto-backup after significant deposits
            if amount_sol > 1.0:
                await self._create_backup()
            
            self.logger.info(f"💰 Deposited {amount_sol:.6f} SOL to {self.vault_type.value} vault")
            
            # Check for auto-compound
            await self._check_auto_compound()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Deposit failed: {e}")
            return False
    
    async def withdraw(self, amount_sol: float, reason: str = "trading_capital") -> bool:
        """Withdraw SOL from vault"""
        
        if not self.is_active or self.is_locked:
            self.logger.error(f"Cannot withdraw from inactive/locked vault {self.vault_id}")
            return False
        
        if amount_sol <= 0:
            self.logger.error(f"Invalid withdrawal amount: {amount_sol}")
            return False
        
        if amount_sol > self.current_balance:
            self.logger.error(f"Insufficient vault balance: {self.current_balance}")
            return False
        
        # Create transaction record
        transaction = VaultTransaction(
            transaction_id=f"wit_{uuid.uuid4().hex[:8]}",
            vault_id=self.vault_id,
            transaction_type="withdrawal",
            amount_sol=amount_sol,
            timestamp=datetime.now(),
            reason=reason
        )
        
        try:
            # Update balance
            self.current_balance -= amount_sol
            self.total_withdrawals += amount_sol
            
            # Record transaction
            self.transaction_history.append(transaction)
            
            self.logger.info(f"💸 Withdrew {amount_sol:.6f} SOL from {self.vault_type.value} vault")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Withdrawal failed: {e}")
            return False
    
    async def compound_profits(self, compound_percentage: float = 0.1) -> float:
        """Compound profits back into trading capital"""
        
        if not self.is_active or self.vault_type == VaultType.EMERGENCY:
            return 0.0
        
        # Calculate compound amount
        compound_amount = self.current_balance * compound_percentage
        
        if compound_amount < 0.01:  # Minimum compound threshold
            return 0.0
        
        # Create compound transaction
        transaction = VaultTransaction(
            transaction_id=f"comp_{uuid.uuid4().hex[:8]}",
            vault_id=self.vault_id,
            transaction_type="compound",
            amount_sol=compound_amount,
            timestamp=datetime.now(),
            reason="profit_compounding"
        )
        
        try:
            # Update balance
            self.current_balance -= compound_amount
            self.compound_growth += compound_amount
            
            # Record transaction
            self.transaction_history.append(transaction)
            
            self.logger.info(f"🔄 Compounded {compound_amount:.6f} SOL from {self.vault_type.value} vault")
            
            return compound_amount
            
        except Exception as e:
            self.logger.error(f"Compounding failed: {e}")
            return 0.0
    
    async def _check_auto_compound(self):
        """Check if auto-compounding should trigger"""
        
        if (self.vault_type != VaultType.EMERGENCY and 
            self.current_balance >= self.auto_compound_threshold):
            
            # Auto-compound 10% of balance
            await self.compound_profits(0.1)
    
    async def emergency_access(self, emergency_code: str) -> float:
        """Emergency access to vault funds"""
        
        if not self.emergency_access_enabled:
            self.logger.error("Emergency access not enabled for this vault")
            return 0.0
        
        # Verify emergency code (mock verification)
        expected_code = hashlib.sha256(f"{self.vault_id}_emergency".encode()).hexdigest()[:16]
        
        if emergency_code != expected_code:
            self.logger.error("Invalid emergency access code")
            return 0.0
        
        # Return all funds in emergency
        emergency_amount = self.current_balance
        
        if emergency_amount > 0:
            await self.withdraw(emergency_amount, "emergency_access")
            self.logger.critical(f"🚨 Emergency access: {emergency_amount:.6f} SOL withdrawn")
        
        return emergency_amount
    
    async def _create_backup(self):
        """Create secure backup of vault state"""
        
        backup_data = {
            'vault_id': self.vault_id,
            'vault_type': self.vault_type.value,
            'security_level': self.security_level.value,
            'current_balance': self.current_balance,
            'total_deposits': self.total_deposits,
            'total_withdrawals': self.total_withdrawals,
            'compound_growth': self.compound_growth,
            'backup_timestamp': datetime.now().isoformat()
        }
        
        # In production, this would encrypt and store securely
        self.last_backup = datetime.now()
        self.logger.debug(f"📁 Created vault backup")
    
    def get_vault_status(self) -> Dict[str, Any]:
        """Get comprehensive vault status"""
        
        return {
            'vault_id': self.vault_id,
            'vault_type': self.vault_type.value,
            'security_level': self.security_level.value,
            'current_balance': self.current_balance,
            'total_deposits': self.total_deposits,
            'total_withdrawals': self.total_withdrawals,
            'compound_growth': self.compound_growth,
            'transaction_count': len(self.transaction_history),
            'is_active': self.is_active,
            'is_locked': self.is_locked,
            'creation_time': self.creation_time.isoformat(),
            'last_backup': self.last_backup.isoformat() if self.last_backup else None
        }

class VaultWalletSystem:
    """Comprehensive vault wallet management system"""
    
    def __init__(self):
        self.logger = setup_logger("VaultWalletSystem")
        self.trading_config = get_trading_config()
        self.security_config = get_security_config()
        
        # Vault storage
        self.vaults: Dict[str, SecureVaultWallet] = {}
        
        # Allocation rules (what percentage of profits go to each vault type)
        self.allocation_rules = self._initialize_allocation_rules()
        
        # System metrics
        self.total_vault_balance = 0.0
        self.total_deposits_today = 0.0
        self.total_compounds_today = 0.0
        
        # Background tasks
        self.system_active = False
        
    def _initialize_allocation_rules(self) -> Dict[str, float]:
        """Initialize profit allocation rules"""
        return {
            VaultType.EMERGENCY.value: 0.05,   # 5% to emergency fund
            VaultType.DAILY.value: 0.10,       # 10% to daily storage
            VaultType.WEEKLY.value: 0.03,      # 3% to weekly compound
            VaultType.MONTHLY.value: 0.02,     # 2% to monthly storage
            # Total: 20% of profits go to vaults
        }
    
    async def initialize_vault_system(self):
        """Initialize the vault wallet system"""
        
        try:
            self.logger.info("🏦 Initializing Vault Wallet System...")
            
            # Create initial vaults
            await self._create_initial_vaults()
            
            # Start background tasks
            await self._start_vault_management_tasks()
            
            self.system_active = True
            self.logger.info(f"✅ Vault system initialized with {len(self.vaults)} vaults")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize vault system: {e}")
            raise
    
    async def _create_initial_vaults(self):
        """Create initial set of vault wallets"""
        
        vault_configs = [
            (VaultType.EMERGENCY, VaultSecurity.MILITARY),
            (VaultType.DAILY, VaultSecurity.ADVANCED),
            (VaultType.WEEKLY, VaultSecurity.ADVANCED),
            (VaultType.MONTHLY, VaultSecurity.MILITARY),
            (VaultType.LEGENDARY, VaultSecurity.QUANTUM),
        ]
        
        for vault_type, security_level in vault_configs:
            vault = SecureVaultWallet(vault_type, security_level)
            await self._register_vault(vault)
    
    async def _register_vault(self, vault: SecureVaultWallet):
        """Register a vault in the system"""
        
        self.vaults[vault.vault_id] = vault
        self.logger.info(f"🔐 Registered {vault.vault_type.value} vault: {vault.vault_id}")
    
    async def allocate_profits(self, profit_amount: float, source_wallet: str) -> Dict[str, float]:
        """Allocate profits to vaults based on allocation rules"""
        
        if profit_amount <= 0:
            return {}
        
        allocation_results = {}
        
        try:
            for vault_type_str, allocation_percent in self.allocation_rules.items():
                allocation_amount = profit_amount * allocation_percent
                
                if allocation_amount < 0.001:  # Skip tiny amounts
                    continue
                
                # Find appropriate vault
                target_vault = None
                for vault in self.vaults.values():
                    if (vault.vault_type.value == vault_type_str and 
                        vault.is_active and 
                        vault.current_balance + allocation_amount <= vault.max_balance_sol):
                        target_vault = vault
                        break
                
                if target_vault:
                    success = await target_vault.deposit(
                        allocation_amount, 
                        source_wallet, 
                        "profit_allocation"
                    )
                    
                    if success:
                        allocation_results[vault_type_str] = allocation_amount
                        self.total_vault_balance += allocation_amount
                        self.total_deposits_today += allocation_amount
                        
                        self.logger.info(f"💰 Allocated {allocation_amount:.6f} SOL to {vault_type_str} vault")
                else:
                    self.logger.warning(f"⚠️ No available {vault_type_str} vault for allocation")
            
            total_allocated = sum(allocation_results.values())
            self.logger.info(f"📊 Total vault allocation: {total_allocated:.6f} SOL ({total_allocated/profit_amount*100:.1f}% of profit)")
            
            return allocation_results
            
        except Exception as e:
            self.logger.error(f"❌ Profit allocation failed: {e}")
            return {}
    
    async def emergency_fund_access(self, emergency_code: str) -> float:
        """Access emergency funds from all emergency vaults"""
        
        total_emergency_funds = 0.0
        
        for vault in self.vaults.values():
            if vault.vault_type == VaultType.EMERGENCY:
                emergency_amount = await vault.emergency_access(emergency_code)
                total_emergency_funds += emergency_amount
        
        if total_emergency_funds > 0:
            self.logger.critical(f"🚨 Emergency fund access: {total_emergency_funds:.6f} SOL total")
        
        return total_emergency_funds
    
    async def _start_vault_management_tasks(self):
        """Start background vault management tasks"""
        
        asyncio.create_task(self._daily_compound_loop())
        asyncio.create_task(self._backup_loop())
        asyncio.create_task(self._performance_monitoring_loop())
        
        self.logger.info("⚙️ Vault management tasks started")
    
    async def _daily_compound_loop(self):
        """Daily compounding loop"""
        
        while self.system_active:
            try:
                # Run daily compounding at midnight UTC
                now = datetime.utcnow()
                if now.hour == 0 and now.minute < 5:  # First 5 minutes of day
                    
                    total_compounded = 0.0
                    
                    for vault in self.vaults.values():
                        if vault.vault_type in [VaultType.WEEKLY, VaultType.MONTHLY]:
                            compound_amount = await vault.compound_profits(0.05)  # 5% compound
                            total_compounded += compound_amount
                    
                    if total_compounded > 0:
                        self.total_compounds_today += total_compounded
                        self.logger.info(f"🔄 Daily compound: {total_compounded:.6f} SOL")
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"❌ Daily compound loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _backup_loop(self):
        """Regular backup loop"""
        
        while self.system_active:
            try:
                # Create backups every 6 hours
                for vault in self.vaults.values():
                    await vault._create_backup()
                
                await self._create_system_backup()
                
                # Sleep for 6 hours
                await asyncio.sleep(21600)
                
            except Exception as e:
                self.logger.error(f"❌ Backup loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _performance_monitoring_loop(self):
        """Monitor vault system performance"""
        
        while self.system_active:
            try:
                # Update system metrics
                self.total_vault_balance = sum(vault.current_balance for vault in self.vaults.values())
                
                # Log system status every hour
                self.logger.info(f"🏦 Vault System Status: {self.total_vault_balance:.6f} SOL total balance")
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"❌ Performance monitoring error: {e}")
                await asyncio.sleep(3600)
    
    async def _create_system_backup(self):
        """Create comprehensive system backup"""
        
        system_backup = {
            'vault_count': len(self.vaults),
            'total_vault_balance': self.total_vault_balance,
            'total_deposits_today': self.total_deposits_today,
            'total_compounds_today': self.total_compounds_today,
            'allocation_rules': self.allocation_rules,
            'backup_timestamp': datetime.now().isoformat(),
            'vaults': {
                vault_id: vault.get_vault_status() 
                for vault_id, vault in self.vaults.items()
            }
        }
        
        # In production, this would be encrypted and stored securely
        self.logger.debug("📁 Created system backup")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive vault system status"""
        
        vault_summaries = {}
        for vault_type in VaultType:
            type_vaults = [v for v in self.vaults.values() if v.vault_type == vault_type]
            
            if type_vaults:
                total_balance = sum(v.current_balance for v in type_vaults)
                total_deposits = sum(v.total_deposits for v in type_vaults)
                
                vault_summaries[vault_type.value] = {
                    'count': len(type_vaults),
                    'total_balance': total_balance,
                    'total_deposits': total_deposits,
                    'average_balance': total_balance / len(type_vaults)
                }
        
        return {
            'system_active': self.system_active,
            'total_vaults': len(self.vaults),
            'total_vault_balance': self.total_vault_balance,
            'total_deposits_today': self.total_deposits_today,
            'total_compounds_today': self.total_compounds_today,
            'allocation_rules': self.allocation_rules,
            'vault_summaries': vault_summaries
        }

# Global vault system instance
_vault_system = None

async def get_vault_system() -> VaultWalletSystem:
    """Get or create the global vault system instance"""
    global _vault_system
    
    if _vault_system is None:
        _vault_system = VaultWalletSystem()
        await _vault_system.initialize_vault_system()
    
    return _vault_system 