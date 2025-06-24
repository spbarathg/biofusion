"""
Stealth Mechanics System
========================

Randomizes wallet, slippage, timing, and size to evade detection.
Implements sophisticated anti-detection measures.
"""

import asyncio
import random
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class StealthProfile:
    """Profile for stealth trading behavior"""
    wallet_rotation_interval: int  # minutes
    min_delay_between_trades: int  # seconds
    max_delay_between_trades: int  # seconds
    slippage_variance: float  # percentage
    size_variance: float  # percentage
    timing_randomness: float  # 0-1


class StealthMechanics:
    """Implements anti-detection stealth measures"""
    
    def __init__(self):
        self.logger = logging.getLogger("StealthMechanics")
        
        # Stealth configuration
        self.stealth_enabled = True
        self.current_profile = StealthProfile(
            wallet_rotation_interval=30,  # 30 minutes
            min_delay_between_trades=5,   # 5 seconds minimum
            max_delay_between_trades=60,  # 60 seconds maximum
            slippage_variance=0.3,        # ±0.3%
            size_variance=0.2,            # ±20%
            timing_randomness=0.8         # High randomness
        )
        
        # Tracking
        self.last_trade_times = {}
        self.wallet_usage_count = {}
        self.behavior_patterns = []
        
    async def apply_stealth_delay(self, wallet_id: str) -> float:
        """Apply randomized delay before trade"""
        
        if not self.stealth_enabled:
            return 0.0
            
        # Calculate base delay
        min_delay = self.current_profile.min_delay_between_trades
        max_delay = self.current_profile.max_delay_between_trades
        
        # Add randomness based on recent activity
        if wallet_id in self.last_trade_times:
            time_since_last = time.time() - self.last_trade_times[wallet_id]
            if time_since_last < 30:  # If recent trade, longer delay
                min_delay *= 2
                max_delay *= 2
                
        # Random delay with some patterns to seem human-like
        if random.random() < 0.3:  # 30% chance for quick trades
            delay = random.uniform(min_delay, min_delay * 2)
        else:
            delay = random.uniform(min_delay, max_delay)
            
        # Apply timing randomness
        randomness = self.current_profile.timing_randomness
        delay *= random.uniform(1 - randomness * 0.5, 1 + randomness * 0.5)
        
        self.last_trade_times[wallet_id] = time.time() + delay
        
        self.logger.debug(f"Stealth delay for {wallet_id}: {delay:.1f}s")
        return delay
        
    def randomize_slippage(self, base_slippage: float) -> float:
        """Randomize slippage to avoid detection"""
        
        if not self.stealth_enabled:
            return base_slippage
            
        variance = self.current_profile.slippage_variance
        
        # Add random variance
        multiplier = random.uniform(1 - variance, 1 + variance)
        randomized_slippage = base_slippage * multiplier
        
        # Ensure reasonable bounds
        return max(0.1, min(randomized_slippage, 5.0))
        
    def randomize_trade_size(self, base_size: float) -> float:
        """Randomize trade size to avoid detection"""
        
        if not self.stealth_enabled:
            return base_size
            
        variance = self.current_profile.size_variance
        
        # Add random variance
        multiplier = random.uniform(1 - variance, 1 + variance)
        randomized_size = base_size * multiplier
        
        # Ensure minimum viable size
        return max(base_size * 0.5, randomized_size)
        
    def select_stealth_wallet(self, available_wallets: List[str]) -> str:
        """Select wallet with stealth considerations"""
        
        if not self.stealth_enabled or not available_wallets:
            return available_wallets[0] if available_wallets else ""
            
        # Calculate weights based on usage
        weights = []
        for wallet in available_wallets:
            usage_count = self.wallet_usage_count.get(wallet, 0)
            
            # Prefer less used wallets
            weight = 1.0 / (1 + usage_count * 0.1)
            
            # Check if wallet needs rotation
            if usage_count > 10:  # Rotate after 10 uses
                weight *= 0.1
                
            weights.append(weight)
            
        # Weighted random selection
        if sum(weights) == 0:
            return random.choice(available_wallets)
            
        selected_wallet = np.random.choice(available_wallets, p=np.array(weights)/sum(weights))
        
        # Update usage count
        self.wallet_usage_count[selected_wallet] = self.wallet_usage_count.get(selected_wallet, 0) + 1
        
        return selected_wallet
        
    def add_human_like_behavior(self, action: str) -> Dict:
        """Add human-like behavior patterns"""
        
        behavior = {
            "action": action,
            "timestamp": datetime.now(),
            "randomized": True
        }
        
        # Simulate human patterns
        if random.random() < 0.1:  # 10% chance of "mistakes"
            behavior["hesitation_delay"] = random.uniform(2, 8)
            
        if random.random() < 0.05:  # 5% chance of "second thoughts"
            behavior["cancel_and_retry"] = True
            
        self.behavior_patterns.append(behavior)
        
        # Keep only recent patterns
        if len(self.behavior_patterns) > 100:
            self.behavior_patterns = self.behavior_patterns[-100:]
            
        return behavior
        
    def get_stealth_metrics(self) -> Dict:
        """Get stealth system metrics"""
        
        return {
            "stealth_enabled": self.stealth_enabled,
            "wallets_tracked": len(self.wallet_usage_count),
            "total_wallet_uses": sum(self.wallet_usage_count.values()),
            "behavior_patterns": len(self.behavior_patterns),
            "avg_delay": (self.current_profile.min_delay_between_trades + self.current_profile.max_delay_between_trades) / 2
        }


# Global instance
stealth_mechanics = StealthMechanics() 