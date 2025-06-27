"""
Smart Ape Mode - Engines Module
===============================

AI and trading engines including:
- Evolution engine with genetic algorithms
- Trading engine with dual confirmation
- Stealth engine for anti-detection
- Profit management engine
- Stealth warfare mechanics
"""

from worker_ant_v1.engines.evolution_engine import EvolutionarySwarmAI
from worker_ant_v1.engines.trading_engine import SmartEntryEngine, DualConfirmation, AntiHypeFilter, TradingSignal, SignalTrust, MemoryPattern
from worker_ant_v1.engines.stealth_engine import StealthSwarmMechanics, StealthWallet, WalletRotationManager, FakeTradeGenerator
from worker_ant_v1.engines.profit_manager import ProfitDisciplineEngine, PositionManager, ProfitTarget, StopLoss
from worker_ant_v1.engines.stealth_warfare import AdvancedStealthWarfare

__all__ = [
    'EvolutionarySwarmAI',
    'SmartEntryEngine',
    'DualConfirmation', 
    'AntiHypeFilter',
    'TradingSignal',
    'SignalTrust',
    'MemoryPattern',
    'StealthSwarmMechanics',
    'StealthWallet',
    'WalletRotationManager', 
    'FakeTradeGenerator',
    'ProfitDisciplineEngine',
    'PositionManager',
    'ProfitTarget',
    'StopLoss',
    'AdvancedStealthWarfare'
]
