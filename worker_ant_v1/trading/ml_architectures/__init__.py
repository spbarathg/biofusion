"""
ML ARCHITECTURES - STATE-OF-THE-ART PREDICTION ENGINE
====================================================

Next-generation machine learning core for Antbot's multi-agent Solana memecoin trading swarm.

This package contains three primary ML systems:

1. Oracle Ant - Predictive Time-Series Transformer
2. Hunter Ant - Reinforcement Learning Trading Agent  
3. Network Ant - Graph Neural Network for On-Chain Intelligence

These systems replace the current ml_predictor.py with advanced architectures
designed for adversarial crypto markets.
"""

from .oracle_ant import OracleAnt
from .hunter_ant import HunterAnt
from .network_ant import NetworkAnt
from .prediction_engine import PredictionEngine

__all__ = [
    'OracleAnt',
    'HunterAnt', 
    'NetworkAnt',
    'PredictionEngine'
] 