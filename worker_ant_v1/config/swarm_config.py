"""
Smart Ape Mode - Swarm Configuration
====================================

Configuration for aggressive meme strategy and ML model settings
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AggressiveMemeStrategy:
    """Configuration for aggressive meme trading strategy"""
    
    # Social media monitoring
    enable_twitter_sentiment: bool = True
    enable_reddit_sentiment: bool = True
    enable_telegram_monitoring: bool = True
    enable_discord_monitoring: bool = False
    
    # Sentiment thresholds
    min_sentiment_score: float = 0.6
    min_confidence_threshold: float = 0.7
    min_mention_count: int = 10
    
    # Trading parameters
    max_position_size: float = 0.02  # 2% of portfolio
    min_liquidity_usd: float = 50000
    max_age_hours: int = 24
    
    # Risk management
    stop_loss_percentage: float = 0.15  # 15% stop loss
    take_profit_percentage: float = 2.0  # 200% take profit
    max_simultaneous_positions: int = 5
    
    # Stealth settings
    enable_stealth_mode: bool = True
    wallet_rotation_frequency: int = 3  # trades per wallet
    fake_trade_probability: float = 0.1


@dataclass
class MLModelConfig:
    """Configuration for machine learning models"""
    
    # Model settings
    sentiment_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    crypto_model_name: str = "ElKulako/cryptobert"
    
    # LSTM model parameters
    lstm_sequence_length: int = 30
    lstm_units_per_layer: int = 50
    lstm_hidden_layers: int = 3
    lstm_dropout_rate: float = 0.2
    
    # Prediction parameters
    prediction_window_hours: int = 4
    confidence_threshold: float = 0.65
    max_features: int = 1000
    
    # Training settings
    retrain_frequency_hours: int = 24
    min_training_samples: int = 100
    validation_split: float = 0.2
    
    # Performance thresholds
    min_accuracy: float = 0.60
    min_precision: float = 0.55
    min_recall: float = 0.50


@dataclass
class SwarmBehaviorConfig:
    """Configuration for swarm behavior and coordination"""
    
    # Ant swarm settings
    ant_count: int = 10
    max_ant_lifetime_hours: int = 168  # 1 week
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    
    # Evolution parameters
    generations_per_day: int = 4
    elite_survival_rate: float = 0.2
    fitness_decay_rate: float = 0.95
    
    # Coordination
    consensus_threshold: float = 0.7
    max_divergence_allowed: float = 0.3
    communication_frequency_minutes: int = 15


# Global configuration instances
aggressive_meme_strategy = AggressiveMemeStrategy()
ml_model_config = MLModelConfig()
swarm_behavior_config = SwarmBehaviorConfig()


def load_config_from_env():
    """Load configuration overrides from environment variables"""
    
    # Update aggressive meme strategy from environment
    aggressive_meme_strategy.enable_twitter_sentiment = os.getenv(
        "ENABLE_TWITTER_SENTIMENT", "true"
    ).lower() == "true"
    
    aggressive_meme_strategy.enable_reddit_sentiment = os.getenv(
        "ENABLE_REDDIT_SENTIMENT", "true"
    ).lower() == "true"
    
    aggressive_meme_strategy.enable_telegram_monitoring = os.getenv(
        "ENABLE_TELEGRAM_MONITORING", "true"
    ).lower() == "true"
    
    aggressive_meme_strategy.max_position_size = float(
        os.getenv("MAX_POSITION_SIZE", "0.02")
    )
    
    aggressive_meme_strategy.enable_stealth_mode = os.getenv(
        "ENABLE_STEALTH_MODE", "true"
    ).lower() == "true"
    
    # Update ML model config from environment
    ml_model_config.sentiment_model_name = os.getenv(
        "SENTIMENT_MODEL_NAME", ml_model_config.sentiment_model_name
    )
    
    ml_model_config.confidence_threshold = float(
        os.getenv("ML_CONFIDENCE_THRESHOLD", "0.65")
    )
    
    # Update swarm behavior from environment
    swarm_behavior_config.ant_count = int(
        os.getenv("SWARM_ANT_COUNT", "10")
    )
    
    swarm_behavior_config.mutation_rate = float(
        os.getenv("SWARM_MUTATION_RATE", "0.1")
    )


def get_strategy_config() -> AggressiveMemeStrategy:
    """Get the current aggressive meme strategy configuration"""
    return aggressive_meme_strategy


def get_ml_config() -> MLModelConfig:
    """Get the current ML model configuration"""
    return ml_model_config


def get_swarm_config() -> SwarmBehaviorConfig:
    """Get the current swarm behavior configuration"""
    return swarm_behavior_config


# Load environment overrides on import
load_config_from_env() 