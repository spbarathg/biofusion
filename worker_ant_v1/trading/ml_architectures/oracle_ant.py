"""
ORACLE ANT - PREDICTIVE TIME-SERIES TRANSFORMER
==============================================

Transformer-based model for multi-variate time-series forecasting of token metrics.
Designed to predict price, volume, holder count, and other key metrics over short time horizons.

Key Features:
- Multi-head attention mechanism for dynamic feature weighting
- Multi-variate input handling (price, volume, sentiment, etc.)
- Probability distribution outputs for uncertainty quantification
- Adversarial robustness for noisy memecoin markets
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
import pickle
import os

from worker_ant_v1.utils.logger import setup_logger

@dataclass
class TimeSeriesFeatures:
    """Multi-variate time series features"""
    price: List[float]
    volume: List[float]
    liquidity: List[float]
    holders: List[float]
    sentiment: List[float]
    volatility: List[float]
    timestamp: datetime
    
    def to_tensor(self, sequence_length: int = 50) -> torch.Tensor:
        """Convert to PyTorch tensor with padding/truncation"""
        features = []
        
        # Normalize and pad/truncate each feature
        for feature_list in [self.price, self.volume, self.liquidity, 
                           self.holders, self.sentiment, self.volatility]:
            if len(feature_list) > sequence_length:
                feature_list = feature_list[-sequence_length:]
            elif len(feature_list) < sequence_length:
                # Pad with last known value
                padding = [feature_list[-1]] * (sequence_length - len(feature_list))
                feature_list = padding + feature_list
            
            # Normalize to [0, 1] range
            feature_array = np.array(feature_list)
            if feature_array.max() != feature_array.min():
                feature_array = (feature_array - feature_array.min()) / (feature_array.max() - feature_array.min())
            features.append(feature_array)
        
        return torch.tensor(features, dtype=torch.float32).T  # Shape: (seq_len, num_features)

@dataclass
class OraclePrediction:
    """Oracle Ant prediction result"""
    token_address: str
    predicted_price: float
    predicted_volume: float
    predicted_holders: float
    price_confidence: float
    volume_confidence: float
    holders_confidence: float
    price_distribution: List[float]  # Probability distribution
    prediction_horizon: int  # minutes
    attention_weights: Dict[str, float]  # Feature importance
    timestamp: datetime

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for time series"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attn_output, attention_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

class OracleAnt(nn.Module):
    """Oracle Ant - Transformer-based time series predictor"""
    
    def __init__(self, 
                 input_dim: int = 6,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 sequence_length: int = 50,
                 prediction_horizons: List[int] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons or [5, 15, 30]
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads for different prediction horizons
        self.output_heads = nn.ModuleDict({
            str(horizon): nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 3)  # price, volume, holders
            ) for horizon in self.prediction_horizons
        })
        
        # Confidence estimation
        self.confidence_heads = nn.ModuleDict({
            str(horizon): nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 3)  # confidence for each metric
            ) for horizon in self.prediction_horizons
        })
        
        # Distribution estimation (for uncertainty quantification)
        self.distribution_heads = nn.ModuleDict({
            str(horizon): nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 10)  # 10 bins for price distribution
            ) for horizon in self.prediction_horizons
        })
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        batch_size, seq_len, input_dim = x.size()
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        x = self.dropout(x)
        
        # Pass through transformer layers
        attention_weights_list = []
        for transformer_layer in self.transformer_layers:
            x, attention_weights = transformer_layer(x, mask)
            attention_weights_list.append(attention_weights)
        
        # Use the last token for prediction
        last_token = x[:, -1, :]  # (batch_size, d_model)
        
        # Generate predictions for each horizon
        predictions = {}
        for horizon in self.prediction_horizons:
            horizon_str = str(horizon)
            
            # Main predictions
            predictions[f'values_{horizon_str}'] = self.output_heads[horizon_str](last_token)
            
            # Confidence scores
            predictions[f'confidence_{horizon_str}'] = torch.sigmoid(
                self.confidence_heads[horizon_str](last_token)
            )
            
            # Price distribution
            price_dist = F.softmax(
                self.distribution_heads[horizon_str](last_token), dim=-1
            )
            predictions[f'distribution_{horizon_str}'] = price_dist
        
        # Average attention weights across layers
        avg_attention = torch.mean(torch.stack(attention_weights_list), dim=0)
        
        return {
            'predictions': predictions,
            'attention_weights': avg_attention,
            'last_hidden_state': last_token
        }

class OracleAntPredictor:
    """High-level interface for Oracle Ant predictions"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = setup_logger("OracleAnt")
        
        # Model configuration
        self.model = OracleAnt(
            input_dim=6,
            d_model=128,
            num_heads=8,
            num_layers=6,
            d_ff=512,
            dropout=0.1,
            sequence_length=50,
            prediction_horizons=[5, 15, 30]
        )
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Feature history for each token
        self.feature_history: Dict[str, deque] = {}
        
        # Performance tracking
        self.prediction_accuracy = {
            '5min': {'mae': 0.0, 'rmse': 0.0, 'count': 0},
            '15min': {'mae': 0.0, 'rmse': 0.0, 'count': 0},
            '30min': {'mae': 0.0, 'rmse': 0.0, 'count': 0}
        }
        
        self.logger.info("✅ Oracle Ant initialized")
    
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.logger.info(f"✅ Loaded Oracle Ant model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def save_model(self, model_path: str):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'prediction_accuracy': self.prediction_accuracy,
                'timestamp': datetime.now()
            }
            torch.save(checkpoint, model_path)
            self.logger.info(f"✅ Saved Oracle Ant model to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    async def predict(self, token_address: str, market_data: Dict[str, Any], 
                     horizon: int = 15) -> OraclePrediction:
        """Generate prediction for a token"""
        
        try:
            # Extract and prepare features
            features = await self._extract_features(token_address, market_data)
            
            # Add to history
            if token_address not in self.feature_history:
                self.feature_history[token_address] = deque(maxlen=100)
            self.feature_history[token_address].append(features)
            
            # Prepare input tensor
            if len(self.feature_history[token_address]) < 10:
                # Not enough history, return fallback prediction
                return self._fallback_prediction(token_address, market_data, horizon)
            
            # Create tensor from recent history
            recent_features = list(self.feature_history[token_address])[-50:]
            input_tensor = torch.stack([f.to_tensor() for f in recent_features])
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            # Extract predictions for requested horizon
            horizon_str = str(horizon)
            values = outputs['predictions'][f'values_{horizon_str}'][0]
            confidence = outputs['predictions'][f'confidence_{horizon_str}'][0]
            distribution = outputs['predictions'][f'distribution_{horizon_str}'][0]
            
            # Calculate attention weights for feature importance
            attention_weights = self._extract_feature_importance(
                outputs['attention_weights'][0], features
            )
            
            # Create prediction result
            prediction = OraclePrediction(
                token_address=token_address,
                predicted_price=float(values[0]),
                predicted_volume=float(values[1]),
                predicted_holders=float(values[2]),
                price_confidence=float(confidence[0]),
                volume_confidence=float(confidence[1]),
                holders_confidence=float(confidence[2]),
                price_distribution=distribution.tolist(),
                prediction_horizon=horizon,
                attention_weights=attention_weights,
                timestamp=datetime.now()
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {token_address}: {e}")
            return self._fallback_prediction(token_address, market_data, horizon)
    
    async def _extract_features(self, token_address: str, market_data: Dict[str, Any]) -> TimeSeriesFeatures:
        """Extract time series features from market data"""
        
        # Extract price history
        price_history = market_data.get('price_history', [])
        prices = [p.get('price', 0) for p in price_history[-50:]]
        
        # Extract volume history
        volume_history = market_data.get('volume_history', [])
        volumes = [v.get('volume', 0) for v in volume_history[-50:]]
        
        # Extract liquidity history
        liquidity_history = market_data.get('liquidity_history', [])
        liquidities = [l.get('liquidity', 0) for l in liquidity_history[-50:]]
        
        # Extract holder count history
        holder_history = market_data.get('holder_history', [])
        holders = [h.get('count', 0) for h in holder_history[-50:]]
        
        # Extract sentiment history
        sentiment_history = market_data.get('sentiment_history', [])
        sentiments = [s.get('score', 0) for s in sentiment_history[-50:]]
        
        # Calculate volatility
        volatilities = []
        for i in range(len(prices)):
            if i >= 10:
                recent_prices = prices[i-10:i]
                volatility = np.std(recent_prices) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0
                volatilities.append(volatility)
            else:
                volatilities.append(0)
        
        # Pad shorter sequences
        max_len = max(len(prices), len(volumes), len(liquidities), 
                     len(holders), len(sentiments), len(volatilities))
        
        prices = self._pad_sequence(prices, max_len)
        volumes = self._pad_sequence(volumes, max_len)
        liquidities = self._pad_sequence(liquidities, max_len)
        holders = self._pad_sequence(holders, max_len)
        sentiments = self._pad_sequence(sentiments, max_len)
        volatilities = self._pad_sequence(volatilities, max_len)
        
        return TimeSeriesFeatures(
            price=prices,
            volume=volumes,
            liquidity=liquidities,
            holders=holders,
            sentiment=sentiments,
            volatility=volatilities,
            timestamp=datetime.now()
        )
    
    def _pad_sequence(self, sequence: List[float], target_length: int) -> List[float]:
        """Pad or truncate sequence to target length"""
        if len(sequence) > target_length:
            return sequence[-target_length:]
        elif len(sequence) < target_length:
            if len(sequence) > 0:
                padding = [sequence[-1]] * (target_length - len(sequence))
                return padding + sequence
            else:
                return [0.0] * target_length
        return sequence
    
    def _extract_feature_importance(self, attention_weights: torch.Tensor, 
                                  features: TimeSeriesFeatures) -> Dict[str, float]:
        """Extract feature importance from attention weights"""
        
        # Average attention weights across heads and time steps
        avg_attention = torch.mean(attention_weights, dim=(0, 1))  # Average over heads and time
        
        # Map to feature names
        feature_names = ['price', 'volume', 'liquidity', 'holders', 'sentiment', 'volatility']
        
        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(avg_attention):
                importance[name] = float(avg_attention[i])
            else:
                importance[name] = 0.0
        
        return importance
    
    def _fallback_prediction(self, token_address: str, market_data: Dict[str, Any], 
                           horizon: int) -> OraclePrediction:
        """Generate fallback prediction when model fails"""
        
        current_price = market_data.get('current_price', 0)
        current_volume = market_data.get('current_volume', 0)
        current_holders = market_data.get('holder_count', 0)
        
        return OraclePrediction(
            token_address=token_address,
            predicted_price=current_price,
            predicted_volume=current_volume,
            predicted_holders=current_holders,
            price_confidence=0.1,
            volume_confidence=0.1,
            holders_confidence=0.1,
            price_distribution=[0.1] * 10,
            prediction_horizon=horizon,
            attention_weights={},
            timestamp=datetime.now()
        )
    
    async def update_performance(self, token_address: str, horizon: int, 
                               predicted: OraclePrediction, actual: Dict[str, float]):
        """Update prediction performance metrics"""
        
        try:
            horizon_str = str(horizon)
            
            # Calculate errors
            price_error = abs(predicted.predicted_price - actual.get('price', 0))
            volume_error = abs(predicted.predicted_volume - actual.get('volume', 0))
            
            # Update metrics
            if horizon_str in self.prediction_accuracy:
                metrics = self.prediction_accuracy[horizon_str]
                metrics['count'] += 1
                
                # Update MAE
                metrics['mae'] = (metrics['mae'] * (metrics['count'] - 1) + price_error) / metrics['count']
                
                # Update RMSE
                metrics['rmse'] = np.sqrt((metrics['rmse']**2 * (metrics['count'] - 1) + price_error**2) / metrics['count'])
                
        except Exception as e:
            self.logger.error(f"Failed to update performance: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'prediction_accuracy': self.prediction_accuracy,
            'active_tokens': len(self.feature_history),
            'model_status': 'active'
        } 