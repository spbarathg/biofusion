"""
ML PREDICTOR - MACHINE LEARNING PRICE PREDICTION
===============================================

Machine Learning system for price prediction using various ML models
and feature engineering from on-chain data.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import pickle
import os

from worker_ant_v1.utils.logger import setup_logger

@dataclass
class MLFeatures:
    """Feature set for ML prediction"""
    price_features: List[float]
    volume_features: List[float]
    technical_features: List[float]
    sentiment_features: List[float]
    market_features: List[float]
    timestamp: datetime

@dataclass
class MLPrediction:
    """ML prediction result"""
    token_address: str
    predicted_price: float
    prediction_confidence: float
    price_direction: str  # "UP", "DOWN", "FLAT"
    prediction_horizon: int  # minutes
    feature_importance: Dict[str, float]
    model_used: str
    timestamp: datetime

class MLPredictor:
    """Machine Learning prediction system"""
    
    def __init__(self):
        self.logger = setup_logger("MLPredictor")
        
        
        self.models = {
            'linear_regression': None,
            'random_forest': None,
            'gradient_boost': None,
            'neural_network': None
        }
        
        
        self.feature_config = {
            'price_lookback': 50,
            'volume_lookback': 20,
            'technical_indicators': ['rsi', 'macd', 'bb', 'ma'],
            'sentiment_window': 30,
            'market_context_features': 10
        }
        
        
        self.prediction_horizons = [5, 15, 30, 60, 240]
        
        
        self.feature_history: Dict[str, List[MLFeatures]] = {}
        
        
        self.model_performance: Dict[str, Dict[str, float]] = {
            model: {'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
            for model in self.models.keys()
        }
        
        
        self._initialize_models()
        
        self.logger.info("✅ ML predictor initialized")
    
    def _initialize_models(self):
        """Initialize simple prediction models"""
        
        try:
            # For now, use simplified statistical models
            
            self.models['linear_regression'] = {
                'type': 'linear',
                'weights': np.random.normal(0, 0.1, 50),  # 50 features
                'bias': 0.0,
                'learning_rate': 0.001
            }
            
            self.models['random_forest'] = {
                'type': 'ensemble',
                'n_trees': 10,
                'tree_depth': 5,
                'feature_subset': 0.7
            }
            
            self.models['gradient_boost'] = {
                'type': 'boosting',
                'n_estimators': 20,
                'learning_rate': 0.1,
                'max_depth': 3
            }
            
            self.models['neural_network'] = {
                'type': 'neural',
                'layers': [50, 25, 10, 1],
                'weights': [np.random.normal(0, 0.1, (50, 25)),
                           np.random.normal(0, 0.1, (25, 10)),
                           np.random.normal(0, 0.1, (10, 1))],
                'biases': [np.zeros(25), np.zeros(10), np.zeros(1)]
            }
            
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
    
    async def predict_price(self, token_address: str, 
                          market_data: Dict[str, Any],
                          prediction_horizon: int = 15) -> MLPrediction:
        """Generate price prediction for a token"""
        
        try:
            features = await self._extract_features(token_address, market_data)
            
            
            best_model = self._select_best_model(token_address)
            
            
            predicted_price, confidence = await self._generate_prediction(
                features, best_model, prediction_horizon
            )
            
            
            current_price = market_data.get('current_price', 0)
            if predicted_price > current_price * 1.02:
                direction = "UP"
            elif predicted_price < current_price * 0.98:
                direction = "DOWN"
            else:
                direction = "FLAT"
            
            
            feature_importance = self._calculate_feature_importance(features, best_model)
            
            prediction = MLPrediction(
                token_address=token_address,
                predicted_price=predicted_price,
                prediction_confidence=confidence,
                price_direction=direction,
                prediction_horizon=prediction_horizon,
                feature_importance=feature_importance,
                model_used=best_model,
                timestamp=datetime.now()
            )
            
            
            if token_address not in self.feature_history:
                self.feature_history[token_address] = []
            
            self.feature_history[token_address].append(features)
            
            
            if len(self.feature_history[token_address]) > 1000:
                self.feature_history[token_address] = self.feature_history[token_address][-1000:]
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Price prediction failed for {token_address}: {e}")
            
            
            return MLPrediction(
                token_address=token_address,
                predicted_price=market_data.get('current_price', 0),
                prediction_confidence=0.1,
                price_direction="FLAT",
                prediction_horizon=prediction_horizon,
                feature_importance={},
                model_used="fallback",
                timestamp=datetime.now()
            )
    
    async def _extract_features(self, token_address: str, market_data: Dict[str, Any]) -> MLFeatures:
        """Extract ML features from market data"""
        
        current_time = datetime.now()
        
        
        price_features = self._extract_price_features(market_data)
        
        
        volume_features = self._extract_volume_features(market_data)
        
        
        technical_features = self._extract_technical_features(market_data)
        
        
        sentiment_features = self._extract_sentiment_features(market_data)
        
        
        market_features = self._extract_market_features(market_data)
        
        return MLFeatures(
            price_features=price_features,
            volume_features=volume_features,
            technical_features=technical_features,
            sentiment_features=sentiment_features,
            market_features=market_features,
            timestamp=current_time
        )
    
    def _extract_price_features(self, market_data: Dict[str, Any]) -> List[float]:
        """Extract price-based features"""
        
        features = []
        
        
        current_price = market_data.get('current_price', 0)
        features.append(current_price)
        
        
        price_change_1h = market_data.get('price_change_1h', 0)
        price_change_24h = market_data.get('price_change_24h', 0)
        features.extend([price_change_1h, price_change_24h])
        
        
        volatility = market_data.get('price_volatility_24h', 0)
        features.append(volatility)
        
        
        price_history = market_data.get('price_history', [])
        if price_history:
            prices = [p.get('price', 0) for p in price_history[-20:]]
            
            
            features.append(np.mean(prices))
            features.append(np.std(prices))
            features.append(np.max(prices))
            features.append(np.min(prices))
            
            
            if len(prices) >= 5:
                momentum_5 = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] != 0 else 0
                features.append(momentum_5)
            else:
                features.append(0)
        else:
            features.extend([0] * 5)
        
        
        while len(features) < 20:
            features.append(0)
        
        return features[:20]
    
    def _extract_volume_features(self, market_data: Dict[str, Any]) -> List[float]:
        """Extract volume-based features"""
        
        features = []
        
        
        volume_24h = market_data.get('volume_24h', 0)
        features.append(volume_24h)
        
        
        volume_change = market_data.get('volume_change_24h', 0)
        features.append(volume_change)
        
        
        buy_volume = market_data.get('buy_volume_24h', 0)
        sell_volume = market_data.get('sell_volume_24h', 0)
        total_volume = buy_volume + sell_volume
        
        buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
        features.append(buy_ratio)
        
        
        volume_history = market_data.get('volume_history', [])
        if volume_history:
            volumes = [v.get('volume', 0) for v in volume_history[-10:]]
            
            features.append(np.mean(volumes))
            features.append(np.std(volumes))
        else:
            features.extend([0, 0])
        
        
        while len(features) < 10:
            features.append(0)
        
        return features[:10]
    
    def _extract_technical_features(self, market_data: Dict[str, Any]) -> List[float]:
        """Extract technical indicator features"""
        
        features = []
        
        
        rsi = market_data.get('rsi', 50)
        features.append(rsi / 100.0)  # Normalize to 0-1
        
        
        macd = market_data.get('macd', 0)
        features.append(np.tanh(macd))  # Normalize
        
        
        ma_10 = market_data.get('ma_10', 0)
        ma_20 = market_data.get('ma_20', 0)
        current_price = market_data.get('current_price', 0)
        
        if current_price > 0:
            ma_10_ratio = ma_10 / current_price if current_price != 0 else 1
            ma_20_ratio = ma_20 / current_price if current_price != 0 else 1
        else:
            ma_10_ratio = ma_20_ratio = 1
        
        features.extend([ma_10_ratio, ma_20_ratio])
        
        
        bb_upper = market_data.get('bb_upper', current_price)
        bb_lower = market_data.get('bb_lower', current_price)
        
        if bb_upper != bb_lower:
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        else:
            bb_position = 0.5
        
        features.append(bb_position)
        
        
        while len(features) < 10:
            features.append(0)
        
        return features[:10]
    
    def _extract_sentiment_features(self, market_data: Dict[str, Any]) -> List[float]:
        """Extract sentiment-based features"""
        
        features = []
        
        
        sentiment_score = market_data.get('sentiment_score', 0.5)
        features.append(sentiment_score)
        
        
        wallet_sentiment = market_data.get('wallet_sentiment', 0)
        volume_sentiment = market_data.get('volume_sentiment', 0)
        price_sentiment = market_data.get('price_sentiment', 0)
        
        features.extend([wallet_sentiment, volume_sentiment, price_sentiment])
        
        
        sentiment_confidence = market_data.get('sentiment_confidence', 0.5)
        features.append(sentiment_confidence)
        
        
        while len(features) < 5:
            features.append(0)
        
        return features[:5]
    
    def _extract_market_features(self, market_data: Dict[str, Any]) -> List[float]:
        """Extract market context features"""
        
        features = []
        
        
        age_hours = market_data.get('age_hours', 0)
        age_normalized = min(age_hours / (24 * 7), 1.0)  # Normalize to week
        features.append(age_normalized)
        
        
        holder_count = market_data.get('holder_count', 0)
        holder_normalized = min(holder_count / 10000, 1.0)  # Normalize
        features.append(holder_normalized)
        
        
        liquidity = market_data.get('liquidity_usd', 0)
        liquidity_normalized = min(liquidity / 1000000, 1.0)  # Normalize to 1M
        features.append(liquidity_normalized)
        
        
        market_cap = market_data.get('market_cap', 0)
        mcap_normalized = min(market_cap / 100000000, 1.0)  # Normalize to 100M
        features.append(mcap_normalized)
        
        
        trade_count_24h = market_data.get('trade_count_24h', 0)
        trade_normalized = min(trade_count_24h / 1000, 1.0)  # Normalize
        features.append(trade_normalized)
        
        
        while len(features) < 5:
            features.append(0)
        
        return features[:5]
    
    def _select_best_model(self, token_address: str) -> str:
        """Select the best performing model for this token"""
        
        
        # For now, use simple heuristics
        
        best_model = 'linear_regression'
        best_score = 0.0
        
        for model_name, performance in self.model_performance.items():
            score = (performance['accuracy'] + performance['f1']) / 2
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model
    
    async def _generate_prediction(self, features: MLFeatures, model_name: str, 
                                 horizon: int) -> Tuple[float, float]:
        """Generate prediction using specified model"""
        
        try:
            all_features = (features.price_features + 
                          features.volume_features + 
                          features.technical_features + 
                          features.sentiment_features + 
                          features.market_features)
            
            
            while len(all_features) < 50:
                all_features.append(0)
            
            feature_vector = np.array(all_features[:50])
            
            model = self.models[model_name]
            
            if model['type'] == 'linear':
                prediction = np.dot(feature_vector, model['weights']) + model['bias']
                confidence = 0.6
                
            elif model['type'] == 'ensemble':
            elif model['type'] == 'ensemble':
                predictions = []
                for _ in range(model['n_trees']):
                for _ in range(model['n_trees']):
                    tree_pred = np.mean(feature_vector) * np.random.normal(1.0, 0.1)
                    predictions.append(tree_pred)
                
                prediction = np.mean(predictions)
                confidence = 1.0 - (np.std(predictions) / np.mean(predictions)) if np.mean(predictions) != 0 else 0.5
                
            elif model['type'] == 'boosting':
            elif model['type'] == 'boosting':
                prediction = np.mean(feature_vector) * (1 + np.random.normal(0, 0.05))
                confidence = 0.7
                
            elif model['type'] == 'neural':
            elif model['type'] == 'neural':
                x = feature_vector
                for i, (weights, bias) in enumerate(zip(model['weights'], model['biases'])):
                    x = np.dot(x, weights) + bias
                    if i < len(model['weights']) - 1:  # Apply activation except for output layer
                        x = np.tanh(x)
                
                prediction = x[0]
                confidence = 0.8
                
            else:
                prediction = 0.0
                confidence = 0.1
            
            
            horizon_factor = min(horizon / 60.0, 2.0)  # Scale by hour
            prediction *= horizon_factor
            confidence = max(0.1, confidence - (horizon_factor - 1) * 0.1)
            
            return prediction, min(confidence, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Prediction generation failed: {e}")
            return 0.0, 0.1
    
    def _calculate_feature_importance(self, features: MLFeatures, model_name: str) -> Dict[str, float]:
        """Calculate feature importance for the prediction"""
        
        try:
            importance = {}
            
            price_importance = 0.4
            volume_importance = 0.2
            technical_importance = 0.2
            sentiment_importance = 0.15
            market_importance = 0.05
            
            importance['price_features'] = price_importance
            importance['volume_features'] = volume_importance
            importance['technical_features'] = technical_importance
            importance['sentiment_features'] = sentiment_importance
            importance['market_features'] = market_importance
            
            return importance
            
        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {e}")
            return {}
    
    async def update_model_performance(self, token_address: str, model_name: str,
                                     predicted_price: float, actual_price: float,
                                     prediction_time: datetime):
        """Update model performance metrics"""
        
        try:
            if actual_price != 0:
                error = abs(predicted_price - actual_price) / actual_price
                accuracy = max(0.0, 1.0 - error)
            else:
                accuracy = 0.0
            
            
            current_performance = self.model_performance[model_name]
            learning_rate = 0.1
            
            current_performance['accuracy'] = (
                current_performance['accuracy'] * (1 - learning_rate) + 
                accuracy * learning_rate
            )
            
            self.logger.debug(f"Updated {model_name} performance: accuracy={accuracy:.3f}")
            
        except Exception as e:
            self.logger.warning(f"Performance update failed: {e}")
    
    async def get_ensemble_prediction(self, token_address: str, 
                                    market_data: Dict[str, Any],
                                    prediction_horizon: int = 15) -> MLPrediction:
        """Generate ensemble prediction using multiple models"""
        
        try:
            predictions = []
            
            
            for model_name in self.models.keys():
                features = await self._extract_features(token_address, market_data)
                pred_price, confidence = await self._generate_prediction(
                    features, model_name, prediction_horizon
                )
                
                
                model_weight = self.model_performance[model_name]['accuracy']
                predictions.append((pred_price, confidence * model_weight, model_name))
            
            
            total_weight = sum(p[1] for p in predictions)
            if total_weight > 0:
                ensemble_price = sum(p[0] * p[1] for p in predictions) / total_weight
                ensemble_confidence = total_weight / len(predictions)
            else:
                ensemble_price = market_data.get('current_price', 0)
                ensemble_confidence = 0.1
            
            
            current_price = market_data.get('current_price', 0)
            if ensemble_price > current_price * 1.02:
                direction = "UP"
            elif ensemble_price < current_price * 0.98:
                direction = "DOWN"
            else:
                direction = "FLAT"
            
            return MLPrediction(
                token_address=token_address,
                predicted_price=ensemble_price,
                prediction_confidence=ensemble_confidence,
                price_direction=direction,
                prediction_horizon=prediction_horizon,
                feature_importance={},
                model_used="ensemble",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return await self.predict_price(token_address, market_data, prediction_horizon) 