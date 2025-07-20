"""
PREDICTION ENGINE - UNIFIED ML INTELLIGENCE CORE
===============================================

Main prediction engine that integrates all three ML architectures:
1. Oracle Ant - Time-series Transformer
2. Hunter Ant - Reinforcement Learning Agent  
3. Network Ant - Graph Neural Network

Provides unified interface for SwarmDecisionEngine and integrates with
existing Antbot systems like BattlePatternIntelligence and NightlyEvolutionSystem.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import os

from worker_ant_v1.utils.logger import setup_logger
from .oracle_ant import OracleAntPredictor, OraclePrediction
from .hunter_ant import HunterAnt, ActionType
from .network_ant import NetworkAntPredictor, NetworkPrediction, WalletType

@dataclass
class UnifiedPrediction:
    """Unified prediction combining all three ML architectures"""
    token_address: str
    
    # Oracle Ant predictions
    oracle_price_prediction: float
    oracle_volume_prediction: float
    oracle_holders_prediction: float
    oracle_confidence: float
    oracle_attention_weights: Dict[str, float]
    
    # Hunter Ant predictions
    hunter_action: int
    hunter_action_confidence: float
    hunter_value_estimate: float
    hunter_position_recommendation: str
    
    # Network Ant predictions
    network_contagion_score: float
    network_smart_money_probability: float
    network_manipulation_risk: float
    network_wallet_classifications: Dict[str, WalletType]
    network_influence_network: Dict[str, List[str]]
    
    # Consensus metrics
    overall_confidence: float
    consensus_score: float
    risk_assessment: Dict[str, float]
    trading_recommendation: str
    position_size_recommendation: float
    
    # Metadata
    prediction_horizon: int
    timestamp: datetime
    model_weights: Dict[str, float]  # Weight given to each model

@dataclass
class ModelPerformance:
    """Performance tracking for each model"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    profit_contribution: float
    last_updated: datetime

class PredictionEngine:
    """Unified prediction engine integrating all ML architectures"""
    
    def __init__(self, model_paths: Optional[Dict[str, str]] = None):
        self.logger = setup_logger("PredictionEngine")
        
        # Initialize ML architectures
        self.oracle_ant = OracleAntPredictor(
            model_path=model_paths.get('oracle') if model_paths else None
        )
        
        # Initialize Hunter Ant for each wallet (will be populated dynamically)
        self.hunter_ants: Dict[str, HunterAnt] = {}
        
        self.network_ant = NetworkAntPredictor(
            model_path=model_paths.get('network') if model_paths else None
        )
        
        # Model performance tracking
        self.model_performance: Dict[str, ModelPerformance] = {
            'oracle': ModelPerformance('oracle', 0.0, 0.0, 0.0, 0.0, 0.0, datetime.now()),
            'hunter': ModelPerformance('hunter', 0.0, 0.0, 0.0, 0.0, 0.0, datetime.now()),
            'network': ModelPerformance('network', 0.0, 0.0, 0.0, 0.0, 0.0, datetime.now())
        }
        
        # Consensus configuration
        self.consensus_config = {
            'min_models_agree': 2,  # At least 2 models must agree
            'min_confidence_threshold': 0.6,
            'max_risk_threshold': 0.7,
            'model_weights': {
                'oracle': 0.4,
                'hunter': 0.35,
                'network': 0.25
            }
        }
        
        # Prediction history
        self.prediction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Integration points
        self.battle_pattern_intelligence = None
        self.nightly_evolution_system = None
        self.swarm_decision_engine = None
        
        self.logger.info("✅ Prediction Engine initialized")
    
    async def initialize_hunter_ants(self, wallet_ids: List[str]):
        """Initialize Hunter Ant agents for each wallet"""
        
        try:
            for wallet_id in wallet_ids:
                if wallet_id not in self.hunter_ants:
                    model_path = f"models/hunter_ant_{wallet_id}.pth"
                    self.hunter_ants[wallet_id] = HunterAnt(wallet_id, model_path)
                    self.logger.info(f"✅ Initialized Hunter Ant for wallet {wallet_id}")
            
            self.logger.info(f"✅ Initialized {len(self.hunter_ants)} Hunter Ant agents")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Hunter Ants: {e}")
    
    async def predict(self, token_address: str, market_data: Dict[str, Any],
                     wallet_id: Optional[str] = None, 
                     prediction_horizon: int = 15) -> UnifiedPrediction:
        """Generate unified prediction using all three ML architectures"""
        
        try:
            # Generate predictions from each model
            oracle_prediction = await self.oracle_ant.predict(
                token_address, market_data, prediction_horizon
            )
            
            # Get Hunter Ant prediction if wallet_id is provided
            hunter_prediction = None
            if wallet_id and wallet_id in self.hunter_ants:
                hunter_action, hunter_value = await self.hunter_ants[wallet_id].act(market_data)
                hunter_prediction = {
                    'action': hunter_action,
                    'confidence': 0.7,  # Default confidence
                    'value': hunter_value,
                    'position_recommendation': self._interpret_hunter_action(hunter_action)
                }
            else:
                # Use ensemble of all Hunter Ants
                hunter_prediction = await self._get_ensemble_hunter_prediction(market_data)
            
            network_prediction = await self.network_ant.predict_network_intelligence(token_address)
            
            # Combine predictions into unified result
            unified_prediction = await self._combine_predictions(
                token_address, oracle_prediction, hunter_prediction, network_prediction,
                market_data, prediction_horizon
            )
            
            # Store prediction history
            self.prediction_history[token_address].append(unified_prediction)
            
            return unified_prediction
            
        except Exception as e:
            self.logger.error(f"Unified prediction failed for {token_address}: {e}")
            return self._fallback_prediction(token_address, market_data, prediction_horizon)
    
    async def _get_ensemble_hunter_prediction(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get ensemble prediction from all Hunter Ant agents"""
        
        try:
            if not self.hunter_ants:
                return {
                    'action': ActionType.HOLD.value,
                    'confidence': 0.5,
                    'value': 0.0,
                    'position_recommendation': 'HOLD'
                }
            
            # Get predictions from all Hunter Ants
            actions = []
            values = []
            
            for hunter_ant in self.hunter_ants.values():
                action, value = await hunter_ant.act(market_data)
                actions.append(action)
                values.append(value)
            
            # Ensemble decision (majority vote for action, average for value)
            action_counts = defaultdict(int)
            for action in actions:
                action_counts[action] += 1
            
            ensemble_action = max(action_counts.items(), key=lambda x: x[1])[0]
            ensemble_value = np.mean(values)
            
            # Calculate confidence based on agreement
            agreement_ratio = action_counts[ensemble_action] / len(actions)
            confidence = 0.5 + agreement_ratio * 0.5  # 0.5 to 1.0
            
            return {
                'action': ensemble_action,
                'confidence': confidence,
                'value': ensemble_value,
                'position_recommendation': self._interpret_hunter_action(ensemble_action)
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble Hunter prediction failed: {e}")
            return {
                'action': ActionType.HOLD.value,
                'confidence': 0.5,
                'value': 0.0,
                'position_recommendation': 'HOLD'
            }
    
    async def _combine_predictions(self, token_address: str,
                                 oracle_prediction: OraclePrediction,
                                 hunter_prediction: Dict[str, Any],
                                 network_prediction: NetworkPrediction,
                                 market_data: Dict[str, Any],
                                 prediction_horizon: int) -> UnifiedPrediction:
        """Combine predictions from all three models"""
        
        try:
            # Calculate consensus score
            consensus_score = self._calculate_consensus_score(
                oracle_prediction, hunter_prediction, network_prediction
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                oracle_prediction, hunter_prediction, network_prediction
            )
            
            # Assess risks
            risk_assessment = self._assess_risks(
                oracle_prediction, hunter_prediction, network_prediction, market_data
            )
            
            # Generate trading recommendation
            trading_recommendation = self._generate_trading_recommendation(
                oracle_prediction, hunter_prediction, network_prediction,
                consensus_score, risk_assessment
            )
            
            # Calculate position size recommendation
            position_size = self._calculate_position_size(
                consensus_score, overall_confidence, risk_assessment
            )
            
            # Determine model weights based on recent performance
            model_weights = self._calculate_model_weights()
            
            return UnifiedPrediction(
                token_address=token_address,
                
                # Oracle Ant predictions
                oracle_price_prediction=oracle_prediction.predicted_price,
                oracle_volume_prediction=oracle_prediction.predicted_volume,
                oracle_holders_prediction=oracle_prediction.predicted_holders,
                oracle_confidence=oracle_prediction.price_confidence,
                oracle_attention_weights=oracle_prediction.attention_weights,
                
                # Hunter Ant predictions
                hunter_action=hunter_prediction['action'],
                hunter_action_confidence=hunter_prediction['confidence'],
                hunter_value_estimate=hunter_prediction['value'],
                hunter_position_recommendation=hunter_prediction['position_recommendation'],
                
                # Network Ant predictions
                network_contagion_score=network_prediction.contagion_score,
                network_smart_money_probability=network_prediction.smart_money_probability,
                network_manipulation_risk=network_prediction.manipulation_risk,
                network_wallet_classifications=network_prediction.wallet_classifications,
                network_influence_network=network_prediction.influence_network,
                
                # Consensus metrics
                overall_confidence=overall_confidence,
                consensus_score=consensus_score,
                risk_assessment=risk_assessment,
                trading_recommendation=trading_recommendation,
                position_size_recommendation=position_size,
                
                # Metadata
                prediction_horizon=prediction_horizon,
                timestamp=datetime.now(),
                model_weights=model_weights
            )
            
        except Exception as e:
            self.logger.error(f"Failed to combine predictions: {e}")
            raise
    
    def _calculate_consensus_score(self, oracle_prediction: OraclePrediction,
                                 hunter_prediction: Dict[str, Any],
                                 network_prediction: NetworkPrediction) -> float:
        """Calculate consensus score between models"""
        
        try:
            # Oracle Ant signal (price direction)
            current_price = 1.0  # Normalized
            oracle_direction = 1 if oracle_prediction.predicted_price > current_price else -1
            
            # Hunter Ant signal (action direction)
            hunter_action = hunter_prediction['action']
            if hunter_action in [ActionType.BUY_25.value, ActionType.BUY_50.value, ActionType.BUY_100.value]:
                hunter_direction = 1
            elif hunter_action in [ActionType.SELL_25.value, ActionType.SELL_50.value, ActionType.SELL_100.value]:
                hunter_direction = -1
            else:
                hunter_direction = 0
            
            # Network Ant signal (contagion and smart money)
            network_signal = 0
            if network_prediction.contagion_score > 0.6:
                network_signal = 1
            elif network_prediction.contagion_score < 0.4:
                network_signal = -1
            
            # Calculate agreement
            signals = [oracle_direction, hunter_direction, network_signal]
            positive_signals = sum(1 for s in signals if s > 0)
            negative_signals = sum(1 for s in signals if s < 0)
            
            if positive_signals > negative_signals:
                consensus_score = positive_signals / len(signals)
            elif negative_signals > positive_signals:
                consensus_score = negative_signals / len(signals)
            else:
                consensus_score = 0.5  # Neutral
            
            return consensus_score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate consensus score: {e}")
            return 0.5
    
    def _calculate_overall_confidence(self, oracle_prediction: OraclePrediction,
                                    hunter_prediction: Dict[str, Any],
                                    network_prediction: NetworkPrediction) -> float:
        """Calculate overall confidence weighted by model performance"""
        
        try:
            weights = self.consensus_config['model_weights']
            
            oracle_conf = oracle_prediction.price_confidence * weights['oracle']
            hunter_conf = hunter_prediction['confidence'] * weights['hunter']
            network_conf = network_prediction.confidence * weights['network']
            
            overall_confidence = oracle_conf + hunter_conf + network_conf
            
            return min(1.0, overall_confidence)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate overall confidence: {e}")
            return 0.5
    
    def _assess_risks(self, oracle_prediction: OraclePrediction,
                     hunter_prediction: Dict[str, Any],
                     network_prediction: NetworkPrediction,
                     market_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess various risk factors"""
        
        try:
            risks = {}
            
            # Price volatility risk
            price_volatility = market_data.get('price_volatility_24h', 0.1)
            risks['volatility_risk'] = min(1.0, price_volatility * 10)
            
            # Liquidity risk
            liquidity_ratio = market_data.get('liquidity_ratio', 1.0)
            risks['liquidity_risk'] = max(0.0, 1.0 - liquidity_ratio)
            
            # Manipulation risk (from Network Ant)
            risks['manipulation_risk'] = network_prediction.manipulation_risk
            
            # Prediction uncertainty risk
            oracle_uncertainty = 1.0 - oracle_prediction.price_confidence
            risks['prediction_uncertainty'] = oracle_uncertainty
            
            # Market timing risk
            if hunter_prediction['action'] == ActionType.HOLD.value:
                risks['timing_risk'] = 0.3  # Moderate risk of missing opportunity
            else:
                risks['timing_risk'] = 0.1  # Lower risk when taking action
            
            # Overall risk score
            risks['overall_risk'] = np.mean(list(risks.values()))
            
            return risks
            
        except Exception as e:
            self.logger.error(f"Failed to assess risks: {e}")
            return {'overall_risk': 0.5}
    
    def _generate_trading_recommendation(self, oracle_prediction: OraclePrediction,
                                       hunter_prediction: Dict[str, Any],
                                       network_prediction: NetworkPrediction,
                                       consensus_score: float,
                                       risk_assessment: Dict[str, float]) -> str:
        """Generate trading recommendation based on all factors"""
        
        try:
            # Check if consensus is strong enough
            if consensus_score < 0.6:
                return "HOLD"
            
            # Check if risk is too high
            if risk_assessment.get('overall_risk', 0.5) > 0.7:
                return "AVOID"
            
            # Check for manipulation
            if network_prediction.manipulation_risk > 0.6:
                return "AVOID"
            
            # Determine direction
            if consensus_score > 0.7:
                return "STRONG_BUY"
            elif consensus_score > 0.6:
                return "BUY"
            elif consensus_score < 0.3:
                return "SELL"
            else:
                return "HOLD"
                
        except Exception as e:
            self.logger.error(f"Failed to generate trading recommendation: {e}")
            return "HOLD"
    
    def _calculate_position_size(self, consensus_score: float,
                               overall_confidence: float,
                               risk_assessment: Dict[str, float]) -> float:
        """Calculate recommended position size"""
        
        try:
            # Base position size on consensus and confidence
            base_size = consensus_score * overall_confidence
            
            # Adjust for risk
            risk_factor = 1.0 - risk_assessment.get('overall_risk', 0.5)
            
            # Final position size (0.0 to 1.0)
            position_size = base_size * risk_factor
            
            # Apply limits
            position_size = max(0.0, min(1.0, position_size))
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Failed to calculate position size: {e}")
            return 0.0
    
    def _calculate_model_weights(self) -> Dict[str, float]:
        """Calculate dynamic model weights based on recent performance"""
        
        try:
            weights = {}
            total_performance = 0
            
            for model_name, performance in self.model_performance.items():
                # Use profit contribution as weight factor
                weight_factor = max(0.1, performance.profit_contribution)
                weights[model_name] = weight_factor
                total_performance += weight_factor
            
            # Normalize weights
            if total_performance > 0:
                for model_name in weights:
                    weights[model_name] /= total_performance
            else:
                # Equal weights if no performance data
                for model_name in weights:
                    weights[model_name] = 1.0 / len(weights)
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Failed to calculate model weights: {e}")
            return self.consensus_config['model_weights']
    
    def _interpret_hunter_action(self, action: int) -> str:
        """Interpret Hunter Ant action into human-readable recommendation"""
        
        action_map = {
            ActionType.HOLD.value: "HOLD",
            ActionType.BUY_25.value: "BUY_25%",
            ActionType.BUY_50.value: "BUY_50%",
            ActionType.BUY_100.value: "BUY_100%",
            ActionType.SELL_25.value: "SELL_25%",
            ActionType.SELL_50.value: "SELL_50%",
            ActionType.SELL_100.value: "SELL_100%"
        }
        
        return action_map.get(action, "HOLD")
    
    def _fallback_prediction(self, token_address: str, market_data: Dict[str, Any],
                           prediction_horizon: int) -> UnifiedPrediction:
        """Generate fallback prediction when models fail"""
        
        return UnifiedPrediction(
            token_address=token_address,
            oracle_price_prediction=market_data.get('current_price', 0),
            oracle_volume_prediction=market_data.get('current_volume', 0),
            oracle_holders_prediction=market_data.get('holder_count', 0),
            oracle_confidence=0.1,
            oracle_attention_weights={},
            hunter_action=ActionType.HOLD.value,
            hunter_action_confidence=0.1,
            hunter_value_estimate=0.0,
            hunter_position_recommendation="HOLD",
            network_contagion_score=0.0,
            network_smart_money_probability=0.0,
            network_manipulation_risk=0.5,
            network_wallet_classifications={},
            network_influence_network={},
            overall_confidence=0.1,
            consensus_score=0.5,
            risk_assessment={'overall_risk': 0.5},
            trading_recommendation="HOLD",
            position_size_recommendation=0.0,
            prediction_horizon=prediction_horizon,
            timestamp=datetime.now(),
            model_weights=self.consensus_config['model_weights']
        )
    
    async def update_performance(self, token_address: str, prediction: UnifiedPrediction,
                               actual_outcome: Dict[str, Any]):
        """Update model performance based on actual outcomes"""
        
        try:
            # Calculate prediction accuracy
            predicted_price = prediction.oracle_price_prediction
            actual_price = actual_outcome.get('price', 0)
            
            if actual_price > 0:
                price_error = abs(predicted_price - actual_price) / actual_price
                
                # Update Oracle Ant performance
                oracle_perf = self.model_performance['oracle']
                oracle_perf.accuracy = 1.0 - min(1.0, price_error)
                oracle_perf.last_updated = datetime.now()
            
            # Update Hunter Ant performance
            if prediction.hunter_position_recommendation != "HOLD":
                hunter_perf = self.model_performance['hunter']
                # Calculate profit contribution (simplified)
                profit_contribution = actual_outcome.get('profit_loss', 0)
                hunter_perf.profit_contribution = max(0.0, profit_contribution)
                hunter_perf.last_updated = datetime.now()
            
            # Update Network Ant performance
            network_perf = self.model_performance['network']
            if prediction.network_manipulation_risk > 0.7:
                # Check if rug pull actually happened
                if actual_outcome.get('rug_pull_detected', False):
                    network_perf.accuracy += 0.1
                else:
                    network_perf.accuracy -= 0.05
            network_perf.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to update performance: {e}")
    
    async def integrate_with_battle_pattern(self, battle_pattern_intelligence):
        """Integrate with BattlePatternIntelligence"""
        
        try:
            self.battle_pattern_intelligence = battle_pattern_intelligence
            
            # Share wallet classifications
            if hasattr(battle_pattern_intelligence, 'monitored_wallets'):
                for address, signature in battle_pattern_intelligence.monitored_wallets.items():
                    wallet_data = {
                        'success_rate': signature.success_rate,
                        'profit_factor': signature.profit_factor,
                        'avg_position_size': signature.avg_position_size,
                        'trade_frequency': signature.trade_frequency,
                        'risk_score': signature.risk_score,
                        'win_rate': signature.win_rate,
                        'avg_profit_per_trade': signature.avg_profit_per_trade,
                        'max_drawdown': signature.max_drawdown,
                        'behavior_type': signature.behavior_type.value
                    }
                    await self.network_ant.add_wallet(address, wallet_data)
            
            self.logger.info("✅ Integrated with BattlePatternIntelligence")
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with BattlePatternIntelligence: {e}")
    
    async def integrate_with_nightly_evolution(self, nightly_evolution_system):
        """Integrate with NightlyEvolutionSystem"""
        
        try:
            self.nightly_evolution_system = nightly_evolution_system
            
            # Share Hunter Ant performance for genetic evolution
            for wallet_id, hunter_ant in self.hunter_ants.items():
                performance = hunter_ant.get_performance_summary()
                
                # Update evolution system with Hunter Ant performance
                if hasattr(nightly_evolution_system, 'update_wallet_performance'):
                    await nightly_evolution_system.update_wallet_performance(
                        wallet_id, performance
                    )
            
            self.logger.info("✅ Integrated with NightlyEvolutionSystem")
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with NightlyEvolutionSystem: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        
        return {
            'model_performance': {
                name: {
                    'accuracy': perf.accuracy,
                    'profit_contribution': perf.profit_contribution,
                    'last_updated': perf.last_updated.isoformat()
                }
                for name, perf in self.model_performance.items()
            },
            'hunter_ants': {
                wallet_id: hunter_ant.get_performance_summary()
                for wallet_id, hunter_ant in self.hunter_ants.items()
            },
            'oracle_ant': self.oracle_ant.get_performance_summary(),
            'network_ant': self.network_ant.get_performance_summary(),
            'total_predictions': sum(len(history) for history in self.prediction_history.values()),
            'active_tokens': len(self.prediction_history)
        }
    
    async def save_all_models(self, base_path: str = "models"):
        """Save all model checkpoints"""
        
        try:
            os.makedirs(base_path, exist_ok=True)
            
            # Save Oracle Ant
            self.oracle_ant.save_model(f"{base_path}/oracle_ant.pth")
            
            # Save Network Ant
            self.network_ant.save_model(f"{base_path}/network_ant.pth")
            
            # Save Hunter Ants
            for wallet_id, hunter_ant in self.hunter_ants.items():
                hunter_ant.save_model(f"{base_path}/hunter_ant_{wallet_id}.pth")
            
            self.logger.info("✅ Saved all model checkpoints")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    async def load_all_models(self, base_path: str = "models"):
        """Load all model checkpoints"""
        
        try:
            # Load Oracle Ant
            if os.path.exists(f"{base_path}/oracle_ant.pth"):
                self.oracle_ant.load_model(f"{base_path}/oracle_ant.pth")
            
            # Load Network Ant
            if os.path.exists(f"{base_path}/network_ant.pth"):
                self.network_ant.load_model(f"{base_path}/network_ant.pth")
            
            # Load Hunter Ants
            for wallet_id in self.hunter_ants:
                model_path = f"{base_path}/hunter_ant_{wallet_id}.pth"
                if os.path.exists(model_path):
                    self.hunter_ants[wallet_id].load_model(model_path)
            
            self.logger.info("✅ Loaded all model checkpoints")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}") 