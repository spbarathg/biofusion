"""
PREDICTION ENGINE - UNIFIED ML INTELLIGENCE CORE
===============================================

Main prediction engine that integrates all three ML architectures:
1. Oracle Ant - Time-series Transformer
2. Hunter Ant - Reinforcement Learning Agent  
3. Network Ant - Graph Neural Network

Enhanced with ProcessPoolManager for CPU-intensive inference tasks.
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
from worker_ant_v1.core.process_pool_manager import get_process_pool_manager, ProcessTaskResult
from .oracle_ant import OracleAntPredictor, OraclePrediction
from .hunter_ant import HunterAnt, ActionType
from .marl_hunter_ant import MADDPGSwarm, MultiAgentMarketState, create_marl_hunter_swarm
from .network_ant import NetworkAntPredictor, NetworkPrediction, WalletType
from .dynamic_network_ant import DynamicNetworkAntPredictor, create_dynamic_network_ant
import torch

@dataclass
class ModelPerformance:
    """Model performance tracking"""
    model_name: str
    accuracy: float
    latency_ms: float
    throughput: float
    error_rate: float
    profit_contribution: float
    last_updated: datetime


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
    
    # Performance metadata
    oracle_latency_ms: float = 0.0
    hunter_latency_ms: float = 0.0
    network_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    parallel_execution: bool = True
    
    # Metadata
    prediction_horizon: int
    timestamp: datetime
    model_weights: Dict[str, float]  # Weight given to each model


class PredictionEngine:
    """Unified prediction engine integrating all ML architectures with process pools"""
    
    def __init__(self, model_paths: Optional[Dict[str, str]] = None):
        self.logger = setup_logger("PredictionEngine")
        
        # Process pool manager for CPU-intensive inference
        self.process_pool_manager = None
        
        # Fallback ML architectures (for non-process pool execution)
        self.oracle_ant = OracleAntPredictor(
            model_path=model_paths.get('oracle') if model_paths else None
        )
        
        # Initialize Hunter Ant for each wallet (will be populated dynamically)
        self.hunter_ants: Dict[str, HunterAnt] = {}
        
        # MARL Hunter Ant Swarm (new multi-agent system)
        self.marl_hunter_swarm: Optional[MADDPGSwarm] = None
        self.use_marl = True  # Flag to enable multi-agent system
        
        self.network_ant = NetworkAntPredictor(
            model_path=model_paths.get('network') if model_paths else None
        )
        
        # Dynamic Network Ant (enhanced version with temporal dynamics)
        self.dynamic_network_ant: Optional[DynamicNetworkAntPredictor] = None
        self.use_dynamic_network = True  # Flag to enable enhanced network analysis
        
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
        
        # Process pool configuration
        self.use_process_pools = True
        self.parallel_inference = True
        self.inference_timeout = 30.0
        
        # Prediction history
        self.prediction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Integration points
        self.battle_pattern_intelligence = None
        self.nightly_evolution_system = None
        self.swarm_decision_engine = None
        
        # System state
        self.initialized = False
        
        self.logger.info("✅ Prediction Engine initialized")

    async def initialize(self) -> bool:
        """Initialize prediction engine and process pools"""
        try:
            self.logger.info("🚀 Initializing Prediction Engine...")
            
            # Initialize process pool manager
            if self.use_process_pools:
                self.process_pool_manager = await get_process_pool_manager()
                self.logger.info("✅ Process pool manager integrated")
            
            # Initialize MARL Hunter Ant Swarm
            if self.use_marl:
                self.marl_hunter_swarm = await create_marl_hunter_swarm(num_agents=10)
                self.logger.info("✅ MARL Hunter Ant Swarm initialized")
            
            # Initialize Dynamic Network Ant
            if self.use_dynamic_network:
                self.dynamic_network_ant = await create_dynamic_network_ant()
                self.logger.info("✅ Dynamic Network Ant initialized")
            
            self.initialized = True
            self.logger.info("✅ Prediction Engine fully initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Prediction Engine: {e}")
            return False
    
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
        """Generate unified prediction using all three ML architectures with process pools"""
        
        start_time = datetime.utcnow()
        
        try:
            if not self.initialized:
                await self.initialize()
            
            # Choose execution strategy
            if self.use_process_pools and self.process_pool_manager:
                if self.parallel_inference:
                    prediction = await self._predict_parallel_process_pools(
                        token_address, market_data, wallet_id, prediction_horizon
                    )
                else:
                    prediction = await self._predict_sequential_process_pools(
                        token_address, market_data, wallet_id, prediction_horizon
                    )
            else:
                # Fallback to direct inference
                prediction = await self._predict_direct(
                    token_address, market_data, wallet_id, prediction_horizon
                )
            
            # Calculate total execution time
            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            prediction.total_latency_ms = total_time
            
            # Store prediction history
            self.prediction_history[token_address].append(prediction)
            
            # Update performance metrics
            await self._update_latency_metrics(prediction)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Unified prediction failed for {token_address}: {e}")
            return self._fallback_prediction(token_address, market_data, prediction_horizon)

    async def _predict_parallel_process_pools(self, token_address: str, market_data: Dict[str, Any],
                                            wallet_id: Optional[str], prediction_horizon: int) -> UnifiedPrediction:
        """Generate predictions using parallel process pool execution"""
        
        try:
            # Submit all inference tasks concurrently
            task_ids = []
            
            # Oracle Ant task
            oracle_task_id = await self.process_pool_manager.submit_oracle_task(
                token_address, market_data, prediction_horizon
            )
            task_ids.append(('oracle', oracle_task_id))
            
            # Hunter Ant task
            if wallet_id:
                hunter_task_id = await self.process_pool_manager.submit_hunter_task(
                    wallet_id, market_data
                )
                task_ids.append(('hunter', hunter_task_id))
            
            # Network Ant task
            network_task_id = await self.process_pool_manager.submit_network_task(
                token_address
            )
            task_ids.append(('network', network_task_id))
            
            # Collect results
            results = {}
            latencies = {}
            
            for model_type, task_id in task_ids:
                try:
                    result = await self.process_pool_manager.get_task_result(
                        task_id, timeout=self.inference_timeout
                    )
                    
                    if result.success:
                        results[model_type] = result.result
                        latencies[model_type] = result.execution_time * 1000  # Convert to ms
                    else:
                        self.logger.warning(f"⚠️ {model_type} inference failed: {result.error}")
                        results[model_type] = None
                        latencies[model_type] = result.execution_time * 1000
                        
                except Exception as e:
                    self.logger.error(f"❌ Failed to get {model_type} result: {e}")
                    results[model_type] = None
                    latencies[model_type] = self.inference_timeout * 1000
            
            # Handle missing Hunter Ant prediction
            if 'hunter' not in results and wallet_id:
                # Use ensemble if individual wallet Hunter Ant failed
                results['hunter'] = await self._get_ensemble_hunter_prediction(market_data)
                latencies['hunter'] = 0.0  # Local execution
            
            # Generate unified prediction from results
            return await self._combine_process_pool_predictions(
                token_address, results, latencies, market_data, prediction_horizon
            )
            
        except Exception as e:
            self.logger.error(f"❌ Parallel process pool prediction failed: {e}")
            # Fallback to direct inference
            return await self._predict_direct(token_address, market_data, wallet_id, prediction_horizon)

    async def _predict_sequential_process_pools(self, token_address: str, market_data: Dict[str, Any],
                                              wallet_id: Optional[str], prediction_horizon: int) -> UnifiedPrediction:
        """Generate predictions using sequential process pool execution"""
        
        try:
            results = {}
            latencies = {}
            
            # Oracle Ant prediction
            oracle_task_id = await self.process_pool_manager.submit_oracle_task(
                token_address, market_data, prediction_horizon
            )
            oracle_result = await self.process_pool_manager.get_task_result(oracle_task_id)
            
            if oracle_result.success:
                results['oracle'] = oracle_result.result
                latencies['oracle'] = oracle_result.execution_time * 1000
            else:
                self.logger.warning(f"⚠️ Oracle Ant inference failed: {oracle_result.error}")
                results['oracle'] = None
                latencies['oracle'] = oracle_result.execution_time * 1000
            
            # Hunter Ant prediction
            if wallet_id:
                hunter_task_id = await self.process_pool_manager.submit_hunter_task(
                    wallet_id, market_data
                )
                hunter_result = await self.process_pool_manager.get_task_result(hunter_task_id)
                
                if hunter_result.success:
                    results['hunter'] = hunter_result.result
                    latencies['hunter'] = hunter_result.execution_time * 1000
                else:
                    self.logger.warning(f"⚠️ Hunter Ant inference failed: {hunter_result.error}")
                    results['hunter'] = await self._get_ensemble_hunter_prediction(market_data)
                    latencies['hunter'] = 0.0
            else:
                results['hunter'] = await self._get_ensemble_hunter_prediction(market_data)
                latencies['hunter'] = 0.0
            
            # Network Ant prediction
            network_task_id = await self.process_pool_manager.submit_network_task(token_address)
            network_result = await self.process_pool_manager.get_task_result(network_task_id)
            
            if network_result.success:
                results['network'] = network_result.result
                latencies['network'] = network_result.execution_time * 1000
            else:
                self.logger.warning(f"⚠️ Network Ant inference failed: {network_result.error}")
                results['network'] = None
                latencies['network'] = network_result.execution_time * 1000
            
            # Generate unified prediction from results
            return await self._combine_process_pool_predictions(
                token_address, results, latencies, market_data, prediction_horizon
            )
            
        except Exception as e:
            self.logger.error(f"❌ Sequential process pool prediction failed: {e}")
            # Fallback to direct inference
            return await self._predict_direct(token_address, market_data, wallet_id, prediction_horizon)

    async def _predict_direct(self, token_address: str, market_data: Dict[str, Any],
                            wallet_id: Optional[str], prediction_horizon: int) -> UnifiedPrediction:
        """Generate predictions using direct inference (fallback method)"""
        
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
            
            # Get Network Ant prediction (use dynamic version if available)
            if self.use_dynamic_network and self.dynamic_network_ant:
                network_prediction = await self.dynamic_network_ant.predict_network_intelligence(token_address)
            else:
                network_prediction = await self.network_ant.predict_network_intelligence(token_address)
            
            # Combine predictions into unified result
            unified_prediction = await self._combine_predictions(
                token_address, oracle_prediction, hunter_prediction, network_prediction,
                market_data, prediction_horizon
            )
            
            return unified_prediction
            
        except Exception as e:
            self.logger.error(f"❌ Direct prediction failed: {e}")
            raise

    async def _combine_process_pool_predictions(self, token_address: str, 
                                              results: Dict[str, Any],
                                              latencies: Dict[str, float],
                                              market_data: Dict[str, Any],
                                              prediction_horizon: int) -> UnifiedPrediction:
        """Combine predictions from process pool results"""
        
        try:
            # Extract Oracle Ant results
            oracle_result = results.get('oracle')
            if oracle_result:
                oracle_price = oracle_result['predicted_price']
                oracle_volume = oracle_result['predicted_volume']
                oracle_holders = oracle_result['predicted_holders']
                oracle_confidence = oracle_result['price_confidence']
                oracle_attention = oracle_result['attention_weights']
            else:
                # Fallback values
                oracle_price = market_data.get('price', 1.0)
                oracle_volume = market_data.get('volume', 0.0)
                oracle_holders = market_data.get('holders', 0)
                oracle_confidence = 0.1
                oracle_attention = {}
            
            # Extract Hunter Ant results
            hunter_result = results.get('hunter')
            if hunter_result:
                hunter_action = hunter_result['action']
                hunter_value = hunter_result['value']
                hunter_confidence = 0.7
                hunter_position = self._interpret_hunter_action(hunter_action)
            else:
                hunter_action = ActionType.HOLD.value
                hunter_value = 0.0
                hunter_confidence = 0.1
                hunter_position = "HOLD"
            
            # Extract Network Ant results
            network_result = results.get('network')
            if network_result:
                network_contagion = network_result['contagion_score']
                network_smart_money = network_result['smart_money_probability']
                network_manipulation = network_result['manipulation_risk']
                network_classifications = {
                    k: WalletType(v) for k, v in network_result['wallet_classifications'].items()
                }
                network_influence = network_result['influence_network']
            else:
                network_contagion = 0.5
                network_smart_money = 0.5
                network_manipulation = 0.3
                network_classifications = {}
                network_influence = {}
            
            # Calculate consensus and recommendations
            overall_confidence, consensus_score, risk_assessment, trading_recommendation, position_size = \
                await self._calculate_consensus_from_results(
                    oracle_price, hunter_action, network_contagion, market_data
                )
            
            # Create unified prediction
            return UnifiedPrediction(
                token_address=token_address,
                
                # Oracle Ant predictions
                oracle_price_prediction=oracle_price,
                oracle_volume_prediction=oracle_volume,
                oracle_holders_prediction=oracle_holders,
                oracle_confidence=oracle_confidence,
                oracle_attention_weights=oracle_attention,
                
                # Hunter Ant predictions
                hunter_action=hunter_action,
                hunter_action_confidence=hunter_confidence,
                hunter_value_estimate=hunter_value,
                hunter_position_recommendation=hunter_position,
                
                # Network Ant predictions
                network_contagion_score=network_contagion,
                network_smart_money_probability=network_smart_money,
                network_manipulation_risk=network_manipulation,
                network_wallet_classifications=network_classifications,
                network_influence_network=network_influence,
                
                # Consensus metrics
                overall_confidence=overall_confidence,
                consensus_score=consensus_score,
                risk_assessment=risk_assessment,
                trading_recommendation=trading_recommendation,
                position_size_recommendation=position_size,
                
                # Performance metadata
                oracle_latency_ms=latencies.get('oracle', 0.0),
                hunter_latency_ms=latencies.get('hunter', 0.0),
                network_latency_ms=latencies.get('network', 0.0),
                parallel_execution=True,
                
                # Metadata
                prediction_horizon=prediction_horizon,
                timestamp=datetime.now(),
                model_weights=self.consensus_config['model_weights']
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to combine process pool predictions: {e}")
            raise

    async def _get_ensemble_hunter_prediction(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get ensemble prediction from Hunter Ant agents (MARL or legacy)"""
        
        try:
            if self.use_marl and self.marl_hunter_swarm:
                return await self._get_marl_hunter_prediction(market_data)
            else:
                return await self._get_legacy_hunter_prediction(market_data)
                
        except Exception as e:
            self.logger.error(f"❌ Ensemble Hunter prediction failed: {e}")
            return {
                'action': ActionType.HOLD.value,
                'confidence': 0.5,
                'value': 0.0,
                'position_recommendation': 'HOLD'
            }

    async def _get_marl_hunter_prediction(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from MARL Hunter Ant swarm"""
        
        try:
            # Create multi-agent market state
            ma_state = self._create_multi_agent_state(market_data)
            
            # Get actions from all agents in the swarm
            agent_actions = []
            agent_values = []
            agent_communications = []
            
            for agent_id in range(self.marl_hunter_swarm.num_agents):
                action, comm_signal = await self.marl_hunter_swarm.act(
                    agent_id, ma_state, training=False
                )
                
                # Estimate value (simplified - could be enhanced)
                value = float(torch.mean(comm_signal).item()) if comm_signal is not None else 0.0
                
                agent_actions.append(action)
                agent_values.append(value)
                agent_communications.append(comm_signal)
            
            # Aggregate swarm decision
            action_counts = {}
            for action in agent_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Majority vote for final action
            swarm_action = max(action_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate confidence based on agreement
            agreement_ratio = action_counts[swarm_action] / len(agent_actions)
            confidence = 0.3 + agreement_ratio * 0.7  # 0.3 to 1.0
            
            # Average value estimate
            avg_value = np.mean(agent_values)
            
            # Interpret action
            position_recommendation = self._interpret_hunter_action(swarm_action)
            
            return {
                'action': swarm_action,
                'confidence': confidence,
                'value': avg_value,
                'position_recommendation': position_recommendation,
                'swarm_agreement': agreement_ratio,
                'agent_actions': agent_actions,
                'communication_signals': agent_communications
            }
            
        except Exception as e:
            self.logger.error(f"❌ MARL Hunter prediction failed: {e}")
            return await self._get_legacy_hunter_prediction(market_data)

    async def _get_legacy_hunter_prediction(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from legacy single-agent Hunter Ants"""
        
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
            self.logger.error(f"❌ Legacy Hunter prediction failed: {e}")
            return {
                'action': ActionType.HOLD.value,
                'confidence': 0.5,
                'value': 0.0,
                'position_recommendation': 'HOLD'
            }

    def _create_multi_agent_state(self, market_data: Dict[str, Any]) -> MultiAgentMarketState:
        """Create MultiAgentMarketState from market data"""
        
        try:
            # Extract basic market features
            current_price = market_data.get('current_price', 1.0)
            
            # Calculate price changes (simplified)
            price_history = market_data.get('price_history', [])
            if len(price_history) >= 15:
                price_change_1m = (current_price - price_history[-1].get('price', current_price)) / current_price
                price_change_5m = (current_price - price_history[-5].get('price', current_price)) / current_price
                price_change_15m = (current_price - price_history[-15].get('price', current_price)) / current_price
            else:
                price_change_1m = price_change_5m = price_change_15m = 0.0
            
            # Calculate volatility
            if len(price_history) >= 20:
                recent_prices = [p.get('price', current_price) for p in price_history[-20:]]
                price_volatility = np.std(recent_prices) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0.1
            else:
                price_volatility = 0.1
            
            # Volume features
            current_volume = market_data.get('current_volume', 1000.0)
            volume_history = market_data.get('volume_history', [])
            if len(volume_history) >= 5:
                volume_change_1m = (current_volume - volume_history[-1].get('volume', current_volume)) / current_volume
                volume_change_5m = (current_volume - volume_history[-5].get('volume', current_volume)) / current_volume
                volume_ma = np.mean([v.get('volume', current_volume) for v in volume_history[-20:]])
                volume_ma_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
            else:
                volume_change_1m = volume_change_5m = 0.0
                volume_ma_ratio = 1.0
            
            # Technical indicators
            rsi = market_data.get('rsi', 50.0)
            macd = market_data.get('macd', 0.0)
            bollinger_position = market_data.get('bollinger_position', 0.5)
            moving_average_ratio = market_data.get('moving_average_ratio', 1.0)
            
            # Sentiment features
            sentiment_score = market_data.get('sentiment_score', 0.0)
            sentiment_change = market_data.get('sentiment_change', 0.0)
            
            # Position features (simplified - would be agent-specific in practice)
            current_position = market_data.get('current_position', 0.0)
            unrealized_pnl = market_data.get('unrealized_pnl', 0.0)
            position_age = market_data.get('position_age', 0.0)
            
            # Market context
            liquidity_ratio = market_data.get('liquidity_ratio', 1.0)
            holder_count_change = market_data.get('holder_count_change', 0.0)
            market_cap = market_data.get('market_cap', 1000000.0)
            
            # Multi-agent specific features
            swarm_total_position = market_data.get('swarm_total_position', 0.0)
            swarm_avg_confidence = market_data.get('swarm_avg_confidence', 0.5)
            agent_count_bullish = market_data.get('agent_count_bullish', 5)
            agent_count_bearish = market_data.get('agent_count_bearish', 5)
            coordination_signal = market_data.get('coordination_signal', 0.0)
            communication_vector = market_data.get('communication_vector', [0.0] * 10)
            
            return MultiAgentMarketState(
                current_price=current_price,
                price_change_1m=price_change_1m,
                price_change_5m=price_change_5m,
                price_change_15m=price_change_15m,
                price_volatility=price_volatility,
                current_volume=current_volume,
                volume_change_1m=volume_change_1m,
                volume_change_5m=volume_change_5m,
                volume_ma_ratio=volume_ma_ratio,
                rsi=rsi,
                macd=macd,
                bollinger_position=bollinger_position,
                moving_average_ratio=moving_average_ratio,
                sentiment_score=sentiment_score,
                sentiment_change=sentiment_change,
                current_position=current_position,
                unrealized_pnl=unrealized_pnl,
                position_age=position_age,
                liquidity_ratio=liquidity_ratio,
                holder_count_change=holder_count_change,
                market_cap=market_cap,
                swarm_total_position=swarm_total_position,
                swarm_avg_confidence=swarm_avg_confidence,
                agent_count_bullish=agent_count_bullish,
                agent_count_bearish=agent_count_bearish,
                coordination_signal=coordination_signal,
                communication_vector=communication_vector
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error creating multi-agent state: {e}")
            # Return default state
            return MultiAgentMarketState(
                current_price=1.0,
                price_change_1m=0.0,
                price_change_5m=0.0,
                price_change_15m=0.0,
                price_volatility=0.1,
                current_volume=1000.0,
                volume_change_1m=0.0,
                volume_change_5m=0.0,
                volume_ma_ratio=1.0,
                rsi=50.0,
                macd=0.0,
                bollinger_position=0.5,
                moving_average_ratio=1.0,
                sentiment_score=0.0,
                sentiment_change=0.0,
                current_position=0.0,
                unrealized_pnl=0.0,
                position_age=0.0,
                liquidity_ratio=1.0,
                holder_count_change=0.0,
                market_cap=1000000.0,
                swarm_total_position=0.0,
                swarm_avg_confidence=0.5,
                agent_count_bullish=5,
                agent_count_bearish=5,
                coordination_signal=0.0,
                communication_vector=[0.0] * 10
            )

    async def _combine_predictions(self, token_address: str,
                                 oracle_prediction: OraclePrediction,
                                 hunter_prediction: Dict[str, Any],
                                 network_prediction: NetworkPrediction,
                                 market_data: Dict[str, Any],
                                 prediction_horizon: int) -> UnifiedPrediction:
        """Combine predictions into unified result (legacy method)"""
        
        try:
            # Calculate consensus score
            consensus_score = self._calculate_consensus_score(
                oracle_prediction, hunter_prediction, network_prediction
            )
            
            # Calculate overall confidence
            confidences = [
                oracle_prediction.price_confidence,
                hunter_prediction['confidence'],
                network_prediction.confidence
            ]
            overall_confidence = np.mean(confidences)
            
            # Risk assessment
            risk_assessment = {
                'price_volatility': 1.0 - oracle_prediction.price_confidence,
                'manipulation_risk': network_prediction.manipulation_risk,
                'market_uncertainty': 1.0 - overall_confidence
            }
            
            # Trading recommendation
            trading_recommendation = self._generate_trading_recommendation(
                oracle_prediction, hunter_prediction, network_prediction, consensus_score
            )
            
            # Position size recommendation
            position_size = self._calculate_position_size(
                oracle_prediction, hunter_prediction, network_prediction, overall_confidence
            )
            
            # Model weights
            model_weights = self.consensus_config['model_weights'].copy()
            
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
                
                # Performance metadata
                parallel_execution=False,
                
                # Metadata
                prediction_horizon=prediction_horizon,
                timestamp=datetime.now(),
                model_weights=model_weights
            )
            
        except Exception as e:
            self.logger.error(f"Failed to combine predictions: {e}")
            raise

    def _interpret_hunter_action(self, action: int) -> str:
        """Interpret Hunter Ant action as position recommendation"""
        try:
            action_type = ActionType(action)
            
            if action_type in [ActionType.BUY_25, ActionType.BUY_50, ActionType.BUY_100]:
                return "BUY"
            elif action_type in [ActionType.SELL_25, ActionType.SELL_50, ActionType.SELL_100]:
                return "SELL"
            else:
                return "HOLD"
                
        except (ValueError, TypeError):
            return "HOLD"

    async def _calculate_consensus_from_results(self, oracle_price: float, hunter_action: int,
                                              network_contagion: float, market_data: Dict[str, Any]) -> Tuple[float, float, Dict[str, float], str, float]:
        """Calculate consensus metrics from individual results"""
        
        try:
            # Oracle signal (price direction)
            current_price = market_data.get('price', 1.0)
            oracle_direction = 1 if oracle_price > current_price else -1
            
            # Hunter signal (action direction)
            if hunter_action in [ActionType.BUY_25.value, ActionType.BUY_50.value, ActionType.BUY_100.value]:
                hunter_direction = 1
            elif hunter_action in [ActionType.SELL_25.value, ActionType.SELL_50.value, ActionType.SELL_100.value]:
                hunter_direction = -1
            else:
                hunter_direction = 0
            
            # Network signal
            network_signal = 1 if network_contagion > 0.6 else (-1 if network_contagion < 0.4 else 0)
            
            # Calculate consensus
            signals = [oracle_direction, hunter_direction, network_signal]
            positive_signals = sum(1 for s in signals if s > 0)
            negative_signals = sum(1 for s in signals if s < 0)
            
            if positive_signals > negative_signals:
                consensus_score = positive_signals / len(signals)
                trading_recommendation = "BUY"
            elif negative_signals > positive_signals:
                consensus_score = negative_signals / len(signals)
                trading_recommendation = "SELL"
            else:
                consensus_score = 0.5
                trading_recommendation = "HOLD"
            
            # Overall confidence
            overall_confidence = consensus_score
            
            # Risk assessment
            risk_assessment = {
                'consensus_risk': 1.0 - consensus_score,
                'model_disagreement': abs(positive_signals - negative_signals) / len(signals),
                'overall_risk': (1.0 - consensus_score) * 0.7 + (abs(positive_signals - negative_signals) / len(signals)) * 0.3
            }
            
            # Position size
            position_size = consensus_score * 0.5  # Max 50% allocation
            
            return overall_confidence, consensus_score, risk_assessment, trading_recommendation, position_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating consensus: {e}")
            return 0.5, 0.5, {'overall_risk': 0.5}, "HOLD", 0.1

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

    def _generate_trading_recommendation(self, oracle_prediction: OraclePrediction,
                                       hunter_prediction: Dict[str, Any],
                                       network_prediction: NetworkPrediction,
                                       consensus_score: float) -> str:
        """Generate trading recommendation"""
        
        try:
            # High manipulation risk overrides everything
            if network_prediction.manipulation_risk > 0.8:
                return "AVOID"
            
            # Strong consensus
            if consensus_score > 0.8:
                return "STRONG_BUY" if oracle_prediction.predicted_price > 1.0 else "STRONG_SELL"
            
            # Moderate consensus
            if consensus_score > 0.6:
                return "BUY" if oracle_prediction.predicted_price > 1.0 else "SELL"
            
            # Weak consensus or disagreement
            if consensus_score < 0.4:
                return "HOLD"
            
            # Default
            return "MONITOR"
            
        except Exception as e:
            self.logger.error(f"Failed to generate trading recommendation: {e}")
            return "HOLD"

    def _calculate_position_size(self, oracle_prediction: OraclePrediction,
                               hunter_prediction: Dict[str, Any],
                               network_prediction: NetworkPrediction,
                               overall_confidence: float) -> float:
        """Calculate position size recommendation"""
        
        try:
            # Base position size on confidence
            base_size = overall_confidence * 0.3  # Max 30% base allocation
            
            # Adjust for manipulation risk
            risk_adjustment = 1.0 - network_prediction.manipulation_risk
            
            # Adjust for price confidence
            price_adjustment = oracle_prediction.price_confidence
            
            # Final position size
            position_size = base_size * risk_adjustment * price_adjustment
            
            # Cap at 50% maximum
            return min(position_size, 0.5)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate position size: {e}")
            return 0.1

    async def _update_latency_metrics(self, prediction: UnifiedPrediction):
        """Update latency performance metrics"""
        try:
            # Update Oracle Ant latency
            if prediction.oracle_latency_ms > 0:
                oracle_perf = self.model_performance['oracle']
                oracle_perf.latency_ms = prediction.oracle_latency_ms
                oracle_perf.last_updated = datetime.now()
            
            # Update Hunter Ant latency
            if prediction.hunter_latency_ms > 0:
                hunter_perf = self.model_performance['hunter']
                hunter_perf.latency_ms = prediction.hunter_latency_ms
                hunter_perf.last_updated = datetime.now()
            
            # Update Network Ant latency
            if prediction.network_latency_ms > 0:
                network_perf = self.model_performance['network']
                network_perf.latency_ms = prediction.network_latency_ms
                network_perf.last_updated = datetime.now()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update latency metrics: {e}")

    def _fallback_prediction(self, token_address: str, market_data: Dict[str, Any], 
                           prediction_horizon: int) -> UnifiedPrediction:
        """Generate fallback prediction when all else fails"""
        
        current_price = market_data.get('price', 1.0)
        
        return UnifiedPrediction(
            token_address=token_address,
            
            # Oracle Ant predictions (conservative)
            oracle_price_prediction=current_price,
            oracle_volume_prediction=market_data.get('volume', 0.0),
            oracle_holders_prediction=market_data.get('holders', 0),
            oracle_confidence=0.1,
            oracle_attention_weights={},
            
            # Hunter Ant predictions (hold)
            hunter_action=ActionType.HOLD.value,
            hunter_action_confidence=0.1,
            hunter_value_estimate=0.0,
            hunter_position_recommendation="HOLD",
            
            # Network Ant predictions (neutral)
            network_contagion_score=0.5,
            network_smart_money_probability=0.5,
            network_manipulation_risk=0.5,
            network_wallet_classifications={},
            network_influence_network={},
            
            # Consensus metrics (conservative)
            overall_confidence=0.1,
            consensus_score=0.5,
            risk_assessment={'overall_risk': 0.9},
            trading_recommendation="HOLD",
            position_size_recommendation=0.0,
            
            # Metadata
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

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        performance_summary = {
            'model_performance': {
                name: {
                    'accuracy': perf.accuracy,
                    'latency_ms': perf.latency_ms,
                    'throughput': perf.throughput,
                    'error_rate': perf.error_rate,
                    'profit_contribution': perf.profit_contribution,
                    'last_updated': perf.last_updated.isoformat()
                }
                for name, perf in self.model_performance.items()
            },
            'process_pool_performance': None,
            'prediction_history_size': {
                token: len(history) for token, history in self.prediction_history.items()
            },
            'system_status': {
                'initialized': self.initialized,
                'use_process_pools': self.use_process_pools,
                'parallel_inference': self.parallel_inference
            }
        }
        
        # Add process pool performance if available
        if self.process_pool_manager:
            performance_summary['process_pool_performance'] = \
                self.process_pool_manager.get_performance_summary()
        
        # Add MARL performance if available
        if self.use_marl and self.marl_hunter_swarm:
            performance_summary['marl_hunter_performance'] = \
                self.marl_hunter_swarm.get_performance_summary()
        
        # Add Dynamic Network Ant performance if available
        if self.use_dynamic_network and self.dynamic_network_ant:
            performance_summary['dynamic_network_performance'] = \
                self.dynamic_network_ant.get_performance_summary()
        
        return performance_summary

    async def shutdown(self):
        """Shutdown prediction engine and process pools"""
        try:
            self.logger.info("🛑 Shutting down Prediction Engine...")
            
            # Clear prediction history
            self.prediction_history.clear()
            
            # Process pool manager will be shutdown globally
            self.process_pool_manager = None
            
            self.logger.info("✅ Prediction Engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}") 