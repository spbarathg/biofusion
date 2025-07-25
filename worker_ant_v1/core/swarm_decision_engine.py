"""
SWARM DECISION ENGINE - NARRATIVE-WEIGHTED OPPORTUNITY ANALYSIS
=============================================================

Central decision-making engine that analyzes trading opportunities
with narrative intelligence weighting and anti-fragile principles.

MATHEMATICAL ENHANCEMENT:
- Gradient Boosting Machine replaces Naive Bayes for feature correlation handling
- LightGBM for fast, efficient ensemble learning
- Feature engineering from multiple signal sources
- Cold start problem handling with pre-trained models
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import json
import pickle
import os
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from worker_ant_v1.utils.logger import setup_logger


@dataclass
class OpportunitySignal:
    """Trading opportunity signal with confidence metrics"""
    token_address: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    price_target: Optional[float]
    risk_level: float
    narrative_alignment: float
    time_horizon: str  # 'short', 'medium', 'long'
    reasoning: str
    metadata: Dict[str, Any]


@dataclass
class SwarmConsensus:
    """Consensus result from swarm analysis"""
    recommended_action: str
    consensus_confidence: float
    narrative_weight: float
    risk_assessment: float
    position_size_recommendation: float
    entry_conditions: Dict[str, Any]
    exit_conditions: Dict[str, Any]
    reasoning: str


class WinRatePredictionModel:
    """
    MATHEMATICAL ENHANCEMENT: Gradient Boosting Machine Win Rate Predictor
    
    Replaces Naive Bayes with LightGBM to handle feature correlation and complex interactions.
    Provides sophisticated signal fusion and cold start problem handling.
    
    Features include:
    - Technical indicators, sentiment scores, market structure
    - Narrative alignment, volatility measures, liquidity metrics
    - Time-based features, market regime indicators
    """
    
    def __init__(self, model_dir: str = "data/win_rate_models"):
        self.logger = setup_logger("WinRatePredictionModel")
        self.model_dir = model_dir
        
        # LightGBM model
        self.gbm_model: Optional[lgb.LGBMClassifier] = None
        self.feature_scaler = StandardScaler()
        
        # Model configuration
        self.model_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Feature engineering configuration
        self.feature_names = [
            # Technical indicators
            'sentiment_score', 'viral_potential', 'technical_momentum',
            'volume_profile', 'volatility_score', 'liquidity_score',
            # Market structure
            'market_cap', 'holder_count', 'transaction_velocity',
            'whale_concentration', 'dev_holdings_percent',
            # Narrative and timing
            'narrative_alignment', 'market_timing_score', 'trend_strength',
            'social_momentum', 'price_momentum', 'volume_momentum',
            # Risk factors
            'rug_risk_score', 'honeypot_risk', 'liquidity_risk',
            'whale_dump_risk', 'contract_risk'
        ]
        
        # Training status
        self.model_trained = False
        self.cold_start_handled = False
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize GBM model with optimal parameters"""
        try:
            self.gbm_model = lgb.LGBMClassifier(**self.model_params)
            self.logger.info("✅ LightGBM win rate model initialized")
            
            # Try to load pre-trained model
            self._load_model()
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing GBM model: {e}")
    
    def _load_model(self):
        """Load pre-trained model if available"""
        try:
            model_path = f"{self.model_dir}/win_rate_gbm_model.pkl"
            scaler_path = f"{self.model_dir}/win_rate_scaler.pkl"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, 'rb') as f:
                    self.gbm_model = pickle.load(f)
                
                with open(scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                
                self.model_trained = True
                self.logger.info("✅ Pre-trained GBM win rate model loaded")
            else:
                self.logger.warning("⚠️ No pre-trained model found, creating default model")
                self._create_cold_start_model()
                
        except Exception as e:
            self.logger.error(f"❌ Error loading pre-trained model: {e}")
            self._create_cold_start_model()
    
    def _create_cold_start_model(self):
        """
        COLD START SOLUTION: Create default model with synthetic training data
        
        Addresses the cold start problem by training on synthetic data patterns
        that represent typical winning vs losing trades.
        """
        try:
            self.logger.info("🚀 Creating cold start GBM model with synthetic data")
            
            # Generate synthetic training data
            np.random.seed(42)
            n_samples = 2000
            
            # Create realistic feature distributions for winning trades
            winning_features = []
            for _ in range(n_samples // 2):
                features = {
                    'sentiment_score': np.random.normal(0.7, 0.15),  # Higher sentiment for wins
                    'viral_potential': np.random.normal(0.6, 0.2),
                    'technical_momentum': np.random.normal(0.65, 0.2),
                    'volume_profile': np.random.normal(0.7, 0.15),
                    'volatility_score': np.random.normal(0.5, 0.2),
                    'liquidity_score': np.random.normal(0.8, 0.1),  # Higher liquidity for wins
                    'market_cap': np.random.lognormal(12, 1),
                    'holder_count': np.random.lognormal(5, 0.5),
                    'transaction_velocity': np.random.normal(0.6, 0.2),
                    'whale_concentration': np.random.normal(0.3, 0.15),  # Lower concentration
                    'dev_holdings_percent': np.random.normal(15, 5),
                    'narrative_alignment': np.random.normal(0.75, 0.15),
                    'market_timing_score': np.random.normal(0.65, 0.2),
                    'trend_strength': np.random.normal(0.7, 0.15),
                    'social_momentum': np.random.normal(0.6, 0.2),
                    'price_momentum': np.random.normal(0.65, 0.2),
                    'volume_momentum': np.random.normal(0.7, 0.15),
                    'rug_risk_score': np.random.normal(0.2, 0.1),  # Lower risk for wins
                    'honeypot_risk': np.random.normal(0.15, 0.1),
                    'liquidity_risk': np.random.normal(0.1, 0.05),
                    'whale_dump_risk': np.random.normal(0.2, 0.1),
                    'contract_risk': np.random.normal(0.1, 0.05)
                }
                winning_features.append([max(0, min(1, features[name])) for name in self.feature_names])
            
            # Create realistic feature distributions for losing trades
            losing_features = []
            for _ in range(n_samples // 2):
                features = {
                    'sentiment_score': np.random.normal(0.3, 0.15),  # Lower sentiment for losses
                    'viral_potential': np.random.normal(0.25, 0.15),
                    'technical_momentum': np.random.normal(0.35, 0.2),
                    'volume_profile': np.random.normal(0.4, 0.2),
                    'volatility_score': np.random.normal(0.7, 0.2),  # Higher volatility
                    'liquidity_score': np.random.normal(0.3, 0.15),  # Lower liquidity
                    'market_cap': np.random.lognormal(10, 1.5),
                    'holder_count': np.random.lognormal(3, 0.8),
                    'transaction_velocity': np.random.normal(0.3, 0.15),
                    'whale_concentration': np.random.normal(0.7, 0.2),  # Higher concentration
                    'dev_holdings_percent': np.random.normal(25, 10),
                    'narrative_alignment': np.random.normal(0.25, 0.15),
                    'market_timing_score': np.random.normal(0.3, 0.2),
                    'trend_strength': np.random.normal(0.25, 0.15),
                    'social_momentum': np.random.normal(0.3, 0.15),
                    'price_momentum': np.random.normal(0.25, 0.2),
                    'volume_momentum': np.random.normal(0.3, 0.15),
                    'rug_risk_score': np.random.normal(0.7, 0.2),  # Higher risk for losses
                    'honeypot_risk': np.random.normal(0.6, 0.2),
                    'liquidity_risk': np.random.normal(0.8, 0.15),
                    'whale_dump_risk': np.random.normal(0.7, 0.2),
                    'contract_risk': np.random.normal(0.6, 0.2)
                }
                losing_features.append([max(0, min(1, features[name])) for name in self.feature_names])
            
            # Combine datasets
            X = np.array(winning_features + losing_features)
            y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))
            
            # Train the model
            self.feature_scaler.fit(X)
            X_scaled = self.feature_scaler.transform(X)
            
            self.gbm_model.fit(X_scaled, y)
            
            self.model_trained = True
            self.cold_start_handled = True
            
            self.logger.info("✅ Cold start GBM model trained successfully with synthetic data")
            
        except Exception as e:
            self.logger.error(f"❌ Error creating cold start model: {e}")
    
    def _extract_features(self, token_address: str, market_data: Dict[str, Any], 
                         current_signals: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from market data and signals"""
        try:
            features = []
            
            # Technical indicators
            features.append(current_signals.get('sentiment_score', 0.5))
            features.append(current_signals.get('viral_potential', 0.3))
            features.append(current_signals.get('technical_momentum', 0.5))
            features.append(current_signals.get('volume_profile', 0.5))
            features.append(market_data.get('volatility_score', 0.5))
            features.append(market_data.get('liquidity_score', 0.5))
            
            # Market structure
            features.append(market_data.get('market_cap', 100000.0) / 1000000.0)  # Normalize
            features.append(market_data.get('holder_count', 100) / 1000.0)  # Normalize
            features.append(current_signals.get('transaction_velocity', 0.5))
            features.append(market_data.get('whale_concentration', 0.3))
            features.append(market_data.get('dev_holdings_percent', 15.0) / 100.0)  # Normalize
            
            # Narrative and timing
            features.append(current_signals.get('narrative_alignment', 0.5))
            features.append(current_signals.get('market_timing_score', 0.5))
            features.append(current_signals.get('trend_strength', 0.5))
            features.append(current_signals.get('social_momentum', 0.5))
            features.append(current_signals.get('price_momentum', 0.5))
            features.append(current_signals.get('volume_momentum', 0.5))
            
            # Risk factors
            features.append(current_signals.get('rug_risk_score', 0.3))
            features.append(current_signals.get('honeypot_risk', 0.2))
            features.append(current_signals.get('liquidity_risk', 0.2))
            features.append(current_signals.get('whale_dump_risk', 0.3))
            features.append(current_signals.get('contract_risk', 0.2))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return np.zeros((1, len(self.feature_names)))
    
    async def predict_win_probability(self, token_address: str, market_data: Dict[str, Any], 
                                    current_signals: Dict[str, Any]) -> float:
        """
        ENHANCED WIN RATE PREDICTION: Use GBM to predict win probability
        
        Handles feature correlations and complex interactions that Naive Bayes cannot.
        """
        try:
            if not self.model_trained:
                self.logger.warning("⚠️ Model not trained, using fallback probability")
                return 0.5
            
            # Extract and scale features
            features = self._extract_features(token_address, market_data, current_signals)
            features_scaled = self.feature_scaler.transform(features)
            
            # Get prediction probability
            win_probability = self.gbm_model.predict_proba(features_scaled)[0][1]
            
            # Apply smoothing to prevent extreme predictions
            win_probability = max(0.05, min(0.95, win_probability))
            
            self.logger.debug(f"📊 GBM win prediction for {token_address[:8]}: {win_probability:.3f}")
            
            return win_probability
            
        except Exception as e:
            self.logger.error(f"Error in GBM win prediction: {e}")
            return 0.5


class SwarmDecisionEngine:
    """
    Central decision engine that processes multiple signals and applies
    narrative weighting to determine optimal trading decisions.
    
    MATHEMATICAL ENHANCEMENT: Uses GBM instead of Naive Bayes for win rate prediction.
    """
    
    def __init__(self, kill_switch=None):
        self.logger = setup_logger("SwarmDecisionEngine")
        
        # MATHEMATICAL ENHANCEMENT: GBM Win Rate Predictor
        self.win_rate_predictor = WinRatePredictionModel()
        
        # Safety systems
        self.kill_switch = kill_switch
        
        # Signal processing configuration
        self.confidence_threshold = 0.6
        self.narrative_weight_multiplier = 1.5
        self.max_risk_per_trade = 0.02
        
        # Signal sources and weights
        self.signal_weights = {
            'technical_analysis': 0.25,
            'sentiment_analysis': 0.25, 
            'narrative_analysis': 0.30,
            'market_structure': 0.20
        }
        
        # Anti-fragile decision factors
        self.anti_fragile_factors = {
            'market_volatility_adjustment': True,
            'position_sizing_discipline': True,
            'narrative_momentum_bias': True,
            'risk_cascading_prevention': True
        }
        
        self.logger.info("🧠 Swarm Decision Engine initialized with GBM win rate prediction")
    
    async def analyze_opportunity(
        self, 
        token_address: str, 
        market_data: Dict[str, Any],
        narrative_weight: float = 1.0
    ) -> float:
        """
        MATHEMATICAL ENHANCEMENT: Gradient Boosting Machine Win Rate Predictor
        
        Replaces Naive Bayes with sophisticated GBM that handles feature correlations.
        Uses LightGBM for fast, accurate ensemble learning with cold start handling.
        
        Args:
            token_address: Token contract address
            market_data: Current market data for the token
            narrative_weight: Narrative strength multiplier
            
        Returns:
            float: Win probability (0.0 to 1.0)
        """
        # CRITICAL SAFETY CHECK: Kill switch verification
        if self.kill_switch and self.kill_switch.is_triggered:
            self.logger.critical("Kill switch is active. Aborting call to analyze_opportunity.")
            return 0.0
            
        try:
            self.logger.debug(f"🧠 GBM analysis for {token_address}")
            
            # Gather comprehensive signals from all AI modules
            current_signals = await self._gather_current_signals(token_address, market_data)
            if not current_signals:
                self.logger.warning("⚠️ No current signals available, using default")
                return 0.5
            
            # Enhanced signal processing with narrative weighting
            enhanced_signals = current_signals.copy()
            
            # Apply narrative weighting to relevant signals
            if narrative_weight != 1.0:
                narrative_influenced_signals = [
                    'narrative_alignment', 'social_momentum', 'sentiment_score',
                    'viral_potential', 'trend_strength'
                ]
                
                for signal_name in narrative_influenced_signals:
                    if signal_name in enhanced_signals:
                        original_value = enhanced_signals[signal_name]
                        # Apply narrative weight with bounds checking
                        weighted_value = original_value * narrative_weight
                        enhanced_signals[signal_name] = max(0.0, min(1.0, weighted_value))
                        
                        self.logger.debug(f"📊 Narrative weighting {signal_name}: "
                                        f"{original_value:.3f} → {enhanced_signals[signal_name]:.3f}")
            
            # Use GBM for sophisticated win probability prediction
            win_probability = await self.win_rate_predictor.predict_win_probability(
                token_address, market_data, enhanced_signals
            )
            
            # Apply additional confidence adjustments based on signal quality
            signal_quality_score = await self._assess_signal_quality(enhanced_signals)
            confidence_adjusted_probability = win_probability * (0.8 + 0.2 * signal_quality_score)
            
            # Apply narrative momentum bias (anti-fragile factor)
            if self.anti_fragile_factors.get('narrative_momentum_bias', False) and narrative_weight > 1.2:
                momentum_boost = min(0.1, (narrative_weight - 1.0) * 0.05)
                confidence_adjusted_probability += momentum_boost
            
            # Ensure bounds
            final_probability = max(0.05, min(0.95, confidence_adjusted_probability))
            
            self.logger.info(f"✅ GBM result for {token_address}: {final_probability:.3f} "
                           f"(quality: {signal_quality_score:.2f}, narrative: {narrative_weight:.2f})")
            
            return final_probability
            
        except Exception as e:
            self.logger.error(f"❌ GBM analysis failed for {token_address}: {e}")
            return 0.5  # Safe default
    
    async def _load_signal_probabilities(self) -> Optional[Dict[str, Any]]:
        """
        Load cached signal probabilities from Redis or JSON file
        
        Returns:
            Dict containing signal probabilities or None if unavailable
        """
        try:
            # Try Redis first (fastest)
            try:
                import redis.asyncio as redis
                redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
                cached_data = await redis_client.get("signal_probabilities")
                await redis_client.close()
                
                if cached_data:
                    return json.loads(cached_data)
            except Exception:
                pass  # Fall back to file
            
            # Fall back to JSON file
            import os
            if os.path.exists("data/signal_probabilities.json"):
                with open("data/signal_probabilities.json", "r") as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading signal probabilities: {e}")
            return None
    
    async def _gather_current_signals(self, token_address: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather current signals from all AI ants for Naive Bayes analysis
        
        Args:
            token_address: Token address being analyzed
            market_data: Current market data
            
        Returns:
            Dict of current signal values
        """
        try:
            signals = {}
            
            # Extract signals from market_data (populated by various AI ants)
            signals['sentiment_score'] = market_data.get('sentiment_score', 0.5)
            signals['rug_risk_score'] = market_data.get('rug_risk_score', 0.5)
            signals['narrative_strength'] = market_data.get('narrative_strength', 0.5)
            signals['volume_momentum'] = market_data.get('volume_momentum', 0.5)
            signals['price_momentum'] = market_data.get('price_momentum', 0.5)
            signals['social_buzz'] = market_data.get('social_buzz', 0.5)
            signals['whale_activity'] = market_data.get('whale_activity', 0.5)
            signals['liquidity_health'] = market_data.get('liquidity_health', 0.5)
            
            # Technical indicators
            signals['rsi_signal'] = 1.0 if market_data.get('rsi', 50) < 30 else 0.0  # Oversold
            signals['volume_spike'] = 1.0 if market_data.get('volume_change_24h', 0) > 2.0 else 0.0
            signals['price_breakout'] = market_data.get('price_breakout_signal', 0.0)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error gathering current signals: {e}")
            return {}
    
    async def _assess_signal_quality(self, signals: Dict[str, Any]) -> float:
        """
        MATHEMATICAL ENHANCEMENT: Assess overall signal quality for confidence adjustment
        
        Evaluates the reliability and completeness of available signals for 
        more accurate confidence weighting in GBM predictions.
        
        Args:
            signals: Dictionary of current signal values
            
        Returns:
            float: Signal quality score (0.0 to 1.0)
        """
        try:
            quality_factors = []
            
            # Check signal completeness (how many expected signals are present)
            expected_signals = [
                'sentiment_score', 'viral_potential', 'technical_momentum',
                'volume_profile', 'narrative_alignment', 'rug_risk_score'
            ]
            
            present_signals = sum(1 for signal in expected_signals if signal in signals)
            completeness_score = present_signals / len(expected_signals)
            quality_factors.append(completeness_score)
            
            # Check signal confidence (how far from neutral 0.5 are the signals)
            confidence_scores = []
            for signal_name, signal_value in signals.items():
                if isinstance(signal_value, (int, float)) and 0 <= signal_value <= 1:
                    # Distance from neutral (0.5) indicates signal strength
                    confidence = abs(signal_value - 0.5) * 2  # Scale to 0-1
                    confidence_scores.append(confidence)
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            quality_factors.append(avg_confidence)
            
            # Check for signal consistency (are related signals aligned?)
            consistency_score = 1.0  # Default high consistency
            
            # Example: sentiment and social momentum should be correlated
            sentiment = signals.get('sentiment_score', 0.5)
            social_momentum = signals.get('social_momentum', 0.5)
            if abs(sentiment - social_momentum) > 0.3:  # Large divergence
                consistency_score *= 0.8
            
            # Technical momentum and price momentum should be correlated
            tech_momentum = signals.get('technical_momentum', 0.5)
            price_momentum = signals.get('price_momentum', 0.5)
            if abs(tech_momentum - price_momentum) > 0.3:
                consistency_score *= 0.8
            
            quality_factors.append(consistency_score)
            
            # Check for extreme values that might indicate data quality issues
            extreme_value_penalty = 1.0
            for signal_value in signals.values():
                if isinstance(signal_value, (int, float)):
                    if signal_value <= 0.01 or signal_value >= 0.99:  # Very extreme values
                        extreme_value_penalty *= 0.95  # Small penalty per extreme value
            
            quality_factors.append(extreme_value_penalty)
            
            # Calculate weighted average
            weights = [0.3, 0.3, 0.25, 0.15]  # Completeness, confidence, consistency, extreme values
            final_quality = sum(factor * weight for factor, weight in zip(quality_factors, weights))
            
            # Ensure bounds
            final_quality = max(0.1, min(1.0, final_quality))
            
            self.logger.debug(f"📊 Signal quality assessment: {final_quality:.3f} "
                           f"(completeness: {completeness_score:.2f}, confidence: {avg_confidence:.2f}, "
                           f"consistency: {consistency_score:.2f})")
            
            return final_quality
            
        except Exception as e:
            self.logger.error(f"Error assessing signal quality: {e}")
            return 0.5  # Default medium quality

    def _signal_is_positive(self, signal_value: Any) -> bool:
        """
        Determine if a signal value should be considered 'positive'
        Same logic as in NightlyEvolutionSystem for consistency
        """
        if isinstance(signal_value, (int, float)):
            return signal_value > 0.5
        elif isinstance(signal_value, bool):
            return signal_value
        elif isinstance(signal_value, str):
            return signal_value.lower() in ['true', 'positive', 'bullish', 'buy']
        else:
            return False
    
    async def _gather_signals(self, token_address: str, market_data: Dict[str, Any]) -> List[OpportunitySignal]:
        """Gather signals from multiple analysis sources"""
        signals = []
        
        try:
            # Technical analysis signal
            technical_signal = await self._generate_technical_signal(token_address, market_data)
            if technical_signal:
                signals.append(technical_signal)
            
            # Sentiment analysis signal
            sentiment_signal = await self._generate_sentiment_signal(token_address, market_data)
            if sentiment_signal:
                signals.append(sentiment_signal)
            
            # Market structure signal
            structure_signal = await self._generate_structure_signal(token_address, market_data)
            if structure_signal:
                signals.append(structure_signal)
            
            # Narrative momentum signal
            narrative_signal = await self._generate_narrative_signal(token_address, market_data)
            if narrative_signal:
                signals.append(narrative_signal)
            
            return signals
            
        except Exception as e:
            self.logger.warning(f"Signal gathering failed for {token_address}: {e}")
            return []
    
    async def _generate_technical_signal(self, token_address: str, market_data: Dict[str, Any]) -> Optional[OpportunitySignal]:
        """Generate technical analysis signal"""
        try:
            # Extract technical indicators from market data
            price = market_data.get('current_price', 0)
            volume = market_data.get('volume_24h', 0)
            price_change_24h = market_data.get('price_change_24h', 0)
            
            # Simple momentum-based signal
            if price_change_24h > 0.05 and volume > 10000:  # 5% gain with volume
                signal_type = 'buy'
                confidence = min(0.8, abs(price_change_24h) * 10)
            elif price_change_24h < -0.05:  # 5% loss
                signal_type = 'sell'
                confidence = min(0.7, abs(price_change_24h) * 8)
            else:
                signal_type = 'hold'
                confidence = 0.5
            
            return OpportunitySignal(
                token_address=token_address,
                signal_type=signal_type,
                confidence=confidence,
                price_target=price * (1 + price_change_24h * 0.5),
                risk_level=abs(price_change_24h),
                narrative_alignment=0.5,  # Neutral for technical
                time_horizon='short',
                reasoning=f"Technical: {price_change_24h:.1%} change, volume {volume:,.0f}",
                metadata={'source': 'technical_analysis', 'price': price, 'volume': volume}
            )
            
        except Exception as e:
            self.logger.warning(f"Technical signal generation failed: {e}")
            return None
    
    async def _generate_sentiment_signal(self, token_address: str, market_data: Dict[str, Any]) -> Optional[OpportunitySignal]:
        """Generate sentiment analysis signal"""
        try:
            # Mock sentiment analysis - would integrate with real sentiment engine
            sentiment_score = market_data.get('sentiment_score', 0.5)
            social_volume = market_data.get('social_volume', 0)
            
            if sentiment_score > 0.7:
                signal_type = 'buy'
                confidence = sentiment_score * 0.9
            elif sentiment_score < 0.3:
                signal_type = 'sell'
                confidence = (1 - sentiment_score) * 0.8
            else:
                signal_type = 'hold'
                confidence = 0.4
            
            return OpportunitySignal(
                token_address=token_address,
                signal_type=signal_type,
                confidence=confidence,
                price_target=None,
                risk_level=1 - sentiment_score if sentiment_score < 0.5 else sentiment_score - 0.5,
                narrative_alignment=sentiment_score,
                time_horizon='medium',
                reasoning=f"Sentiment: {sentiment_score:.2f} score, social volume {social_volume}",
                metadata={'source': 'sentiment_analysis', 'sentiment': sentiment_score}
            )
            
        except Exception as e:
            self.logger.warning(f"Sentiment signal generation failed: {e}")
            return None
    
    async def _generate_structure_signal(self, token_address: str, market_data: Dict[str, Any]) -> Optional[OpportunitySignal]:
        """Generate market structure signal"""
        try:
            # Market structure analysis
            liquidity = market_data.get('liquidity_usd', 0)
            holder_count = market_data.get('holder_count', 0)
            mcap = market_data.get('market_cap', 0)
            
            # Structure score based on liquidity and distribution
            structure_score = 0.0
            if liquidity > 50000:  # Good liquidity
                structure_score += 0.3
            if holder_count > 1000:  # Good distribution
                structure_score += 0.3
            if 1000000 < mcap < 100000000:  # Sweet spot market cap
                structure_score += 0.4
            
            signal_type = 'buy' if structure_score > 0.6 else 'hold' if structure_score > 0.3 else 'sell'
            confidence = structure_score
            
            return OpportunitySignal(
                token_address=token_address,
                signal_type=signal_type,
                confidence=confidence,
                price_target=None,
                risk_level=1 - structure_score,
                narrative_alignment=0.5,
                time_horizon='long',
                reasoning=f"Structure: liquidity ${liquidity:,.0f}, {holder_count} holders",
                metadata={'source': 'market_structure', 'liquidity': liquidity}
            )
            
        except Exception as e:
            self.logger.warning(f"Structure signal generation failed: {e}")
            return None
    
    async def _generate_narrative_signal(self, token_address: str, market_data: Dict[str, Any]) -> Optional[OpportunitySignal]:
        """Generate narrative momentum signal"""
        try:
            # Narrative indicators
            trending_keywords = market_data.get('trending_keywords', [])
            narrative_mentions = market_data.get('narrative_mentions', 0)
            influencer_sentiment = market_data.get('influencer_sentiment', 0.5)
            
            # Calculate narrative momentum
            narrative_score = 0.0
            if len(trending_keywords) > 3:
                narrative_score += 0.3
            if narrative_mentions > 100:
                narrative_score += 0.3
            narrative_score += influencer_sentiment * 0.4
            
            signal_type = 'buy' if narrative_score > 0.7 else 'hold' if narrative_score > 0.4 else 'sell'
            confidence = narrative_score
            
            return OpportunitySignal(
                token_address=token_address,
                signal_type=signal_type,
                confidence=confidence,
                price_target=None,
                risk_level=0.3,  # Narrative trades are medium risk
                narrative_alignment=narrative_score,
                time_horizon='short',
                reasoning=f"Narrative: {len(trending_keywords)} keywords, {narrative_mentions} mentions",
                metadata={'source': 'narrative_analysis', 'keywords': trending_keywords}
            )
            
        except Exception as e:
            self.logger.warning(f"Narrative signal generation failed: {e}")
            return None
    
    def _apply_narrative_weighting(self, signals: List[OpportunitySignal], narrative_weight: float) -> List[OpportunitySignal]:
        """Apply narrative weighting to boost signal confidence"""
        weighted_signals = []
        
        for signal in signals:
            # Clone the signal
            weighted_signal = OpportunitySignal(
                token_address=signal.token_address,
                signal_type=signal.signal_type,
                confidence=signal.confidence,
                price_target=signal.price_target,
                risk_level=signal.risk_level,
                narrative_alignment=signal.narrative_alignment,
                time_horizon=signal.time_horizon,
                reasoning=signal.reasoning,
                metadata=signal.metadata.copy()
            )
            
            # Apply narrative weighting to confidence
            if narrative_weight > 1.0:  # Strong narrative
                boost_factor = min(1.5, 1.0 + (narrative_weight - 1.0) * 0.5)
                weighted_signal.confidence = min(1.0, signal.confidence * boost_factor)
                weighted_signal.reasoning += f" [Narrative boost: {boost_factor:.2f}x]"
            
            weighted_signals.append(weighted_signal)
        
        return weighted_signals
    
    async def _calculate_swarm_consensus(self, signals: List[OpportunitySignal], narrative_weight: float) -> SwarmConsensus:
        """Calculate consensus from weighted signals"""
        if not signals:
            return self._create_safe_consensus("hold", 0.0, narrative_weight)
        
        try:
            # CRITICAL: TRUE CONSENSUS LOGIC - ALL SOURCES MUST AGREE FOR BUY
            # Categorize signals by source
            technical_signals = [s for s in signals if s.metadata.get('source') == 'technical_analysis']
            sentiment_signals = [s for s in signals if s.metadata.get('source') == 'sentiment_analysis']
            narrative_signals = [s for s in signals if s.metadata.get('source') == 'narrative_analysis']
            security_signals = [s for s in signals if s.metadata.get('source') == 'security_analysis']
            
            # Check for unanimous BUY consensus across ALL critical sources
            consensus_action = "hold"  # Default to safe hold
            consensus_confidence = 0.0
            
            # Require ALL sources to have BUY signals for a BUY recommendation
            technical_buy = any(s.signal_type == 'buy' and s.confidence >= self.confidence_threshold for s in technical_signals)
            sentiment_buy = any(s.signal_type == 'buy' and s.confidence >= self.confidence_threshold for s in sentiment_signals)
            narrative_buy = any(s.signal_type == 'buy' and s.confidence >= self.confidence_threshold for s in narrative_signals)
            security_buy = any(s.signal_type == 'buy' and s.confidence >= self.confidence_threshold for s in security_signals)
            
            # TRUE AND CONDITION: ALL must be true for BUY
            if technical_buy and sentiment_buy and narrative_buy and security_buy:
                consensus_action = "buy"
                # Calculate minimum confidence across sources (weakest link determines overall confidence)
                all_confidences = [s.confidence for s in signals if s.signal_type == 'buy']
                consensus_confidence = min(all_confidences) if all_confidences else 0.0
            
            # Check for ANY sell signal (any source can trigger sell)
            elif any(s.signal_type == 'sell' and s.confidence >= self.confidence_threshold for s in signals):
                consensus_action = "sell"
                sell_confidences = [s.confidence for s in signals if s.signal_type == 'sell']
                consensus_confidence = max(sell_confidences) if sell_confidences else 0.0
            
            # Default: hold with average confidence
            else:
                consensus_action = "hold"
                all_confidences = [s.confidence for s in signals]
                consensus_confidence = np.mean(all_confidences) if all_confidences else 0.0
            
            # Calculate risk assessment
            risk_scores = [s.risk_level for s in signals]
            avg_risk = np.mean(risk_scores) if risk_scores else 0.5
            
            # Position sizing based on confidence and risk
            position_size = self._calculate_position_size(consensus_confidence, avg_risk, narrative_weight)
            
            return SwarmConsensus(
                recommended_action=consensus_action,
                consensus_confidence=consensus_confidence,
                narrative_weight=narrative_weight,
                risk_assessment=avg_risk,
                position_size_recommendation=position_size,
                entry_conditions={'min_confidence': 0.6, 'max_risk': 0.02},
                exit_conditions={'stop_loss': 0.05, 'take_profit': 0.15},
                reasoning=f"Swarm consensus: {len(signals)} signals, narrative weight {narrative_weight:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Consensus calculation failed: {e}")
            return self._create_safe_consensus("hold", 0.0, narrative_weight)
    
    def _calculate_position_size(self, confidence: float, risk: float, narrative_weight: float) -> float:
        """Calculate recommended position size based on confidence and risk"""
        base_size = min(self.max_risk_per_trade, confidence * 0.03)
        
        # Adjust for risk
        risk_adjusted_size = base_size * (1 - risk)
        
        # Narrative weighting adjustment
        if narrative_weight > 1.2:  # Strong narrative
            risk_adjusted_size *= min(1.3, narrative_weight)
        
        return max(0.005, min(0.05, risk_adjusted_size))  # Cap between 0.5% and 5%
    
    async def _apply_anti_fragile_filters(self, consensus: SwarmConsensus, market_data: Dict[str, Any]) -> SwarmConsensus:
        """Apply anti-fragile decision filters"""
        filtered_consensus = consensus
        
        try:
            # Market volatility adjustment
            if self.anti_fragile_factors['market_volatility_adjustment']:
                market_volatility = market_data.get('volatility_score', 0.5)
                if market_volatility > 0.8:  # High volatility
                    filtered_consensus.position_size_recommendation *= 0.5
                    filtered_consensus.reasoning += " [Volatility reduction applied]"
            
            # Position sizing discipline
            if self.anti_fragile_factors['position_sizing_discipline']:
                if filtered_consensus.position_size_recommendation > self.max_risk_per_trade:
                    filtered_consensus.position_size_recommendation = self.max_risk_per_trade
                    filtered_consensus.reasoning += " [Position size capped]"
            
            # Confidence threshold enforcement
            if filtered_consensus.consensus_confidence < self.confidence_threshold:
                filtered_consensus.recommended_action = "hold"
                filtered_consensus.reasoning += " [Confidence too low, defaulting to hold]"
            
            return filtered_consensus
            
        except Exception as e:
            self.logger.warning(f"Anti-fragile filter application failed: {e}")
            return consensus
    
    def _create_safe_consensus(self, action: str, confidence: float, narrative_weight: float) -> SwarmConsensus:
        """Create a safe consensus for error cases"""
        return SwarmConsensus(
            recommended_action=action,
            consensus_confidence=confidence,
            narrative_weight=narrative_weight,
            risk_assessment=0.5,
            position_size_recommendation=0.0,
            entry_conditions={},
            exit_conditions={},
            reasoning="Safe consensus due to insufficient data or error"
        )
    
    def get_narrative_weight_for_token(self, token_address: str, narrative_data: Dict[str, Any]) -> float:
        """Calculate narrative weight for a specific token"""
        try:
            # Extract narrative metrics
            narrative_strength = narrative_data.get('narrative_strength', 0.5)
            cultural_relevance = narrative_data.get('cultural_relevance', 0.5)
            momentum_score = narrative_data.get('momentum_score', 0.5)
            
            # Calculate composite narrative weight
            narrative_weight = (
                narrative_strength * 0.4 +
                cultural_relevance * 0.3 +
                momentum_score * 0.3
            )
            
            # Scale to 0.5 - 2.0 range
            scaled_weight = 0.5 + (narrative_weight * 1.5)
            
            return max(0.5, min(2.0, scaled_weight))
            
        except Exception as e:
            self.logger.warning(f"Narrative weight calculation failed: {e}")
            return 1.0
    
    def update_signal_weights(self, new_weights: Dict[str, float]):
        """Update signal source weights"""
        self.signal_weights.update(new_weights)
        self.logger.info(f"📊 Updated signal weights: {self.signal_weights}")
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold for decisions"""
        self.confidence_threshold = max(0.1, min(0.9, new_threshold))
        self.logger.info(f"🎯 Updated confidence threshold: {self.confidence_threshold}") 