"""
Technical Analysis Engine for Aggressive Memecoin Trading
======================================================

Advanced technical indicators optimized for memecoin volatility and short-term trading.
Implements RSI, MACD, Bollinger Bands, EMA, and custom momentum indicators.
"""

import asyncio
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import talib
from scipy import stats

from worker_ant_v1.swarm_config import aggressive_meme_strategy, ml_model_config
from worker_ant_v1.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TechnicalSignals:
    """Technical analysis signals for trading decisions"""
    token_symbol: str
    timestamp: datetime
    
    # RSI Signals
    rsi_value: float
    rsi_signal: str  # "buy", "sell", "hold"
    rsi_strength: float  # 0 to 1
    
    # MACD Signals
    macd_line: float
    macd_signal: float
    macd_histogram: float
    macd_trend: str  # "bullish", "bearish", "neutral"
    macd_strength: float  # 0 to 1
    
    # Bollinger Bands
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: float  # -1 (lower) to +1 (upper)
    bb_squeeze: bool  # Low volatility indicator
    
    # Moving Averages
    ema_12: float
    ema_26: float
    ema_crossover: str  # "golden", "death", "none"
    
    # Volume Analysis
    volume_spike: bool
    volume_trend: str  # "increasing", "decreasing", "stable"
    
    # Custom Memecoin Indicators
    momentum_score: float  # -1 to +1
    volatility_index: float  # 0 to 1
    breakout_probability: float  # 0 to 1
    
    # Overall Signal
    overall_signal: str  # "strong_buy", "buy", "hold", "sell", "strong_sell"
    confidence: float  # 0 to 1


@dataclass
class PriceData:
    """OHLCV price data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class TechnicalAnalyzer:
    """Advanced technical analysis engine for memecoin trading"""
    
    def __init__(self):
        self.price_history: Dict[str, List[PriceData]] = {}
        self.signals_history: Dict[str, List[TechnicalSignals]] = {}
        
        # Technical indicator periods (optimized for memecoin volatility)
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal_period = 9
        self.bb_period = 20
        self.bb_std = 2.0
        self.ema_short = 12
        self.ema_long = 26
        
        # Memecoin-specific parameters
        self.volume_spike_threshold = 2.0  # 2x average volume
        self.momentum_period = 10
        self.volatility_period = 20
        
    async def analyze_token_technicals(self, token_symbol: str, price_data: List[PriceData]) -> TechnicalSignals:
        """Perform comprehensive technical analysis on token"""
        
        # Update price history
        self._update_price_history(token_symbol, price_data)
        
        # Get sufficient data for analysis
        if len(self.price_history[token_symbol]) < 50:
            logger.warning(f"Insufficient data for {token_symbol} technical analysis")
            return self._create_neutral_signals(token_symbol)
        
        df = self._price_data_to_dataframe(self.price_history[token_symbol])
        
        # Calculate all technical indicators
        signals = TechnicalSignals(
            token_symbol=token_symbol,
            timestamp=datetime.now(),
            
            # RSI Analysis
            **self._calculate_rsi_signals(df),
            
            # MACD Analysis  
            **self._calculate_macd_signals(df),
            
            # Bollinger Bands
            **self._calculate_bollinger_signals(df),
            
            # Moving Averages
            **self._calculate_ema_signals(df),
            
            # Volume Analysis
            **self._calculate_volume_signals(df),
            
            # Custom Memecoin Indicators
            **self._calculate_memecoin_indicators(df)
        )
        
        # Generate overall signal
        signals.overall_signal, signals.confidence = self._generate_overall_signal(signals)
        
        # Store in history
        if token_symbol not in self.signals_history:
            self.signals_history[token_symbol] = []
        self.signals_history[token_symbol].append(signals)
        
        # Keep only last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        self.signals_history[token_symbol] = [
            s for s in self.signals_history[token_symbol]
            if s.timestamp > cutoff
        ]
        
        return signals
    
    def _update_price_history(self, token_symbol: str, new_data: List[PriceData]):
        """Update price history with new data"""
        
        if token_symbol not in self.price_history:
            self.price_history[token_symbol] = []
        
        # Add new data
        self.price_history[token_symbol].extend(new_data)
        
        # Sort by timestamp
        self.price_history[token_symbol].sort(key=lambda x: x.timestamp)
        
        # Keep only last 500 data points (about 8 hours of 1-minute data)
        if len(self.price_history[token_symbol]) > 500:
            self.price_history[token_symbol] = self.price_history[token_symbol][-500:]
    
    def _price_data_to_dataframe(self, price_data: List[PriceData]) -> pd.DataFrame:
        """Convert price data to pandas DataFrame"""
        
        data = {
            'timestamp': [p.timestamp for p in price_data],
            'open': [p.open for p in price_data],
            'high': [p.high for p in price_data],
            'low': [p.low for p in price_data],
            'close': [p.close for p in price_data],
            'volume': [p.volume for p in price_data]
        }
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _calculate_rsi_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate RSI-based signals"""
        
        # Calculate RSI
        rsi = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
        current_rsi = rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else 50.0
        
        # Generate RSI signals
        if current_rsi <= aggressive_meme_strategy.rsi_oversold_threshold:
            rsi_signal = "buy"
            rsi_strength = (aggressive_meme_strategy.rsi_oversold_threshold - current_rsi) / aggressive_meme_strategy.rsi_oversold_threshold
        elif current_rsi >= aggressive_meme_strategy.rsi_overbought_threshold:
            rsi_signal = "sell" 
            rsi_strength = (current_rsi - aggressive_meme_strategy.rsi_overbought_threshold) / (100 - aggressive_meme_strategy.rsi_overbought_threshold)
        else:
            rsi_signal = "hold"
            # Strength based on distance from neutral (50)
            rsi_strength = abs(current_rsi - 50) / 50
        
        return {
            'rsi_value': current_rsi,
            'rsi_signal': rsi_signal,
            'rsi_strength': min(rsi_strength, 1.0)
        }
    
    def _calculate_macd_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate MACD-based signals"""
        
        # Calculate MACD
        macd_line, macd_signal, macd_hist = talib.MACD(
            df['close'].values,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal_period
        )
        
        current_macd = macd_line[-1] if len(macd_line) > 0 and not np.isnan(macd_line[-1]) else 0.0
        current_signal = macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0.0
        current_hist = macd_hist[-1] if len(macd_hist) > 0 and not np.isnan(macd_hist[-1]) else 0.0
        
        # MACD trend analysis
        if current_macd > current_signal and current_hist > 0:
            macd_trend = "bullish"
        elif current_macd < current_signal and current_hist < 0:
            macd_trend = "bearish"
        else:
            macd_trend = "neutral"
        
        # MACD strength based on histogram and crossover
        macd_strength = min(abs(current_hist) / (abs(current_macd) + 1e-6), 1.0)
        
        return {
            'macd_line': current_macd,
            'macd_signal': current_signal,
            'macd_histogram': current_hist,
            'macd_trend': macd_trend,
            'macd_strength': macd_strength
        }
    
    def _calculate_bollinger_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate Bollinger Bands signals"""
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            df['close'].values,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0
        )
        
        current_price = df['close'].iloc[-1]
        current_upper = bb_upper[-1] if len(bb_upper) > 0 and not np.isnan(bb_upper[-1]) else current_price
        current_middle = bb_middle[-1] if len(bb_middle) > 0 and not np.isnan(bb_middle[-1]) else current_price
        current_lower = bb_lower[-1] if len(bb_lower) > 0 and not np.isnan(bb_lower[-1]) else current_price
        
        # Calculate position within bands (-1 to +1)
        if current_upper != current_lower:
            bb_position = (current_price - current_middle) / (current_upper - current_middle)
        else:
            bb_position = 0.0
        
        # Bollinger Squeeze detection (low volatility)
        band_width = (current_upper - current_lower) / current_middle
        avg_width = np.mean([(bb_upper[i] - bb_lower[i]) / bb_middle[i] 
                            for i in range(-20, 0) 
                            if not np.isnan(bb_upper[i]) and bb_middle[i] != 0])
        bb_squeeze = band_width < avg_width * 0.8
        
        return {
            'bb_upper': current_upper,
            'bb_middle': current_middle,
            'bb_lower': current_lower,
            'bb_position': np.clip(bb_position, -1, 1),
            'bb_squeeze': bb_squeeze
        }
    
    def _calculate_ema_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate EMA crossover signals"""
        
        # Calculate EMAs
        ema_12 = talib.EMA(df['close'].values, timeperiod=self.ema_short)
        ema_26 = talib.EMA(df['close'].values, timeperiod=self.ema_long)
        
        current_ema_12 = ema_12[-1] if len(ema_12) > 0 and not np.isnan(ema_12[-1]) else df['close'].iloc[-1]
        current_ema_26 = ema_26[-1] if len(ema_26) > 0 and not np.isnan(ema_26[-1]) else df['close'].iloc[-1]
        
        # Check for crossovers
        if len(ema_12) >= 2 and len(ema_26) >= 2:
            prev_ema_12 = ema_12[-2]
            prev_ema_26 = ema_26[-2]
            
            # Golden cross (bullish)
            if prev_ema_12 <= prev_ema_26 and current_ema_12 > current_ema_26:
                ema_crossover = "golden"
            # Death cross (bearish)
            elif prev_ema_12 >= prev_ema_26 and current_ema_12 < current_ema_26:
                ema_crossover = "death"
            else:
                ema_crossover = "none"
        else:
            ema_crossover = "none"
        
        return {
            'ema_12': current_ema_12,
            'ema_26': current_ema_26,
            'ema_crossover': ema_crossover
        }
    
    def _calculate_volume_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate volume-based signals"""
        
        current_volume = df['volume'].iloc[-1]
        
        # Volume trend analysis
        if len(df) >= 10:
            recent_volumes = df['volume'].iloc[-10:].values
            avg_volume = np.mean(recent_volumes[:-1])
            
            # Volume spike detection
            volume_spike = current_volume > avg_volume * self.volume_spike_threshold
            
            # Volume trend
            volume_slope, _, _, _, _ = stats.linregress(range(len(recent_volumes)), recent_volumes)
            if volume_slope > avg_volume * 0.1:
                volume_trend = "increasing"
            elif volume_slope < -avg_volume * 0.1:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"
        else:
            volume_spike = False
            volume_trend = "stable"
        
        return {
            'volume_spike': volume_spike,
            'volume_trend': volume_trend
        }
    
    def _calculate_memecoin_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate custom memecoin-specific indicators"""
        
        # Momentum Score (rate of change with volatility adjustment)
        if len(df) >= self.momentum_period:
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-self.momentum_period]) / df['close'].iloc[-self.momentum_period]
            volatility = df['close'].iloc[-self.momentum_period:].std() / df['close'].iloc[-self.momentum_period:].mean()
            momentum_score = np.tanh(price_change / (volatility + 0.01)) # Normalize to -1, +1
        else:
            momentum_score = 0.0
        
        # Volatility Index
        if len(df) >= self.volatility_period:
            returns = df['close'].pct_change().iloc[-self.volatility_period:]
            volatility_index = min(returns.std() * np.sqrt(len(returns)), 1.0)
        else:
            volatility_index = 0.0
        
        # Breakout Probability (combination of volume, price action, and volatility)
        if len(df) >= 20:
            # Price near resistance/support
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            current_price = df['close'].iloc[-1]
            
            # Distance from range bounds
            range_size = recent_high - recent_low
            if range_size > 0:
                upper_distance = (recent_high - current_price) / range_size
                lower_distance = (current_price - recent_low) / range_size
                
                # Volume confirmation
                avg_volume = df['volume'].iloc[-20:-1].mean()
                current_volume = df['volume'].iloc[-1]
                volume_factor = min(current_volume / avg_volume, 2.0) / 2.0
                
                # Volatility factor
                volatility_factor = min(volatility_index * 2, 1.0)
                
                # Breakout probability higher near bounds with volume and volatility
                breakout_probability = (
                    (1 - min(upper_distance, lower_distance)) * 0.5 +
                    volume_factor * 0.3 +
                    volatility_factor * 0.2
                )
            else:
                breakout_probability = 0.0
        else:
            breakout_probability = 0.0
        
        return {
            'momentum_score': np.clip(momentum_score, -1, 1),
            'volatility_index': volatility_index,
            'breakout_probability': min(breakout_probability, 1.0)
        }
    
    def _generate_overall_signal(self, signals: TechnicalSignals) -> Tuple[str, float]:
        """Generate overall trading signal with confidence"""
        
        # Signal scoring system
        signal_score = 0.0
        confidence_factors = []
        
        # RSI contribution
        if signals.rsi_signal == "buy":
            signal_score += 1.0 * signals.rsi_strength
            confidence_factors.append(signals.rsi_strength)
        elif signals.rsi_signal == "sell":
            signal_score -= 1.0 * signals.rsi_strength
            confidence_factors.append(signals.rsi_strength)
        
        # MACD contribution
        if signals.macd_trend == "bullish":
            signal_score += 1.0 * signals.macd_strength
            confidence_factors.append(signals.macd_strength)
        elif signals.macd_trend == "bearish":
            signal_score -= 1.0 * signals.macd_strength
            confidence_factors.append(signals.macd_strength)
        
        # EMA crossover contribution
        if signals.ema_crossover == "golden":
            signal_score += 1.5  # Strong signal
            confidence_factors.append(0.8)
        elif signals.ema_crossover == "death":
            signal_score -= 1.5  # Strong signal
            confidence_factors.append(0.8)
        
        # Bollinger Bands contribution
        if signals.bb_position < -0.8:  # Near lower band
            signal_score += 0.8
            confidence_factors.append(0.6)
        elif signals.bb_position > 0.8:  # Near upper band
            signal_score -= 0.8
            confidence_factors.append(0.6)
        
        # Volume spike boost
        if signals.volume_spike:
            signal_score *= 1.3  # Amplify signal with volume confirmation
            confidence_factors.append(0.7)
        
        # Momentum contribution
        signal_score += signals.momentum_score * 0.8
        confidence_factors.append(abs(signals.momentum_score))
        
        # Breakout probability
        if signals.breakout_probability > 0.7:
            signal_score *= 1.2  # Boost signal near breakout
            confidence_factors.append(signals.breakout_probability)
        
        # Generate signal classification
        if signal_score >= 2.5:
            overall_signal = "strong_buy"
        elif signal_score >= 1.0:
            overall_signal = "buy"
        elif signal_score <= -2.5:
            overall_signal = "strong_sell"
        elif signal_score <= -1.0:
            overall_signal = "sell"
        else:
            overall_signal = "hold"
        
        # Calculate confidence
        confidence = min(np.mean(confidence_factors) if confidence_factors else 0.0, 1.0)
        
        return overall_signal, confidence
    
    def _create_neutral_signals(self, token_symbol: str) -> TechnicalSignals:
        """Create neutral signals when insufficient data"""
        
        return TechnicalSignals(
            token_symbol=token_symbol,
            timestamp=datetime.now(),
            rsi_value=50.0,
            rsi_signal="hold",
            rsi_strength=0.0,
            macd_line=0.0,
            macd_signal=0.0,
            macd_histogram=0.0,
            macd_trend="neutral",
            macd_strength=0.0,
            bb_upper=0.0,
            bb_middle=0.0,
            bb_lower=0.0,
            bb_position=0.0,
            bb_squeeze=False,
            ema_12=0.0,
            ema_26=0.0,
            ema_crossover="none",
            volume_spike=False,
            volume_trend="stable",
            momentum_score=0.0,
            volatility_index=0.0,
            breakout_probability=0.0,
            overall_signal="hold",
            confidence=0.0
        )
    
    def get_trading_signals(self, token_symbol: str) -> Dict[str, float]:
        """Get processed trading signals for decision making"""
        
        if token_symbol not in self.signals_history or not self.signals_history[token_symbol]:
            return {
                'technical_signal': 0.0,
                'confidence': 0.0,
                'momentum': 0.0,
                'volatility': 0.0,
                'breakout_probability': 0.0
            }
        
        latest = self.signals_history[token_symbol][-1]
        
        # Convert signal to numeric
        signal_map = {
            "strong_buy": 1.0,
            "buy": 0.5,
            "hold": 0.0,
            "sell": -0.5,
            "strong_sell": -1.0
        }
        
        technical_signal = signal_map.get(latest.overall_signal, 0.0)
        
        return {
            'technical_signal': technical_signal,
            'confidence': latest.confidence,
            'momentum': latest.momentum_score,
            'volatility': latest.volatility_index,
            'breakout_probability': latest.breakout_probability,
            'rsi': latest.rsi_value,
            'volume_spike': latest.volume_spike
        }


# Global technical analyzer instance
technical_analyzer = TechnicalAnalyzer() 