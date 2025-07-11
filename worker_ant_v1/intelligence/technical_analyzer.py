"""
Technical Analysis Engine
=======================

Advanced technical analysis system with pattern recognition
and trend detection capabilities.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging

from worker_ant_v1.utils.logger import setup_logger

class SignalStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"

@dataclass
class TechnicalSignal:
    """Technical analysis signal"""
    indicator: str
    signal_type: str  # buy, sell, hold
    strength: SignalStrength
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TechnicalAnalysis:
    """Technical analysis result"""
    trend_direction: str
    trend_strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    signals: List[Dict[str, Any]]
    overall_score: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

class TechnicalAnalyzer:
    """Technical analysis engine with advanced pattern recognition"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        
        
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.max_history_length: int = 1000
        
        
        self.indicators = {
            'sma_periods': [7, 21, 50],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd_params': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'bollinger_params': {
                'period': 20,
                'std_dev': 2
            }
        }
        
        
        self.thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_surge': 2.0,
            'trend_strength': 0.1
        }
        
    async def analyze_token(self, token_address: str, price_data: Dict) -> TechnicalAnalysis:
        """Perform comprehensive technical analysis"""
        
        try:
            await self._update_price_history(token_address, price_data)
            
            
            prices = self.price_history.get(token_address, [])
            volumes = self.volume_history.get(token_address, [])
            
            if len(prices) < 10:
                return self._create_default_analysis(token_address)
            
            
            sma_signals = self._calculate_sma_signals(prices)
            ema_signals = self._calculate_ema_signals(prices)
            rsi_signal = self._calculate_rsi_signal(prices)
            macd_signal = self._calculate_macd_signal(prices)
            bollinger_signal = self._calculate_bollinger_signal(prices)
            volume_signal = self._calculate_volume_signal(prices, volumes)
            
            
            all_signals = [
                sma_signals, ema_signals, rsi_signal,
                macd_signal, bollinger_signal, volume_signal
            ]
            
            
            valid_signals = [s for s in all_signals if s is not None]
            
            
            trend_direction, trend_strength = self._analyze_trend(prices)
            
            
            support_levels = self._calculate_support_levels(prices)
            resistance_levels = self._calculate_resistance_levels(prices)
            
            
            overall_score = self._calculate_overall_score(valid_signals)
            
            
            confidence = self._calculate_confidence(len(prices), len(valid_signals))
            
            return TechnicalAnalysis(
                token_address=token_address,
                timestamp=datetime.now(),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                signals=valid_signals,
                overall_score=overall_score,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed for {token_address}: {e}")
            return self._create_default_analysis(token_address)
    
    async def _update_price_history(self, token_address: str, price_data: Dict):
        """Update price history for a token"""
        
        current_price = 0.0
        current_volume = 0.0
        
        
        if 'price' in price_data:
            current_price = float(price_data['price'])
        elif 'current_price' in price_data:
            current_price = float(price_data['current_price'])
        elif 'last_price' in price_data:
            current_price = float(price_data['last_price'])
        else:
            self.logger.warning(f"No price data found for {token_address}")
            return
        
        
        if 'volume' in price_data:
            current_volume = float(price_data['volume'])
        elif 'volume_24h' in price_data:
            current_volume = float(price_data['volume_24h'])
        
        
        if token_address not in self.price_history:
            self.price_history[token_address] = []
            self.volume_history[token_address] = []
        
        
        self.price_history[token_address].append(current_price)
        self.volume_history[token_address].append(current_volume)
        
        
        if len(self.price_history[token_address]) > self.max_history_length:
            self.price_history[token_address] = self.price_history[token_address][-self.max_history_length:]
            self.volume_history[token_address] = self.volume_history[token_address][-self.max_history_length:]
    
    def _calculate_sma_signals(self, prices: List[float]) -> Optional[TechnicalSignal]:
        """Calculate Simple Moving Average signals"""
        
        try:
            if len(prices) < max(self.indicators['sma_periods']):
                return None
            
            current_price = prices[-1]
            signals = []
            
            for period in self.indicators['sma_periods']:
                if len(prices) >= period:
                    sma = np.mean(prices[-period:])
                    
                    if current_price > sma:
                        signals.append(1)  # Bullish
                    elif current_price < sma:
                        signals.append(-1)  # Bearish
                    else:
                        signals.append(0)  # Neutral
            
            if not signals:
                return None
            
            
            avg_signal = np.mean(signals)
            
            
            if avg_signal > 0.3:
                signal_type = "buy"
                strength = SignalStrength.MODERATE if avg_signal > 0.7 else SignalStrength.WEAK
            elif avg_signal < -0.3:
                signal_type = "sell"
                strength = SignalStrength.MODERATE if avg_signal < -0.7 else SignalStrength.WEAK
            else:
                signal_type = "hold"
                strength = SignalStrength.WEAK
            
            return TechnicalSignal(
                indicator="SMA",
                signal_type=signal_type,
                strength=strength,
                value=avg_signal,
                timestamp=datetime.now(),
                metadata={"periods": self.indicators['sma_periods']}
            )
            
        except Exception as e:
            self.logger.warning(f"SMA calculation failed: {e}")
            return None
    
    def _calculate_ema_signals(self, prices: List[float]) -> Optional[TechnicalSignal]:
        """Calculate Exponential Moving Average signals"""
        
        try:
            if len(prices) < max(self.indicators['ema_periods']):
                return None
            
            def calculate_ema(data, period):
                alpha = 2 / (period + 1)
                ema = [data[0]]
                for price in data[1:]:
                    ema.append(alpha * price + (1 - alpha) * ema[-1])
                return ema
            
            current_price = prices[-1]
            
            
            ema_fast = calculate_ema(prices, self.indicators['ema_periods'][0])
            ema_slow = calculate_ema(prices, self.indicators['ema_periods'][1])
            
            
            if ema_fast[-1] > ema_slow[-1]:
                signal_type = "buy"
                strength = SignalStrength.MODERATE
                value = 0.5
            elif ema_fast[-1] < ema_slow[-1]:
                signal_type = "sell"
                strength = SignalStrength.MODERATE
                value = -0.5
            else:
                signal_type = "hold"
                strength = SignalStrength.WEAK
                value = 0.0
            
            return TechnicalSignal(
                indicator="EMA",
                signal_type=signal_type,
                strength=strength,
                value=value,
                timestamp=datetime.now(),
                metadata={
                    "fast_period": self.indicators['ema_periods'][0],
                    "slow_period": self.indicators['ema_periods'][1]
                }
            )
            
        except Exception as e:
            self.logger.warning(f"EMA calculation failed: {e}")
            return None
    
    def _calculate_rsi_signal(self, prices: List[float]) -> Optional[TechnicalSignal]:
        """Calculate RSI signal"""
        
        try:
            period = self.indicators['rsi_period']
            if len(prices) < period + 1:
                return None
            
            
            changes = np.diff(prices)
            
            
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)
            
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            
            if rsi > self.thresholds['rsi_overbought']:
                signal_type = "sell"
                strength = SignalStrength.STRONG
            elif rsi < self.thresholds['rsi_oversold']:
                signal_type = "buy"
                strength = SignalStrength.STRONG
            else:
                signal_type = "hold"
                strength = SignalStrength.WEAK
            
            
            normalized_rsi = (rsi - 50) / 50
            
            return TechnicalSignal(
                indicator="RSI",
                signal_type=signal_type,
                strength=strength,
                value=normalized_rsi,
                timestamp=datetime.now(),
                metadata={"rsi_value": rsi, "period": period}
            )
            
        except Exception as e:
            self.logger.warning(f"RSI calculation failed: {e}")
            return None
    
    def _calculate_macd_signal(self, prices: List[float]) -> Optional[TechnicalSignal]:
        """Calculate MACD signal"""
        
        try:
            fast_period = self.indicators['macd_params']['fast_period']
            slow_period = self.indicators['macd_params']['slow_period']
            signal_period = self.indicators['macd_params']['signal_period']
            
            if len(prices) < slow_period:
                return None
            
            def calculate_ema(data, period):
                alpha = 2 / (period + 1)
                ema = [data[0]]
                for price in data[1:]:
                    ema.append(alpha * price + (1 - alpha) * ema[-1])
                return ema
            
            
            ema_fast = calculate_ema(prices, fast_period)
            ema_slow = calculate_ema(prices, slow_period)
            
            macd_line = np.array(ema_fast) - np.array(ema_slow)
            signal_line = calculate_ema(macd_line.tolist(), signal_period)
            
            
            current_macd = macd_line[-1]
            current_signal = signal_line[-1]
            
            if current_macd > current_signal:
                signal_type = "buy"
                strength = SignalStrength.MODERATE
                value = 0.4
            elif current_macd < current_signal:
                signal_type = "sell"
                strength = SignalStrength.MODERATE
                value = -0.4
            else:
                signal_type = "hold"
                strength = SignalStrength.WEAK
                value = 0.0
            
            return TechnicalSignal(
                indicator="MACD",
                signal_type=signal_type,
                strength=strength,
                value=value,
                timestamp=datetime.now(),
                metadata={
                    "macd": current_macd,
                    "signal": current_signal,
                    "fast_period": fast_period,
                    "slow_period": slow_period
                }
            )
            
        except Exception as e:
            self.logger.warning(f"MACD calculation failed: {e}")
            return None
    
    def _calculate_bollinger_signal(self, prices: List[float]) -> Optional[TechnicalSignal]:
        """Calculate Bollinger Bands signal"""
        
        try:
            period = self.indicators['bollinger_params']['period']
            std_dev = self.indicators['bollinger_params']['std_dev']
            
            if len(prices) < period:
                return None
            
            
            middle_band = np.mean(prices[-period:])
            
            
            std = np.std(prices[-period:])
            
            
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            current_price = prices[-1]
            
            
            if current_price > upper_band:
                signal_type = "sell"
                strength = SignalStrength.MODERATE
                value = -0.3
            elif current_price < lower_band:
                signal_type = "buy"
                strength = SignalStrength.MODERATE
                value = 0.3
            else:
                signal_type = "hold"
                strength = SignalStrength.WEAK
                value = 0.0
            
            return TechnicalSignal(
                indicator="BOLLINGER",
                signal_type=signal_type,
                strength=strength,
                value=value,
                timestamp=datetime.now(),
                metadata={
                    "upper_band": upper_band,
                    "middle_band": middle_band,
                    "lower_band": lower_band,
                    "current_price": current_price
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Bollinger Bands calculation failed: {e}")
            return None
    
    def _calculate_volume_signal(self, prices: List[float], volumes: List[float]) -> Optional[TechnicalSignal]:
        """Calculate volume-based signal"""
        
        try:
            if len(volumes) < 10 or len(prices) < 10:
                return None
            
            
            avg_volume = np.mean(volumes[-10:])
            current_volume = volumes[-1]
            
            
            price_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
            
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            
            if volume_ratio > 1.5 and price_change > 0:
                signal_type = "buy"
                strength = SignalStrength.STRONG
                value = 0.6
            elif volume_ratio > 1.5 and price_change < 0:
                signal_type = "sell"
                strength = SignalStrength.STRONG
                value = -0.6
            else:
                signal_type = "hold"
                strength = SignalStrength.WEAK
                value = 0.0
            
            return TechnicalSignal(
                indicator="VOLUME",
                signal_type=signal_type,
                strength=strength,
                value=value,
                timestamp=datetime.now(),
                metadata={
                    "volume_ratio": volume_ratio,
                    "price_change": price_change,
                    "current_volume": current_volume,
                    "avg_volume": avg_volume
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Volume signal calculation failed: {e}")
            return None
    
    def _analyze_trend(self, prices: List[float]) -> Tuple[TrendDirection, float]:
        """Analyze overall trend direction and strength"""
        
        try:
            if len(prices) < 10:
                return TrendDirection.SIDEWAYS, 0.0
            
            
            x = np.arange(len(prices))
            y = np.array(prices)
            
            
            slope, intercept = np.polyfit(x, y, 1)
            
            
            avg_price = np.mean(prices)
            normalized_slope = slope / avg_price if avg_price > 0 else 0
            
            
            if normalized_slope > 0.001:
                direction = TrendDirection.BULLISH
            elif normalized_slope < -0.001:
                direction = TrendDirection.BEARISH
            else:
                direction = TrendDirection.SIDEWAYS
            
            
            strength = min(abs(normalized_slope) * 1000, 1.0)
            
            return direction, strength
            
        except Exception as e:
            self.logger.warning(f"Trend analysis failed: {e}")
            return TrendDirection.SIDEWAYS, 0.0
    
    def _calculate_support_levels(self, prices: List[float]) -> List[float]:
        """Calculate support levels"""
        
        try:
            if len(prices) < 20:
                return []
            
            
            support_levels = []
            for i in range(1, len(prices) - 1):
                if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    support_levels.append(prices[i])
            
            
            return sorted(list(set(support_levels)))[-3:]  # Top 3 support levels
            
        except Exception as e:
            self.logger.warning(f"Support level calculation failed: {e}")
            return []
    
    def _calculate_resistance_levels(self, prices: List[float]) -> List[float]:
        """Calculate resistance levels"""
        
        try:
            if len(prices) < 20:
                return []
            
            
            resistance_levels = []
            for i in range(1, len(prices) - 1):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    resistance_levels.append(prices[i])
            
            
            return sorted(list(set(resistance_levels)), reverse=True)[:3]  # Top 3 resistance levels
            
        except Exception as e:
            self.logger.warning(f"Resistance level calculation failed: {e}")
            return []
    
    def _calculate_overall_score(self, signals: List[TechnicalSignal]) -> float:
        """Calculate overall technical score"""
        
        if not signals:
            return 0.0
        
        scores = []
        for signal in signals:
            if signal.signal_type == "buy":
                scores.append(signal.value)
            elif signal.signal_type == "sell":
                scores.append(signal.value)
            else:  # hold
                scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_confidence(self, price_data_length: int, signal_count: int) -> float:
        """Calculate confidence in the analysis"""
        
        
        data_confidence = min(price_data_length / 50.0, 1.0)
        
        
        signal_confidence = min(signal_count / 5.0, 1.0)
        
        
        return (data_confidence + signal_confidence) / 2.0
    
    def _create_default_analysis(self, token_address: str) -> TechnicalAnalysis:
        """Create default analysis when insufficient data"""
        
        return TechnicalAnalysis(
            token_address=token_address,
            timestamp=datetime.now(),
            trend_direction=TrendDirection.SIDEWAYS,
            trend_strength=0.0,
            support_levels=[],
            resistance_levels=[],
            signals=[],
            overall_score=0.0,
            confidence=0.0
        ) 