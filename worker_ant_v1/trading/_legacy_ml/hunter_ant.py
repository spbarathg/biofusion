"""
HUNTER ANT - REINFORCEMENT LEARNING TRADING AGENT
================================================

Reinforcement Learning agent that learns optimal trading policies through
trial and error in simulated market environments. Embodies the "digital predator" concept.

Key Features:
- Proximal Policy Optimization (PPO) for stable learning
- Sophisticated reward function encouraging survival and profit
- Multi-wallet deployment with genetic evolution
- Real-time adaptation to market conditions
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
import random
from enum import Enum

from worker_ant_v1.utils.logger import setup_logger

class ActionType(Enum):
    """Trading actions available to the Hunter Ant"""
    HOLD = 0
    BUY_25 = 1
    BUY_50 = 2
    BUY_100 = 3
    SELL_25 = 4
    SELL_50 = 5
    SELL_100 = 6

@dataclass
class MarketState:
    """Market state representation for RL agent"""
    # Price features
    current_price: float
    price_change_1m: float
    price_change_5m: float
    price_change_15m: float
    price_volatility: float
    
    # Volume features
    current_volume: float
    volume_change_1m: float
    volume_change_5m: float
    volume_ma_ratio: float
    
    # Technical features
    rsi: float
    macd: float
    bollinger_position: float
    moving_average_ratio: float
    
    # Sentiment features
    sentiment_score: float
    sentiment_change: float
    
    # Position features
    current_position: float  # -1 to 1 (short to long)
    unrealized_pnl: float
    position_age: float
    
    # Market context
    liquidity_ratio: float
    holder_count_change: float
    market_cap: float
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor"""
        features = [
            self.current_price, self.price_change_1m, self.price_change_5m, self.price_change_15m,
            self.price_volatility, self.current_volume, self.volume_change_1m, self.volume_change_5m,
            self.volume_ma_ratio, self.rsi, self.macd, self.bollinger_position,
            self.moving_average_ratio, self.sentiment_score, self.sentiment_change,
            self.current_position, self.unrealized_pnl, self.position_age,
            self.liquidity_ratio, self.holder_count_change, self.market_cap
        ]
        return torch.tensor(features, dtype=torch.float32)

@dataclass
class TradingEpisode:
    """Complete trading episode for training"""
    states: List[MarketState]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    episode_return: float
    episode_length: int
    final_pnl: float
    sharpe_ratio: float

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_dim: int = 21, action_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value"""
        features = self.feature_extractor(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, float, float]:
        """Get action, log probability, and value for a state"""
        action_logits, value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

class PPOMemory:
    """Memory buffer for PPO training"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def add(self, state: torch.Tensor, action: int, reward: float, 
            value: float, log_prob: float, done: bool):
        """Add experience to memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
        if len(self.states) > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
            self.dones.pop(0)
    
    def clear(self):
        """Clear memory"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get batch of experiences"""
        return (torch.stack(self.states), torch.tensor(self.actions),
                torch.tensor(self.rewards), torch.tensor(self.values),
                torch.tensor(self.log_probs), torch.tensor(self.dones))

class MarketEnvironment:
    """Simulated market environment for RL training"""
    
    def __init__(self, initial_balance: float = 1.0):
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self) -> MarketState:
        """Reset environment to initial state"""
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_entry_price = 0.0
        self.position_age = 0.0
        self.total_trades = 0
        self.successful_trades = 0
        self.episode_return = 0.0
        
        # Initialize with neutral market state
        return MarketState(
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
            market_cap=1000000.0
        )
    
    def step(self, action: int, market_data: Dict[str, Any]) -> Tuple[MarketState, float, bool]:
        """Take action and return new state, reward, and done flag"""
        
        # Update market state from real data
        state = self._update_market_state(market_data)
        
        # Execute action
        reward = self._execute_action(action, state)
        
        # Update position age
        if self.position != 0:
            self.position_age += 1
        
        # Check if episode should end
        done = self._check_episode_end(state)
        
        return state, reward, done
    
    def _update_market_state(self, market_data: Dict[str, Any]) -> MarketState:
        """Update market state from real market data"""
        
        current_price = market_data.get('current_price', 1.0)
        
        # Calculate price changes
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
        
        # Sentiment
        sentiment_score = market_data.get('sentiment_score', 0.0)
        sentiment_history = market_data.get('sentiment_history', [])
        if len(sentiment_history) >= 2:
            sentiment_change = sentiment_score - sentiment_history[-2].get('score', sentiment_score)
        else:
            sentiment_change = 0.0
        
        # Calculate unrealized PnL
        if self.position != 0 and self.position_entry_price > 0:
            unrealized_pnl = self.position * (current_price - self.position_entry_price) / self.position_entry_price
        else:
            unrealized_pnl = 0.0
        
        return MarketState(
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
            current_position=self.position,
            unrealized_pnl=unrealized_pnl,
            position_age=self.position_age,
            liquidity_ratio=market_data.get('liquidity_ratio', 1.0),
            holder_count_change=market_data.get('holder_count_change', 0.0),
            market_cap=market_data.get('market_cap', 1000000.0)
        )
    
    def _execute_action(self, action: int, state: MarketState) -> float:
        """Execute trading action and calculate reward"""
        
        current_price = state.current_price
        transaction_fee = 0.003  # 0.3% transaction fee
        
        if action == ActionType.HOLD.value:
            # Small reward for holding during good conditions
            reward = state.unrealized_pnl * 0.1
            return reward
        
        elif action in [ActionType.BUY_25.value, ActionType.BUY_50.value, ActionType.BUY_100.value]:
            # Buy actions
            if self.position >= 1.0:  # Already fully long
                return -0.01  # Penalty for invalid action
            
            # Calculate buy amount
            if action == ActionType.BUY_25.value:
                buy_amount = 0.25
            elif action == ActionType.BUY_50.value:
                buy_amount = 0.5
            else:  # BUY_100
                buy_amount = 1.0 - self.position
            
            # Execute buy
            old_position = self.position
            self.position = min(1.0, self.position + buy_amount)
            
            # Update entry price
            if old_position == 0:
                self.position_entry_price = current_price
            else:
                # Weighted average entry price
                self.position_entry_price = (old_position * self.position_entry_price + 
                                           buy_amount * current_price) / self.position
            
            # Calculate reward
            reward = -transaction_fee * buy_amount  # Transaction cost
            return reward
        
        elif action in [ActionType.SELL_25.value, ActionType.SELL_50.value, ActionType.SELL_100.value]:
            # Sell actions
            if self.position <= 0:  # No position to sell
                return -0.01  # Penalty for invalid action
            
            # Calculate sell amount
            if action == ActionType.SELL_25.value:
                sell_amount = 0.25
            elif action == ActionType.SELL_50.value:
                sell_amount = 0.5
            else:  # SELL_100
                sell_amount = self.position
            
            # Execute sell
            self.position = max(0, self.position - sell_amount)
            
            # Calculate realized PnL
            if self.position_entry_price > 0:
                realized_pnl = sell_amount * (current_price - self.position_entry_price) / self.position_entry_price
            else:
                realized_pnl = 0
            
            # Update entry price if fully sold
            if self.position == 0:
                self.position_entry_price = 0
                self.position_age = 0
            
            # Calculate reward
            reward = realized_pnl - transaction_fee * sell_amount
            
            # Update trade statistics
            self.total_trades += 1
            if realized_pnl > 0:
                self.successful_trades += 1
            
            return reward
        
        return 0.0
    
    def _check_episode_end(self, state: MarketState) -> bool:
        """Check if episode should end"""
        
        # End if balance is too low
        if self.balance < 0.1:
            return True
        
        # End if too many consecutive losses
        if self.total_trades >= 50:
            return True
        
        # End if position age is too high (prevent overholding)
        if self.position_age > 100:
            return True
        
        return False

class HunterAnt:
    """Hunter Ant - RL trading agent using PPO"""
    
    def __init__(self, wallet_id: str, model_path: Optional[str] = None):
        self.logger = setup_logger(f"HunterAnt_{wallet_id}")
        self.wallet_id = wallet_id
        
        # PPO hyperparameters
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4
        
        # Network
        self.actor_critic = ActorCritic(state_dim=21, action_dim=7, hidden_dim=128)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        
        # Memory
        self.memory = PPOMemory(max_size=10000)
        
        # Environment
        self.env = MarketEnvironment(initial_balance=1.0)
        
        # Performance tracking
        self.episode_returns = deque(maxlen=100)
        self.total_episodes = 0
        self.best_return = -float('inf')
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.logger.info(f"✅ Hunter Ant {wallet_id} initialized")
    
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_return = checkpoint.get('best_return', self.best_return)
            self.total_episodes = checkpoint.get('total_episodes', self.total_episodes)
            self.logger.info(f"✅ Loaded Hunter Ant model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def save_model(self, model_path: str):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'model_state_dict': self.actor_critic.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_return': self.best_return,
                'total_episodes': self.total_episodes,
                'wallet_id': self.wallet_id,
                'timestamp': datetime.now()
            }
            torch.save(checkpoint, model_path)
            self.logger.info(f"✅ Saved Hunter Ant model to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    async def act(self, market_data: Dict[str, Any]) -> Tuple[int, float]:
        """Choose action for current market state"""
        
        try:
            # Get current state
            state = self.env._update_market_state(market_data)
            state_tensor = state.to_tensor().unsqueeze(0)
            
            # Get action from policy
            action, log_prob, value = self.actor_critic.get_action(state_tensor)
            
            # Store experience
            self.memory.add(state_tensor.squeeze(0), action, 0.0, value, log_prob, False)
            
            return action, value
            
        except Exception as e:
            self.logger.error(f"Action selection failed: {e}")
            return ActionType.HOLD.value, 0.0
    
    async def update(self, reward: float, done: bool):
        """Update agent with reward and done signal"""
        
        try:
            # Update last experience with reward
            if len(self.memory.rewards) > 0:
                self.memory.rewards[-1] = reward
                self.memory.dones[-1] = done
            
            # If episode is done, train the agent
            if done:
                await self._train_episode()
                self.env.reset()
                
        except Exception as e:
            self.logger.error(f"Update failed: {e}")
    
    async def _train_episode(self):
        """Train agent on completed episode"""
        
        try:
            if len(self.memory.states) < 100:  # Need minimum batch size
                return
            
            # Calculate returns and advantages
            returns, advantages = self._compute_returns_and_advantages()
            
            # Get batch
            states, actions, old_log_probs, dones = self.memory.get_batch()[:4]
            
            # PPO training
            for epoch in range(self.ppo_epochs):
                # Forward pass
                action_logits, values = self.actor_critic(states)
                action_probs = F.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                log_probs = action_dist.log_prob(actions)
                
                # Calculate ratios
                ratios = torch.exp(log_probs - old_log_probs)
                
                # PPO loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), returns)
                
                # Entropy bonus
                entropy = action_dist.entropy().mean()
                
                # Total loss
                loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            # Clear memory
            self.memory.clear()
            
            # Update episode statistics
            self.total_episodes += 1
            episode_return = returns.mean().item()
            self.episode_returns.append(episode_return)
            
            if episode_return > self.best_return:
                self.best_return = episode_return
                # Save best model
                self.save_model(f"models/hunter_ant_{self.wallet_id}_best.pth")
            
            if self.total_episodes % 10 == 0:
                avg_return = np.mean(list(self.episode_returns))
                self.logger.info(f"Episode {self.total_episodes}: Avg Return = {avg_return:.4f}, "
                               f"Best = {self.best_return:.4f}")
                
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
    
    def _compute_returns_and_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and advantages using GAE"""
        
        states, actions, rewards, values, log_probs, dones = self.memory.get_batch()
        
        # Convert to numpy for easier computation
        rewards = rewards.numpy()
        values = values.numpy()
        dones = dones.numpy()
        
        # Compute returns using GAE
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'wallet_id': self.wallet_id,
            'total_episodes': self.total_episodes,
            'best_return': self.best_return,
            'avg_return': np.mean(list(self.episode_returns)) if self.episode_returns else 0.0,
            'recent_returns': list(self.episode_returns)[-10:],
            'model_status': 'active'
        }
    
    def clone_genetics(self, source_agent: 'HunterAnt'):
        """Clone genetics from another agent (for evolution)"""
        try:
            self.actor_critic.load_state_dict(source_agent.actor_critic.state_dict())
            self.best_return = source_agent.best_return
            self.logger.info(f"✅ Cloned genetics from {source_agent.wallet_id}")
        except Exception as e:
            self.logger.error(f"Failed to clone genetics: {e}")
    
    def mutate_genetics(self, mutation_rate: float = 0.1):
        """Mutate model weights (for evolution)"""
        try:
            with torch.no_grad():
                for param in self.actor_critic.parameters():
                    if random.random() < mutation_rate:
                        noise = torch.randn_like(param) * 0.01
                        param.add_(noise)
            self.logger.info(f"✅ Mutated genetics with rate {mutation_rate}")
        except Exception as e:
            self.logger.error(f"Failed to mutate genetics: {e}") 