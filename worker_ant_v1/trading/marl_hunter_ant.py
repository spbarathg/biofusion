"""
MULTI-AGENT HUNTER ANT SWARM - MADDPG REINFORCEMENT LEARNING
===========================================================

Multi-Agent Deep Deterministic Policy Gradient (MADDPG) system for coordinated trading.
Implements cooperative learning where Hunter Ants share experiences and coordinate actions.

Key Features:
- Centralized Critic with Decentralized Actors (CTDE)
- Experience sharing across agents for faster learning
- Communication protocols for coordinated actions
- Cooperative reward structure promoting swarm success
- Advanced experience replay with prioritized sampling
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque, namedtuple
import pickle
import os
import random
from enum import Enum
import threading
import copy

from worker_ant_v1.utils.logger import setup_logger
from worker_ant_v1.core.message_bus import get_message_bus, MessageBus, MessageEnvelope, MessageType, MessagePriority

class ActionType(Enum):
    """Trading actions available to the Hunter Ant"""
    HOLD = 0
    BUY_25 = 1
    BUY_50 = 2
    BUY_100 = 3
    SELL_25 = 4
    SELL_50 = 5
    SELL_100 = 6

class CommunicationType(Enum):
    """Types of communication between agents"""
    MARKET_SIGNAL = "market_signal"
    POSITION_UPDATE = "position_update"
    RISK_ALERT = "risk_alert"
    OPPORTUNITY_SHARE = "opportunity_share"
    COORDINATION_REQUEST = "coordination_request"

@dataclass
class AgentCommunication:
    """Communication message between agents"""
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    communication_type: CommunicationType
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class MultiAgentMarketState:
    """Enhanced market state with multi-agent information"""
    # Individual agent state (same as before)
    current_price: float
    price_change_1m: float
    price_change_5m: float
    price_change_15m: float
    price_volatility: float
    current_volume: float
    volume_change_1m: float
    volume_change_5m: float
    volume_ma_ratio: float
    rsi: float
    macd: float
    bollinger_position: float
    moving_average_ratio: float
    sentiment_score: float
    sentiment_change: float
    current_position: float
    unrealized_pnl: float
    position_age: float
    liquidity_ratio: float
    holder_count_change: float
    market_cap: float
    
    # Multi-agent features
    swarm_total_position: float  # Total position across all agents
    swarm_avg_confidence: float  # Average confidence across agents
    agent_count_bullish: int     # Number of agents with bullish sentiment
    agent_count_bearish: int     # Number of agents with bearish sentiment
    coordination_signal: float   # Coordination signal from swarm
    communication_vector: List[float]  # Communication from other agents
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor"""
        individual_features = [
            self.current_price, self.price_change_1m, self.price_change_5m, self.price_change_15m,
            self.price_volatility, self.current_volume, self.volume_change_1m, self.volume_change_5m,
            self.volume_ma_ratio, self.rsi, self.macd, self.bollinger_position,
            self.moving_average_ratio, self.sentiment_score, self.sentiment_change,
            self.current_position, self.unrealized_pnl, self.position_age,
            self.liquidity_ratio, self.holder_count_change, self.market_cap
        ]
        
        # Multi-agent features
        swarm_features = [
            self.swarm_total_position, self.swarm_avg_confidence,
            float(self.agent_count_bullish), float(self.agent_count_bearish),
            self.coordination_signal
        ]
        
        # Communication vector (fixed size of 10)
        comm_features = self.communication_vector[:10] if len(self.communication_vector) >= 10 else \
                       self.communication_vector + [0.0] * (10 - len(self.communication_vector))
        
        all_features = individual_features + swarm_features + comm_features
        return torch.tensor(all_features, dtype=torch.float32)

# Experience tuple for MADDPG
Experience = namedtuple('Experience', [
    'states', 'actions', 'rewards', 'next_states', 'dones', 'communications'
])

class MADDPGActor(nn.Module):
    """Decentralized Actor network for MADDPG"""
    
    def __init__(self, state_dim: int = 36, action_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        
        # Individual state processing
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Communication processing
        self.comm_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim // 4),  # Communication vector size
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 4)
        )
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Policy layers
        self.policy_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # Communication output
        self.communication_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 10)  # Communication vector
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor, communication: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action probabilities and communication signal"""
        
        # Process individual state
        state_features = self.state_encoder(state)
        
        # Process communication
        comm_features = self.comm_encoder(communication)
        
        # Fuse features
        fused_features = torch.cat([state_features, comm_features], dim=-1)
        fused_features = self.fusion_layer(fused_features)
        
        # Generate action logits
        action_logits = self.policy_layers(fused_features)
        
        # Generate communication signal
        comm_signal = torch.tanh(self.communication_head(fused_features))
        
        return action_logits, comm_signal

class MADDPGCritic(nn.Module):
    """Centralized Critic network for MADDPG"""
    
    def __init__(self, num_agents: int = 10, state_dim: int = 36, action_dim: int = 7, hidden_dim: int = 512):
        super().__init__()
        
        self.num_agents = num_agents
        self.global_state_dim = state_dim * num_agents
        self.global_action_dim = action_dim * num_agents
        
        # Global state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Global action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(self.global_action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # Q-value estimation
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, global_states: torch.Tensor, global_actions: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-value for global state-action pair"""
        
        # Flatten global states and actions
        batch_size = global_states.shape[0]
        global_states = global_states.view(batch_size, -1)
        global_actions = global_actions.view(batch_size, -1)
        
        # Encode global state and actions
        state_features = self.state_encoder(global_states)
        action_features = self.action_encoder(global_actions)
        
        # Combine and estimate Q-value
        combined_features = torch.cat([state_features, action_features], dim=-1)
        q_value = self.q_network(combined_features)
        
        return q_value

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer for MADDPG"""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, experience: Experience):
        """Add experience to buffer"""
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Set priority to max for new experiences
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with priority-based sampling"""
        
        if len(self.buffer) < batch_size:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class MADDPGSwarm:
    """Multi-Agent Deep Deterministic Policy Gradient Swarm"""
    
    def __init__(self, num_agents: int = 10, state_dim: int = 36, action_dim: int = 7):
        self.logger = setup_logger("MADDPGSwarm")
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        self.gamma = 0.99
        self.tau = 0.01  # Soft update parameter
        self.batch_size = 256
        self.update_frequency = 100
        self.exploration_noise = 0.1
        
        # Networks
        self.actors = []
        self.target_actors = []
        self.critic = MADDPGCritic(num_agents, state_dim, action_dim)
        self.target_critic = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizers = []
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Initialize actors
        for i in range(num_agents):
            actor = MADDPGActor(state_dim, action_dim)
            target_actor = copy.deepcopy(actor)
            
            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.actor_optimizers.append(torch.optim.Adam(actor.parameters(), lr=self.lr_actor))
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
        
        # Communication system
        self.communication_buffers = {i: deque(maxlen=100) for i in range(num_agents)}
        self.communication_lock = threading.Lock()
        
        # Performance tracking
        self.training_step = 0
        self.episode_rewards = deque(maxlen=1000)
        self.cooperative_rewards = deque(maxlen=1000)
        
        # Message bus for distributed communication
        self.message_bus: Optional[MessageBus] = None
        
        self.logger.info(f"✅ MADDPG Swarm initialized with {num_agents} agents")
    
    async def initialize(self):
        """Initialize the MADDPG swarm"""
        try:
            # Initialize message bus for agent communication
            self.message_bus = await get_message_bus()
            
            # Subscribe to inter-agent communication
            await self.message_bus.subscribe(
                "agents.communication.*",
                self._handle_agent_communication,
                queue_group="maddpg_swarm"
            )
            
            self.logger.info("✅ MADDPG Swarm initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize MADDPG Swarm: {e}")
            raise
    
    async def _handle_agent_communication(self, message: MessageEnvelope):
        """Handle communication between agents"""
        try:
            comm_data = message.data
            sender_id = comm_data.get('sender_id')
            receiver_id = comm_data.get('receiver_id')
            communication = AgentCommunication(**comm_data['communication'])
            
            if receiver_id is None:  # Broadcast
                for agent_id in range(self.num_agents):
                    if agent_id != sender_id:
                        self._add_communication(agent_id, communication)
            else:
                self._add_communication(receiver_id, communication)
                
        except Exception as e:
            self.logger.error(f"❌ Error handling agent communication: {e}")
    
    def _add_communication(self, agent_id: int, communication: AgentCommunication):
        """Add communication to agent's buffer"""
        with self.communication_lock:
            self.communication_buffers[agent_id].append(communication)
    
    def _get_communication_vector(self, agent_id: int) -> List[float]:
        """Get communication vector for agent"""
        with self.communication_lock:
            comm_buffer = self.communication_buffers[agent_id]
            
            if not comm_buffer:
                return [0.0] * 10  # Empty communication
            
            # Process recent communications
            recent_comms = list(comm_buffer)[-5:]  # Last 5 communications
            
            # Encode communications into vector
            comm_vector = [0.0] * 10
            
            for i, comm in enumerate(recent_comms):
                if i >= 5:
                    break
                
                # Encode communication type
                comm_vector[i * 2] = float(list(CommunicationType).index(comm.communication_type)) / len(CommunicationType)
                
                # Encode priority and content
                content_value = 0.0
                if 'confidence' in comm.content:
                    content_value = comm.content['confidence']
                elif 'signal_strength' in comm.content:
                    content_value = comm.content['signal_strength']
                
                comm_vector[i * 2 + 1] = content_value
            
            return comm_vector
    
    async def act(self, agent_id: int, state: MultiAgentMarketState, training: bool = True) -> Tuple[int, torch.Tensor]:
        """Get action for specific agent"""
        try:
            # Get communication vector
            comm_vector = self._get_communication_vector(agent_id)
            comm_tensor = torch.tensor(comm_vector, dtype=torch.float32).unsqueeze(0)
            
            # Get state tensor
            state_tensor = state.to_tensor().unsqueeze(0)
            
            # Get action from actor
            actor = self.actors[agent_id]
            with torch.no_grad():
                action_logits, comm_signal = actor(state_tensor, comm_tensor)
                
                if training:
                    # Add exploration noise
                    noise = torch.randn_like(action_logits) * self.exploration_noise
                    action_logits = action_logits + noise
                
                # Convert to discrete action
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()
            
            # Broadcast communication signal if significant
            if torch.abs(comm_signal).max() > 0.5:
                await self._broadcast_communication(agent_id, comm_signal, state)
            
            return action, comm_signal
            
        except Exception as e:
            self.logger.error(f"❌ Action selection failed for agent {agent_id}: {e}")
            return ActionType.HOLD.value, torch.zeros(10)
    
    async def _broadcast_communication(self, agent_id: int, comm_signal: torch.Tensor, state: MultiAgentMarketState):
        """Broadcast communication signal to other agents"""
        try:
            # Interpret communication signal
            signal_values = comm_signal.squeeze().tolist()
            
            # Determine communication type and content
            signal_strength = max(signal_values)
            if signal_strength > 0.8:
                comm_type = CommunicationType.OPPORTUNITY_SHARE
                content = {
                    'confidence': signal_strength,
                    'price_prediction': state.current_price * (1 + signal_values[0] * 0.1),
                    'urgency': signal_strength
                }
            elif signal_strength < -0.8:
                comm_type = CommunicationType.RISK_ALERT
                content = {
                    'risk_level': abs(signal_strength),
                    'alert_type': 'market_downturn',
                    'recommended_action': 'reduce_exposure'
                }
            else:
                comm_type = CommunicationType.MARKET_SIGNAL
                content = {
                    'signal_strength': signal_strength,
                    'market_sentiment': 'bullish' if signal_strength > 0 else 'bearish'
                }
            
            # Create communication
            communication = AgentCommunication(
                sender_id=str(agent_id),
                receiver_id=None,  # Broadcast
                communication_type=comm_type,
                content=content,
                timestamp=datetime.utcnow(),
                priority=1 if abs(signal_strength) > 0.7 else 2
            )
            
            # Send via message bus
            if self.message_bus:
                message = MessageEnvelope(
                    message_id=f"agent_comm_{agent_id}_{int(datetime.utcnow().timestamp())}",
                    message_type=MessageType.SQUAD_SIGNAL,
                    subject="agents.communication.broadcast",
                    sender_id=f"agent_{agent_id}",
                    timestamp=datetime.utcnow(),
                    priority=MessagePriority.MEDIUM,
                    data={
                        'sender_id': agent_id,
                        'receiver_id': None,
                        'communication': {
                            'sender_id': communication.sender_id,
                            'receiver_id': communication.receiver_id,
                            'communication_type': communication.communication_type,
                            'content': communication.content,
                            'timestamp': communication.timestamp,
                            'priority': communication.priority
                        }
                    }
                )
                await self.message_bus.publish(message)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to broadcast communication: {e}")
    
    def add_experience(self, states: List[MultiAgentMarketState], actions: List[int], 
                      rewards: List[float], next_states: List[MultiAgentMarketState], 
                      dones: List[bool], communications: List[List[float]]):
        """Add multi-agent experience to replay buffer"""
        
        experience = Experience(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            communications=communications
        )
        
        self.replay_buffer.add(experience)
    
    async def update(self):
        """Update all agents using MADDPG"""
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        try:
            # Sample batch
            experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
            
            if not experiences:
                return
            
            # Convert to tensors
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            batch_dones = []
            batch_communications = []
            
            for exp in experiences:
                # Stack states across agents
                states_tensor = torch.stack([state.to_tensor() for state in exp.states])
                next_states_tensor = torch.stack([state.to_tensor() for state in exp.next_states])
                
                batch_states.append(states_tensor)
                batch_next_states.append(next_states_tensor)
                batch_actions.append(torch.tensor(exp.actions, dtype=torch.long))
                batch_rewards.append(torch.tensor(exp.rewards, dtype=torch.float32))
                batch_dones.append(torch.tensor(exp.dones, dtype=torch.float32))
                batch_communications.append(torch.tensor(exp.communications, dtype=torch.float32))
            
            # Stack batches
            batch_states = torch.stack(batch_states)  # (batch_size, num_agents, state_dim)
            batch_next_states = torch.stack(batch_next_states)
            batch_actions = torch.stack(batch_actions)  # (batch_size, num_agents)
            batch_rewards = torch.stack(batch_rewards)  # (batch_size, num_agents)
            batch_dones = torch.stack(batch_dones)
            batch_communications = torch.stack(batch_communications)  # (batch_size, num_agents, comm_dim)
            
            # Update critic
            critic_loss = await self._update_critic(
                batch_states, batch_actions, batch_rewards, 
                batch_next_states, batch_dones, batch_communications, weights
            )
            
            # Update actors
            actor_losses = []
            for agent_id in range(self.num_agents):
                actor_loss = await self._update_actor(
                    agent_id, batch_states, batch_communications, weights
                )
                actor_losses.append(actor_loss)
            
            # Update target networks
            self._soft_update_targets()
            
            # Update priorities in replay buffer
            with torch.no_grad():
                td_errors = self._calculate_td_errors(
                    batch_states, batch_actions, batch_rewards, 
                    batch_next_states, batch_dones, batch_communications
                )
                priorities = (td_errors.abs() + 1e-6).cpu().numpy()
                self.replay_buffer.update_priorities(indices, priorities)
            
            self.training_step += 1
            
            # Log training progress
            if self.training_step % 100 == 0:
                self.logger.info(
                    f"Training Step {self.training_step}: "
                    f"Critic Loss: {critic_loss:.4f}, "
                    f"Avg Actor Loss: {np.mean(actor_losses):.4f}"
                )
                
        except Exception as e:
            self.logger.error(f"❌ MADDPG update failed: {e}")
    
    async def _update_critic(self, states: torch.Tensor, actions: torch.Tensor, 
                           rewards: torch.Tensor, next_states: torch.Tensor, 
                           dones: torch.Tensor, communications: torch.Tensor, 
                           weights: np.ndarray) -> float:
        """Update centralized critic"""
        
        batch_size = states.shape[0]
        
        # Get current Q-values
        current_q = self.critic(states, F.one_hot(actions, self.action_dim).float())
        
        # Calculate target Q-values
        with torch.no_grad():
            # Get next actions from target actors
            next_actions = []
            for agent_id in range(self.num_agents):
                next_action_logits, _ = self.target_actors[agent_id](
                    next_states[:, agent_id], communications[:, agent_id]
                )
                next_action_probs = F.softmax(next_action_logits, dim=-1)
                next_actions.append(next_action_probs)
            
            next_actions_tensor = torch.stack(next_actions, dim=1)  # (batch_size, num_agents, action_dim)
            
            # Calculate target Q-values
            target_q = self.target_critic(next_states, next_actions_tensor)
            
            # Calculate cooperative reward (sum of individual rewards)
            cooperative_rewards = torch.sum(rewards, dim=1, keepdim=True)
            target_q = cooperative_rewards + self.gamma * target_q * (1 - torch.max(dones, dim=1, keepdim=True)[0])
        
        # Calculate critic loss with importance sampling
        td_errors = target_q - current_q
        weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        critic_loss = torch.mean(weights_tensor * td_errors.pow(2))
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    async def _update_actor(self, agent_id: int, states: torch.Tensor, 
                          communications: torch.Tensor, weights: np.ndarray) -> float:
        """Update individual actor"""
        
        # Get current actions from all actors
        current_actions = []
        for i in range(self.num_agents):
            if i == agent_id:
                # Use the actor we're updating
                action_logits, _ = self.actors[i](states[:, i], communications[:, i])
                action_probs = F.softmax(action_logits, dim=-1)
                current_actions.append(action_probs)
            else:
                # Use other actors with detached gradients
                with torch.no_grad():
                    action_logits, _ = self.actors[i](states[:, i], communications[:, i])
                    action_probs = F.softmax(action_logits, dim=-1)
                    current_actions.append(action_probs)
        
        current_actions_tensor = torch.stack(current_actions, dim=1)
        
        # Calculate Q-value for these actions
        q_value = self.critic(states, current_actions_tensor)
        
        # Actor loss (negative Q-value to maximize)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        actor_loss = -torch.mean(weights_tensor * q_value)
        
        # Update actor
        self.actor_optimizers[agent_id].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), max_norm=10.0)
        self.actor_optimizers[agent_id].step()
        
        return actor_loss.item()
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        
        # Update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update target actors
        for agent_id in range(self.num_agents):
            for target_param, param in zip(self.target_actors[agent_id].parameters(), 
                                         self.actors[agent_id].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _calculate_td_errors(self, states: torch.Tensor, actions: torch.Tensor, 
                           rewards: torch.Tensor, next_states: torch.Tensor, 
                           dones: torch.Tensor, communications: torch.Tensor) -> torch.Tensor:
        """Calculate TD errors for priority updates"""
        
        with torch.no_grad():
            # Current Q-values
            current_q = self.critic(states, F.one_hot(actions, self.action_dim).float())
            
            # Target Q-values
            next_actions = []
            for agent_id in range(self.num_agents):
                next_action_logits, _ = self.target_actors[agent_id](
                    next_states[:, agent_id], communications[:, agent_id]
                )
                next_action_probs = F.softmax(next_action_logits, dim=-1)
                next_actions.append(next_action_probs)
            
            next_actions_tensor = torch.stack(next_actions, dim=1)
            target_q = self.target_critic(next_states, next_actions_tensor)
            
            cooperative_rewards = torch.sum(rewards, dim=1, keepdim=True)
            target_q = cooperative_rewards + self.gamma * target_q * (1 - torch.max(dones, dim=1, keepdim=True)[0])
            
            td_errors = target_q - current_q
        
        return td_errors.squeeze()
    
    def save_models(self, base_path: str):
        """Save all models"""
        try:
            os.makedirs(base_path, exist_ok=True)
            
            # Save critic
            torch.save({
                'model_state_dict': self.critic.state_dict(),
                'optimizer_state_dict': self.critic_optimizer.state_dict(),
                'training_step': self.training_step
            }, f"{base_path}/critic.pth")
            
            # Save actors
            for agent_id in range(self.num_agents):
                torch.save({
                    'model_state_dict': self.actors[agent_id].state_dict(),
                    'optimizer_state_dict': self.actor_optimizers[agent_id].state_dict(),
                    'agent_id': agent_id
                }, f"{base_path}/actor_{agent_id}.pth")
            
            self.logger.info(f"✅ Saved MADDPG models to {base_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save models: {e}")
    
    def load_models(self, base_path: str):
        """Load all models"""
        try:
            # Load critic
            critic_path = f"{base_path}/critic.pth"
            if os.path.exists(critic_path):
                checkpoint = torch.load(critic_path, map_location='cpu')
                self.critic.load_state_dict(checkpoint['model_state_dict'])
                self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_step = checkpoint.get('training_step', 0)
            
            # Load actors
            for agent_id in range(self.num_agents):
                actor_path = f"{base_path}/actor_{agent_id}.pth"
                if os.path.exists(actor_path):
                    checkpoint = torch.load(actor_path, map_location='cpu')
                    self.actors[agent_id].load_state_dict(checkpoint['model_state_dict'])
                    self.actor_optimizers[agent_id].load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Update target networks
            self.target_critic = copy.deepcopy(self.critic)
            self.target_actors = [copy.deepcopy(actor) for actor in self.actors]
            
            self.logger.info(f"✅ Loaded MADDPG models from {base_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load models: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'training_step': self.training_step,
            'num_agents': self.num_agents,
            'replay_buffer_size': len(self.replay_buffer),
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'avg_cooperative_reward': np.mean(self.cooperative_rewards) if self.cooperative_rewards else 0.0,
            'communication_activity': {
                agent_id: len(buffer) for agent_id, buffer in self.communication_buffers.items()
            }
        }


# Factory function for creating MARL Hunter Ants
async def create_marl_hunter_swarm(num_agents: int = 10) -> MADDPGSwarm:
    """Create and initialize MARL Hunter Ant swarm"""
    swarm = MADDPGSwarm(num_agents=num_agents)
    await swarm.initialize()
    return swarm 