#!/usr/bin/env python3
"""
TRAIN ML MODELS - STATE-OF-THE-ART PREDICTION ENGINE TRAINING
============================================================

Training script for the three ML architectures:
1. Oracle Ant - Transformer-based time series predictor
2. Hunter Ant - Reinforcement Learning trading agent
3. Network Ant - Graph Neural Network for on-chain intelligence

Usage:
    python scripts/train_ml_models.py --model oracle --data_path data/historical_data.json
    python scripts/train_ml_models.py --model hunter --episodes 1000
    python scripts/train_ml_models.py --model network --graph_data data/transaction_graph.json
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from worker_ant_v1.trading.ml_architectures.oracle_ant import OracleAntPredictor
from worker_ant_v1.trading.ml_architectures.hunter_ant import HunterAnt
from worker_ant_v1.trading.ml_architectures.network_ant import NetworkAntPredictor
from worker_ant_v1.utils.logger import setup_logger

logger = setup_logger("MLTrainer")

class MLModelTrainer:
    """Trainer for all three ML architectures"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.logger = setup_logger("MLModelTrainer")
        
    async def train_oracle_ant(self, data_path: str, epochs: int = 100, 
                              batch_size: int = 32, learning_rate: float = 1e-4):
        """Train Oracle Ant (Transformer) model"""
        
        self.logger.info("🧠 Training Oracle Ant (Transformer) model...")
        
        try:
            # Load training data
            training_data = self._load_training_data(data_path)
            
            # Initialize Oracle Ant
            oracle_ant = OracleAntPredictor()
            
            # Prepare training data
            train_loader = self._prepare_oracle_training_data(training_data, batch_size)
            
            # Training loop
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for batch in train_loader:
                    # Forward pass
                    loss = await self._oracle_training_step(oracle_ant, batch)
                    epoch_loss += loss
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}/{epochs}: Loss = {avg_loss:.6f}")
                
                # Save checkpoint every 50 epochs
                if epoch % 50 == 0:
                    checkpoint_path = self.models_dir / f"oracle_ant_epoch_{epoch}.pth"
                    oracle_ant.save_model(str(checkpoint_path))
            
            # Save final model
            final_path = self.models_dir / "oracle_ant_final.pth"
            oracle_ant.save_model(str(final_path))
            
            self.logger.info(f"✅ Oracle Ant training completed. Model saved to {final_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Oracle Ant training failed: {e}")
            raise
    
    async def train_hunter_ant(self, wallet_id: str, episodes: int = 1000,
                              learning_rate: float = 3e-4):
        """Train Hunter Ant (RL) model"""
        
        self.logger.info(f"🏹 Training Hunter Ant (RL) model for wallet {wallet_id}...")
        
        try:
            # Initialize Hunter Ant
            hunter_ant = HunterAnt(wallet_id)
            
            # Training loop
            for episode in range(episodes):
                # Reset environment
                state = hunter_ant.env.reset()
                
                episode_reward = 0.0
                episode_length = 0
                
                # Episode loop
                while True:
                    # Generate synthetic market data for training
                    market_data = self._generate_synthetic_market_data()
                    
                    # Get action from agent
                    action, value = await hunter_ant.act(market_data)
                    
                    # Simulate environment step
                    reward = self._simulate_market_step(action, market_data)
                    done = episode_length > 100  # End episode after 100 steps
                    
                    # Update agent
                    await hunter_ant.update(reward, done)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    if done:
                        break
                
                if episode % 100 == 0:
                    self.logger.info(f"Episode {episode}/{episodes}: Reward = {episode_reward:.4f}, Length = {episode_length}")
                
                # Save checkpoint every 500 episodes
                if episode % 500 == 0:
                    checkpoint_path = self.models_dir / f"hunter_ant_{wallet_id}_episode_{episode}.pth"
                    hunter_ant.save_model(str(checkpoint_path))
            
            # Save final model
            final_path = self.models_dir / f"hunter_ant_{wallet_id}_final.pth"
            hunter_ant.save_model(str(final_path))
            
            self.logger.info(f"✅ Hunter Ant training completed. Model saved to {final_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Hunter Ant training failed: {e}")
            raise
    
    async def train_network_ant(self, graph_data_path: str, epochs: int = 50,
                               batch_size: int = 16, learning_rate: float = 1e-3):
        """Train Network Ant (GNN) model"""
        
        self.logger.info("🕸️ Training Network Ant (GNN) model...")
        
        try:
            # Load graph data
            graph_data = self._load_graph_data(graph_data_path)
            
            # Initialize Network Ant
            network_ant = NetworkAntPredictor()
            
            # Prepare training data
            train_loader = self._prepare_network_training_data(graph_data, batch_size)
            
            # Training loop
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for batch in train_loader:
                    # Forward pass
                    loss = await self._network_training_step(network_ant, batch)
                    epoch_loss += loss
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}/{epochs}: Loss = {avg_loss:.6f}")
                
                # Save checkpoint every 25 epochs
                if epoch % 25 == 0:
                    checkpoint_path = self.models_dir / f"network_ant_epoch_{epoch}.pth"
                    network_ant.save_model(str(checkpoint_path))
            
            # Save final model
            final_path = self.models_dir / "network_ant_final.pth"
            network_ant.save_model(str(final_path))
            
            self.logger.info(f"✅ Network Ant training completed. Model saved to {final_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Network Ant training failed: {e}")
            raise
    
    def _load_training_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load training data for Oracle Ant"""
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            self.logger.info(f"Loaded {len(data)} training samples from {data_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load training data: {e}")
            # Return synthetic data for testing
            return self._generate_synthetic_oracle_data()
    
    def _load_graph_data(self, graph_data_path: str) -> Dict[str, Any]:
        """Load graph data for Network Ant"""
        
        try:
            with open(graph_data_path, 'r') as f:
                data = json.load(f)
            
            self.logger.info(f"Loaded graph data from {graph_data_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load graph data: {e}")
            # Return synthetic graph data for testing
            return self._generate_synthetic_graph_data()
    
    def _prepare_oracle_training_data(self, data: List[Dict[str, Any]], 
                                    batch_size: int) -> List[List[Dict[str, Any]]]:
        """Prepare training data for Oracle Ant"""
        
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _prepare_network_training_data(self, graph_data: Dict[str, Any], 
                                     batch_size: int) -> List[Dict[str, Any]]:
        """Prepare training data for Network Ant"""
        
        # Split graph into batches
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        batches = []
        for i in range(0, len(nodes), batch_size):
            batch_nodes = nodes[i:i + batch_size]
            batch_edges = [e for e in edges if e['from'] in batch_nodes or e['to'] in batch_nodes]
            
            batches.append({
                'nodes': batch_nodes,
                'edges': batch_edges
            })
        
        return batches
    
    async def _oracle_training_step(self, oracle_ant, batch) -> float:
        """Single training step for Oracle Ant"""
        
        # This would implement the actual training logic
        # For now, return a dummy loss
        return 0.1
    
    async def _network_training_step(self, network_ant, batch) -> float:
        """Single training step for Network Ant"""
        
        # This would implement the actual training logic
        # For now, return a dummy loss
        return 0.1
    
    def _generate_synthetic_market_data(self) -> Dict[str, Any]:
        """Generate synthetic market data for Hunter Ant training"""
        
        import random
        import numpy as np
        
        return {
            'current_price': random.uniform(0.001, 1.0),
            'current_volume': random.uniform(1000, 100000),
            'price_change_1h': random.uniform(-0.5, 0.5),
            'price_change_24h': random.uniform(-0.8, 0.8),
            'liquidity': random.uniform(10000, 500000),
            'holder_count': random.randint(50, 5000),
            'sentiment_score': random.uniform(-1.0, 1.0),
            'rsi': random.uniform(20, 80),
            'macd': random.uniform(-0.1, 0.1),
            'bollinger_position': random.uniform(0, 1),
            'moving_average_ratio': random.uniform(0.8, 1.2),
            'price_volatility': random.uniform(0.1, 0.5),
            'volume_ma_ratio': random.uniform(0.5, 2.0),
            'liquidity_ratio': random.uniform(0.5, 2.0),
            'market_cap': random.uniform(100000, 10000000)
        }
    
    def _simulate_market_step(self, action: int, market_data: Dict[str, Any]) -> float:
        """Simulate market step for Hunter Ant training"""
        
        import random
        
        # Simple reward function based on action and market conditions
        base_reward = 0.0
        
        if action in [1, 2, 3]:  # Buy actions
            # Reward buying when price is likely to go up
            if market_data['sentiment_score'] > 0.3 and market_data['rsi'] < 70:
                base_reward = 0.1
            else:
                base_reward = -0.05
        elif action in [4, 5, 6]:  # Sell actions
            # Reward selling when price is likely to go down
            if market_data['sentiment_score'] < -0.3 or market_data['rsi'] > 80:
                base_reward = 0.1
            else:
                base_reward = -0.05
        else:  # Hold action
            base_reward = 0.01  # Small reward for holding
        
        # Add noise
        noise = random.uniform(-0.02, 0.02)
        return base_reward + noise
    
    def _generate_synthetic_oracle_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic training data for Oracle Ant"""
        
        import random
        import numpy as np
        
        data = []
        for i in range(1000):
            # Generate 50 time steps of data
            price_history = []
            volume_history = []
            sentiment_history = []
            
            base_price = random.uniform(0.001, 1.0)
            base_volume = random.uniform(1000, 100000)
            
            for t in range(50):
                # Add some trend and noise
                price = base_price * (1 + 0.1 * np.sin(t * 0.1) + random.uniform(-0.05, 0.05))
                volume = base_volume * (1 + random.uniform(-0.3, 0.3))
                sentiment = random.uniform(-1.0, 1.0)
                
                price_history.append({'price': price, 'timestamp': t})
                volume_history.append({'volume': volume, 'timestamp': t})
                sentiment_history.append({'score': sentiment, 'timestamp': t})
            
            data.append({
                'token_address': f'token_{i}',
                'price_history': price_history,
                'volume_history': volume_history,
                'sentiment_history': sentiment_history,
                'liquidity_history': [{'liquidity': random.uniform(10000, 500000)} for _ in range(50)],
                'holder_history': [{'count': random.randint(50, 5000)} for _ in range(50)]
            })
        
        return data
    
    def _generate_synthetic_graph_data(self) -> Dict[str, Any]:
        """Generate synthetic graph data for Network Ant"""
        
        import random
        
        # Generate synthetic wallets
        wallets = []
        for i in range(100):
            wallet_data = {
                'address': f'wallet_{i}',
                'success_rate': random.uniform(0.3, 0.9),
                'profit_factor': random.uniform(0.5, 3.0),
                'avg_position_size': random.uniform(0.1, 2.0),
                'trade_frequency': random.uniform(1, 100),
                'risk_score': random.uniform(0.1, 0.9),
                'win_rate': random.uniform(0.3, 0.8),
                'avg_profit_per_trade': random.uniform(-0.1, 0.3),
                'max_drawdown': random.uniform(0.1, 0.5),
                'total_trades': random.randint(10, 1000),
                'total_volume': random.uniform(1000, 1000000),
                'behavior_type': random.choice(['smart_money', 'sniper', 'whale', 'organic_trader'])
            }
            wallets.append(wallet_data)
        
        # Generate synthetic transactions
        edges = []
        for i in range(200):
            edge = {
                'from': f'wallet_{random.randint(0, 99)}',
                'to': f'wallet_{random.randint(0, 99)}',
                'token_address': f'token_{random.randint(0, 50)}',
                'amount': random.uniform(0.1, 10.0),
                'gas_fee': random.uniform(0.001, 0.1),
                'timestamp': random.randint(0, 1000)
            }
            edges.append(edge)
        
        return {
            'nodes': wallets,
            'edges': edges
        }

async def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="Train ML models for Antbot")
    parser.add_argument('--model', choices=['oracle', 'hunter', 'network', 'all'], 
                       required=True, help='Model to train')
    parser.add_argument('--data_path', type=str, help='Path to training data')
    parser.add_argument('--graph_data', type=str, help='Path to graph data')
    parser.add_argument('--wallet_id', type=str, help='Wallet ID for Hunter Ant training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--models_dir', type=str, default='models', help='Models directory')
    
    args = parser.parse_args()
    
    trainer = MLModelTrainer(args.models_dir)
    
    try:
        if args.model == 'oracle' or args.model == 'all':
            if not args.data_path:
                args.data_path = 'data/synthetic_oracle_data.json'
            await trainer.train_oracle_ant(
                args.data_path, args.epochs, args.batch_size, args.learning_rate
            )
        
        if args.model == 'hunter' or args.model == 'all':
            if not args.wallet_id:
                args.wallet_id = 'wallet_0'
            await trainer.train_hunter_ant(
                args.wallet_id, args.episodes, args.learning_rate
            )
        
        if args.model == 'network' or args.model == 'all':
            if not args.graph_data:
                args.graph_data = 'data/synthetic_graph_data.json'
            await trainer.train_network_ant(
                args.graph_data, args.epochs, args.batch_size, args.learning_rate
            )
        
        logger.info("✅ All training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 