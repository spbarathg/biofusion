"""
NETWORK ANT - GRAPH NEURAL NETWORK FOR ON-CHAIN INTELLIGENCE
==========================================================

Graph Neural Network (GNN) for modeling wallet and transaction networks to detect
manipulation and smart money movements before they become obvious.

Key Features:
- Graph Convolutional Network (GCN) for wallet classification
- Link prediction for transaction forecasting
- Contagion scoring for token influence detection
- Integration with BattlePatternIntelligence
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import pickle
import os
from enum import Enum

from worker_ant_v1.utils.logger import setup_logger

class WalletType(Enum):
    """Wallet classification types"""
    SNIPER = "sniper"
    WHALE = "whale"
    RUG_PULLER = "rug_puller"
    EXCHANGE = "exchange"
    ORGANIC_TRADER = "organic_trader"
    SMART_MONEY = "smart_money"
    UNKNOWN = "unknown"

@dataclass
class WalletNode:
    """Wallet node in the transaction graph"""
    address: str
    wallet_type: WalletType
    features: torch.Tensor
    success_rate: float
    profit_factor: float
    avg_position_size: float
    trade_frequency: float
    risk_score: float
    last_updated: datetime
    
    def update_features(self, new_features: torch.Tensor):
        """Update wallet features"""
        self.features = new_features
        self.last_updated = datetime.now()

@dataclass
class TransactionEdge:
    """Transaction edge in the graph"""
    from_wallet: str
    to_wallet: str
    token_address: str
    amount: float
    timestamp: datetime
    gas_fee: float
    features: torch.Tensor

@dataclass
class NetworkPrediction:
    """Network Ant prediction result"""
    token_address: str
    contagion_score: float
    smart_money_probability: float
    manipulation_risk: float
    wallet_classifications: Dict[str, WalletType]
    link_predictions: Dict[str, float]  # wallet -> probability of interaction
    influence_network: Dict[str, List[str]]  # influential wallets and their targets
    confidence: float
    timestamp: datetime

class GraphConvolutionalLayer(nn.Module):
    """Graph Convolutional Layer for GNN"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of graph convolution
        x: Node features (num_nodes, in_features)
        adj: Adjacency matrix (num_nodes, num_nodes)
        """
        # Graph convolution: H = σ(D^(-1/2) A D^(-1/2) X W)
        # Normalize adjacency matrix
        adj_normalized = self._normalize_adjacency(adj)
        
        # Apply convolution
        support = torch.mm(adj_normalized, x)
        output = self.linear(support)
        output = F.relu(output)
        output = self.dropout(output)
        
        return output
    
    def _normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        """Normalize adjacency matrix using symmetric normalization"""
        # Add self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Compute degree matrix
        degree = torch.sum(adj, dim=1)
        degree_matrix = torch.diag(torch.pow(degree, -0.5))
        
        # Normalize
        adj_normalized = torch.mm(torch.mm(degree_matrix, adj), degree_matrix)
        return adj_normalized

class AttentionLayer(nn.Module):
    """Attention mechanism for graph attention"""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        
        # Linear transformations for each head
        self.W = nn.ModuleList([
            nn.Linear(in_features, out_features // num_heads)
            for _ in range(num_heads)
        ])
        
        # Attention mechanism
        self.attention = nn.ModuleList([
            nn.Linear(2 * (out_features // num_heads), 1)
            for _ in range(num_heads)
        ])
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-head attention
        x: Node features (num_nodes, in_features)
        adj: Adjacency matrix (num_nodes, num_nodes)
        """
        batch_size = x.size(0)
        outputs = []
        
        for head in range(self.num_heads):
            # Transform features
            h = self.W[head](x)  # (num_nodes, out_features // num_heads)
            
            # Compute attention scores
            attention_scores = []
            for i in range(batch_size):
                for j in range(batch_size):
                    if adj[i, j] > 0:  # Only for connected nodes
                        concat = torch.cat([h[i], h[j]], dim=0)
                        score = self.attention[head](concat)
                        attention_scores.append(score)
                    else:
                        attention_scores.append(torch.tensor(0.0, device=x.device))
            
            attention_scores = torch.stack(attention_scores).view(batch_size, batch_size)
            attention_probs = F.softmax(attention_scores, dim=1)
            
            # Apply attention
            h_out = torch.mm(attention_probs, h)
            outputs.append(h_out)
        
        # Concatenate heads
        output = torch.cat(outputs, dim=1)
        return output

class NetworkAnt(nn.Module):
    """Network Ant - Graph Neural Network for on-chain intelligence"""
    
    def __init__(self, 
                 node_features: int = 32,
                 hidden_dim: int = 64,
                 num_classes: int = 7,  # Number of wallet types
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph convolutional layers
        self.gcn_layers = nn.ModuleList([
            GraphConvolutionalLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            AttentionLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Node classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Contagion scoring head
        self.contagion_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        x: Node features (num_nodes, node_features)
        adj: Adjacency matrix (num_nodes, num_nodes)
        """
        # Encode node features
        h = self.node_encoder(x)
        
        # Pass through GCN layers with attention
        for i in range(self.num_layers):
            # Graph convolution
            h_gcn = self.gcn_layers[i](h, adj)
            
            # Attention mechanism
            h_attn = self.attention_layers[i](h, adj)
            
            # Combine
            h = h_gcn + h_attn
            h = F.relu(h)
        
        # Node classification
        node_logits = self.classifier(h)
        
        # Contagion scoring (global pooling)
        contagion_score = self.contagion_scorer(torch.mean(h, dim=0, keepdim=True))
        
        return {
            'node_logits': node_logits,
            'node_embeddings': h,
            'contagion_score': contagion_score
        }
    
    def predict_links(self, node_embeddings: torch.Tensor, 
                     node_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """Predict link probabilities for node pairs"""
        link_features = []
        for i, j in node_pairs:
            pair_features = torch.cat([node_embeddings[i], node_embeddings[j]], dim=0)
            link_features.append(pair_features)
        
        if link_features:
            link_features = torch.stack(link_features)
            link_probs = torch.sigmoid(self.link_predictor(link_features))
            return link_probs
        else:
            return torch.tensor([])

class NetworkAntPredictor:
    """High-level interface for Network Ant predictions"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = setup_logger("NetworkAnt")
        
        # Model configuration
        self.model = NetworkAnt(
            node_features=32,
            hidden_dim=64,
            num_classes=7,
            num_layers=3,
            dropout=0.1
        )
        
        # Graph data structures
        self.wallet_nodes: Dict[str, WalletNode] = {}
        self.transaction_edges: List[TransactionEdge] = []
        self.adjacency_matrix: Optional[torch.Tensor] = None
        self.node_features: Optional[torch.Tensor] = None
        
        # Performance tracking
        self.classification_accuracy = 0.0
        self.link_prediction_accuracy = 0.0
        self.contagion_score_accuracy = 0.0
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.logger.info("✅ Network Ant initialized")
    
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.logger.info(f"✅ Loaded Network Ant model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def save_model(self, model_path: str):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'classification_accuracy': self.classification_accuracy,
                'link_prediction_accuracy': self.link_prediction_accuracy,
                'contagion_score_accuracy': self.contagion_score_accuracy,
                'timestamp': datetime.now()
            }
            torch.save(checkpoint, model_path)
            self.logger.info(f"✅ Saved Network Ant model to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    async def add_wallet(self, address: str, wallet_data: Dict[str, Any]):
        """Add or update wallet in the graph"""
        
        try:
            # Extract wallet features
            features = self._extract_wallet_features(wallet_data)
            
            # Determine wallet type
            wallet_type = self._classify_wallet_type(wallet_data)
            
            # Create or update wallet node
            if address in self.wallet_nodes:
                self.wallet_nodes[address].update_features(features)
                self.wallet_nodes[address].wallet_type = wallet_type
            else:
                self.wallet_nodes[address] = WalletNode(
                    address=address,
                    wallet_type=wallet_type,
                    features=features,
                    success_rate=wallet_data.get('success_rate', 0.0),
                    profit_factor=wallet_data.get('profit_factor', 0.0),
                    avg_position_size=wallet_data.get('avg_position_size', 0.0),
                    trade_frequency=wallet_data.get('trade_frequency', 0.0),
                    risk_score=wallet_data.get('risk_score', 0.5),
                    last_updated=datetime.now()
                )
            
            # Update graph structure
            await self._update_graph_structure()
            
        except Exception as e:
            self.logger.error(f"Failed to add wallet {address}: {e}")
    
    async def add_transaction(self, from_wallet: str, to_wallet: str, 
                            token_address: str, amount: float, gas_fee: float):
        """Add transaction to the graph"""
        
        try:
            # Create transaction edge
            edge = TransactionEdge(
                from_wallet=from_wallet,
                to_wallet=to_wallet,
                token_address=token_address,
                amount=amount,
                timestamp=datetime.now(),
                gas_fee=gas_fee,
                features=self._extract_transaction_features(amount, gas_fee)
            )
            
            self.transaction_edges.append(edge)
            
            # Update graph structure
            await self._update_graph_structure()
            
        except Exception as e:
            self.logger.error(f"Failed to add transaction: {e}")
    
    async def predict_network_intelligence(self, token_address: str) -> NetworkPrediction:
        """Generate network intelligence prediction for a token"""
        
        try:
            # Get relevant wallets and transactions
            relevant_wallets = self._get_relevant_wallets(token_address)
            relevant_edges = self._get_relevant_edges(token_address)
            
            if len(relevant_wallets) < 2:
                return self._fallback_prediction(token_address)
            
            # Build subgraph
            node_features, adjacency_matrix = self._build_subgraph(relevant_wallets, relevant_edges)
            
            # Generate predictions
            with torch.no_grad():
                outputs = self.model(node_features, adjacency_matrix)
            
            # Extract predictions
            node_logits = outputs['node_logits']
            node_embeddings = outputs['node_embeddings']
            contagion_score = outputs['contagion_score']
            
            # Classify wallets
            wallet_classifications = {}
            for i, wallet_address in enumerate(relevant_wallets):
                if i < len(node_logits):
                    wallet_type_idx = torch.argmax(node_logits[i]).item()
                    wallet_type = list(WalletType)[wallet_type_idx]
                    wallet_classifications[wallet_address] = wallet_type
            
            # Predict links
            link_predictions = {}
            for wallet_address in relevant_wallets:
                if wallet_address in self.wallet_nodes:
                    # Predict probability of interaction with new wallets
                    link_prob = self._predict_wallet_interaction(wallet_address, node_embeddings)
                    link_predictions[wallet_address] = link_prob
            
            # Calculate influence network
            influence_network = self._calculate_influence_network(
                relevant_wallets, wallet_classifications, node_embeddings
            )
            
            # Calculate risk metrics
            manipulation_risk = self._calculate_manipulation_risk(
                wallet_classifications, relevant_edges
            )
            
            smart_money_probability = self._calculate_smart_money_probability(
                wallet_classifications, relevant_wallets
            )
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(
                len(relevant_wallets), len(relevant_edges), node_logits
            )
            
            return NetworkPrediction(
                token_address=token_address,
                contagion_score=float(contagion_score),
                smart_money_probability=smart_money_probability,
                manipulation_risk=manipulation_risk,
                wallet_classifications=wallet_classifications,
                link_predictions=link_predictions,
                influence_network=influence_network,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Network intelligence prediction failed for {token_address}: {e}")
            return self._fallback_prediction(token_address)
    
    def _extract_wallet_features(self, wallet_data: Dict[str, Any]) -> torch.Tensor:
        """Extract features for wallet node"""
        
        features = [
            wallet_data.get('success_rate', 0.0),
            wallet_data.get('profit_factor', 0.0),
            wallet_data.get('avg_position_size', 0.0),
            wallet_data.get('trade_frequency', 0.0),
            wallet_data.get('risk_score', 0.5),
            wallet_data.get('win_rate', 0.0),
            wallet_data.get('avg_profit_per_trade', 0.0),
            wallet_data.get('max_drawdown', 0.0),
            wallet_data.get('reaction_speed_ms', 0.0),
            wallet_data.get('entry_timing_percentile', 0.0),
            wallet_data.get('rug_participation_count', 0),
            wallet_data.get('suspicious_patterns_detected', 0),
            wallet_data.get('total_trades', 0),
            wallet_data.get('total_volume', 0.0),
            wallet_data.get('avg_gas_fee', 0.0),
            wallet_data.get('preferred_tokens', 0),
            wallet_data.get('time_active_days', 0),
            wallet_data.get('balance', 0.0),
            wallet_data.get('token_diversity', 0.0),
            wallet_data.get('trade_size_volatility', 0.0),
            wallet_data.get('holding_time_avg', 0.0),
            wallet_data.get('peak_profit', 0.0),
            wallet_data.get('recovery_rate', 0.0),
            wallet_data.get('market_timing_score', 0.0),
            wallet_data.get('liquidity_preference', 0.0),
            wallet_data.get('slippage_tolerance', 0.0),
            wallet_data.get('bot_detection_score', 0.0),
            wallet_data.get('whale_interaction_rate', 0.0),
            wallet_data.get('rug_detection_accuracy', 0.0),
            wallet_data.get('sentiment_correlation', 0.0),
            wallet_data.get('volume_correlation', 0.0),
            wallet_data.get('price_correlation', 0.0)
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_transaction_features(self, amount: float, gas_fee: float) -> torch.Tensor:
        """Extract features for transaction edge"""
        
        features = [
            amount,
            gas_fee,
            amount / gas_fee if gas_fee > 0 else 0,
            np.log(amount + 1),
            np.log(gas_fee + 1)
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _classify_wallet_type(self, wallet_data: Dict[str, Any]) -> WalletType:
        """Classify wallet type based on behavior patterns"""
        
        success_rate = wallet_data.get('success_rate', 0.0)
        profit_factor = wallet_data.get('profit_factor', 0.0)
        risk_score = wallet_data.get('risk_score', 0.5)
        rug_participation = wallet_data.get('rug_participation_count', 0)
        
        if rug_participation > 5:
            return WalletType.RUG_PULLER
        elif success_rate > 0.8 and profit_factor > 2.0:
            return WalletType.SMART_MONEY
        elif success_rate > 0.7 and profit_factor > 1.5:
            return WalletType.SNIPER
        elif wallet_data.get('total_volume', 0) > 1000000:  # 1M SOL
            return WalletType.WHALE
        elif wallet_data.get('trade_frequency', 0) > 100:  # High frequency
            return WalletType.EXCHANGE
        else:
            return WalletType.ORGANIC_TRADER
    
    async def _update_graph_structure(self):
        """Update graph adjacency matrix and node features"""
        
        try:
            if not self.wallet_nodes:
                return
            
            # Create node feature matrix
            wallet_addresses = list(self.wallet_nodes.keys())
            node_features = []
            
            for address in wallet_addresses:
                node = self.wallet_nodes[address]
                node_features.append(node.features)
            
            self.node_features = torch.stack(node_features)
            
            # Create adjacency matrix
            num_nodes = len(wallet_addresses)
            adjacency_matrix = torch.zeros(num_nodes, num_nodes)
            
            # Add edges from transactions
            for edge in self.transaction_edges:
                if edge.from_wallet in wallet_addresses and edge.to_wallet in wallet_addresses:
                    i = wallet_addresses.index(edge.from_wallet)
                    j = wallet_addresses.index(edge.to_wallet)
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1  # Undirected graph
            
            self.adjacency_matrix = adjacency_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to update graph structure: {e}")
    
    def _get_relevant_wallets(self, token_address: str) -> List[str]:
        """Get wallets relevant to a specific token"""
        
        relevant_wallets = set()
        
        # Find wallets that have traded this token
        for edge in self.transaction_edges:
            if edge.token_address == token_address:
                relevant_wallets.add(edge.from_wallet)
                relevant_wallets.add(edge.to_wallet)
        
        # Add wallets with similar behavior patterns
        for wallet_address, node in self.wallet_nodes.items():
            if node.wallet_type in [WalletType.SMART_MONEY, WalletType.SNIPER, WalletType.WHALE]:
                relevant_wallets.add(wallet_address)
        
        return list(relevant_wallets)
    
    def _get_relevant_edges(self, token_address: str) -> List[TransactionEdge]:
        """Get transactions relevant to a specific token"""
        
        relevant_edges = []
        
        for edge in self.transaction_edges:
            if edge.token_address == token_address:
                relevant_edges.append(edge)
        
        return relevant_edges
    
    def _build_subgraph(self, wallet_addresses: List[str], 
                       edges: List[TransactionEdge]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build subgraph for specific wallets and edges"""
        
        # Create node feature matrix
        node_features = []
        for address in wallet_addresses:
            if address in self.wallet_nodes:
                node_features.append(self.wallet_nodes[address].features)
            else:
                # Create default features for unknown wallets
                default_features = torch.zeros(32)
                node_features.append(default_features)
        
        node_features = torch.stack(node_features)
        
        # Create adjacency matrix
        num_nodes = len(wallet_addresses)
        adjacency_matrix = torch.zeros(num_nodes, num_nodes)
        
        # Add edges
        for edge in edges:
            if edge.from_wallet in wallet_addresses and edge.to_wallet in wallet_addresses:
                i = wallet_addresses.index(edge.from_wallet)
                j = wallet_addresses.index(edge.to_wallet)
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
        
        return node_features, adjacency_matrix
    
    def _predict_wallet_interaction(self, wallet_address: str, 
                                  node_embeddings: torch.Tensor) -> float:
        """Predict probability of wallet interaction with new tokens"""
        
        try:
            if wallet_address not in self.wallet_nodes:
                return 0.0
            
            # Get wallet embedding
            wallet_idx = list(self.wallet_nodes.keys()).index(wallet_address)
            if wallet_idx < len(node_embeddings):
                wallet_embedding = node_embeddings[wallet_idx]
                
                # Calculate interaction probability based on wallet type and features
                wallet_type = self.wallet_nodes[wallet_address].wallet_type
                
                if wallet_type == WalletType.SMART_MONEY:
                    base_prob = 0.8
                elif wallet_type == WalletType.SNIPER:
                    base_prob = 0.6
                elif wallet_type == WalletType.WHALE:
                    base_prob = 0.4
                else:
                    base_prob = 0.2
                
                # Adjust based on wallet features
                success_rate = self.wallet_nodes[wallet_address].success_rate
                profit_factor = self.wallet_nodes[wallet_address].profit_factor
                
                adjusted_prob = base_prob * (1 + success_rate * 0.5 + profit_factor * 0.3)
                return min(1.0, adjusted_prob)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to predict wallet interaction: {e}")
            return 0.0
    
    def _calculate_influence_network(self, wallet_addresses: List[str],
                                   classifications: Dict[str, WalletType],
                                   node_embeddings: torch.Tensor) -> Dict[str, List[str]]:
        """Calculate influence network of wallets"""
        
        influence_network = {}
        
        for wallet_address in wallet_addresses:
            if wallet_address in classifications:
                wallet_type = classifications[wallet_address]
                
                if wallet_type in [WalletType.SMART_MONEY, WalletType.WHALE]:
                    # Find wallets that might be influenced by this wallet
                    influenced_wallets = []
                    
                    for other_address in wallet_addresses:
                        if other_address != wallet_address:
                            other_type = classifications.get(other_address, WalletType.UNKNOWN)
                            
                            # Organic traders and snipers are more likely to be influenced
                            if other_type in [WalletType.ORGANIC_TRADER, WalletType.SNIPER]:
                                influenced_wallets.append(other_address)
                    
                    influence_network[wallet_address] = influenced_wallets
        
        return influence_network
    
    def _calculate_manipulation_risk(self, classifications: Dict[str, WalletType],
                                   edges: List[TransactionEdge]) -> float:
        """Calculate manipulation risk based on wallet types and transaction patterns"""
        
        risk_score = 0.0
        
        # Count suspicious wallet types
        rug_pullers = sum(1 for wt in classifications.values() if wt == WalletType.RUG_PULLER)
        smart_money = sum(1 for wt in classifications.values() if wt == WalletType.SMART_MONEY)
        
        # Calculate risk based on proportions
        total_wallets = len(classifications)
        if total_wallets > 0:
            rug_puller_ratio = rug_pullers / total_wallets
            smart_money_ratio = smart_money / total_wallets
            
            # High rug puller ratio increases risk
            risk_score += rug_puller_ratio * 0.8
            
            # High smart money ratio can indicate manipulation
            if smart_money_ratio > 0.3:
                risk_score += smart_money_ratio * 0.4
        
        # Analyze transaction patterns
        if edges:
            # Check for coordinated transactions
            transaction_times = [edge.timestamp for edge in edges]
            time_diffs = []
            for i in range(1, len(transaction_times)):
                diff = (transaction_times[i] - transaction_times[i-1]).total_seconds()
                time_diffs.append(diff)
            
            # If many transactions happen in quick succession, increase risk
            if time_diffs:
                avg_time_diff = np.mean(time_diffs)
                if avg_time_diff < 60:  # Less than 1 minute between transactions
                    risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def _calculate_smart_money_probability(self, classifications: Dict[str, WalletType],
                                         wallet_addresses: List[str]) -> float:
        """Calculate probability of smart money involvement"""
        
        smart_money_count = sum(1 for wt in classifications.values() 
                               if wt == WalletType.SMART_MONEY)
        
        total_wallets = len(classifications)
        if total_wallets > 0:
            return smart_money_count / total_wallets
        
        return 0.0
    
    def _calculate_prediction_confidence(self, num_wallets: int, num_edges: int,
                                       node_logits: torch.Tensor) -> float:
        """Calculate confidence in prediction based on data quality"""
        
        # Base confidence on amount of data
        wallet_confidence = min(1.0, num_wallets / 10.0)  # More wallets = higher confidence
        edge_confidence = min(1.0, num_edges / 20.0)      # More edges = higher confidence
        
        # Check prediction certainty
        if len(node_logits) > 0:
            max_probs = torch.max(F.softmax(node_logits, dim=1), dim=1)[0]
            avg_certainty = torch.mean(max_probs).item()
        else:
            avg_certainty = 0.5
        
        # Combine factors
        confidence = (wallet_confidence * 0.4 + edge_confidence * 0.3 + avg_certainty * 0.3)
        return min(1.0, confidence)
    
    def _fallback_prediction(self, token_address: str) -> NetworkPrediction:
        """Generate fallback prediction when model fails"""
        
        return NetworkPrediction(
            token_address=token_address,
            contagion_score=0.0,
            smart_money_probability=0.0,
            manipulation_risk=0.5,
            wallet_classifications={},
            link_predictions={},
            influence_network={},
            confidence=0.1,
            timestamp=datetime.now()
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'total_wallets': len(self.wallet_nodes),
            'total_transactions': len(self.transaction_edges),
            'classification_accuracy': self.classification_accuracy,
            'link_prediction_accuracy': self.link_prediction_accuracy,
            'contagion_score_accuracy': self.contagion_score_accuracy,
            'model_status': 'active'
        } 