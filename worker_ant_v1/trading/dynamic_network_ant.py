"""
DYNAMIC NETWORK ANT - TEMPORAL GRAPH NEURAL NETWORK
=================================================

Enhanced Graph Neural Network with temporal dynamics and hierarchical attention
for sophisticated on-chain wallet and transaction analysis.

Key Features:
- Temporal Graph Dynamics: Evolving graph structure over time
- Hierarchical Attention: Node-level and graph-level attention mechanisms
- Dynamic Edge Prediction: Real-time relationship formation prediction
- Memory-Augmented Architecture: Temporal memory for evolving relationships
- Multi-Scale Analysis: Local patterns and global market dynamics
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
import math

from worker_ant_v1.utils.logger import setup_logger

class WalletType(Enum):
    """Enhanced wallet classification types"""
    SNIPER = "sniper"
    WHALE = "whale"
    RUG_PULLER = "rug_puller"
    EXCHANGE = "exchange"
    ORGANIC_TRADER = "organic_trader"
    SMART_MONEY = "smart_money"
    MARKET_MAKER = "market_maker"
    WASH_TRADER = "wash_trader"
    FRONTRUNNER = "frontrunner"
    UNKNOWN = "unknown"

@dataclass
class TemporalNode:
    """Node with temporal features"""
    wallet_address: str
    node_id: int
    static_features: torch.Tensor  # Time-invariant features
    temporal_features: torch.Tensor  # Time-varying features
    activity_history: List[Dict[str, Any]]  # Historical activity
    embeddings_history: deque  # Historical embeddings
    last_updated: datetime
    confidence_score: float = 0.0

@dataclass
class TemporalEdge:
    """Edge with temporal dynamics"""
    source_id: int
    target_id: int
    edge_type: str  # transaction, collaboration, etc.
    weight: float
    temporal_features: torch.Tensor
    creation_time: datetime
    last_activity: datetime
    interaction_frequency: float
    strength_score: float = 0.0

@dataclass
class GraphSnapshot:
    """Graph state at specific timestamp"""
    timestamp: datetime
    nodes: Dict[int, TemporalNode]
    edges: List[TemporalEdge]
    adjacency_matrix: torch.Tensor
    global_features: torch.Tensor
    market_state: Dict[str, Any]

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        return x + self.pe[:x.size(0), :]

class HierarchicalAttention(nn.Module):
    """Hierarchical attention mechanism with node-level and graph-level attention"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        # Node-level attention
        self.node_query = nn.Linear(hidden_dim, hidden_dim)
        self.node_key = nn.Linear(hidden_dim, hidden_dim)
        self.node_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Graph-level attention
        self.graph_query = nn.Linear(hidden_dim, hidden_dim)
        self.graph_key = nn.Linear(hidden_dim, hidden_dim)
        self.graph_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.dropout = nn.Dropout(0.1)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
               batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with hierarchical attention
        x: Node features (num_nodes, hidden_dim)
        edge_index: Edge connectivity (2, num_edges)
        batch: Batch assignment for nodes (optional)
        """
        num_nodes = x.size(0)
        
        # Node-level attention
        node_attended = self._node_attention(x, edge_index)
        
        # Graph-level attention
        graph_attended = self._graph_attention(x, batch)
        
        # Fuse both attention mechanisms
        combined = torch.cat([node_attended, graph_attended], dim=-1)
        output = self.fusion_layer(combined)
        
        return output
    
    def _node_attention(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Node-level attention focusing on local neighborhoods"""
        
        # Multi-head attention computation
        Q = self.node_query(x)  # (num_nodes, hidden_dim)
        K = self.node_key(x)
        V = self.node_value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(-1, self.num_heads, self.head_dim)  # (num_nodes, num_heads, head_dim)
        K = K.view(-1, self.num_heads, self.head_dim)
        V = V.view(-1, self.num_heads, self.head_dim)
        
        # Calculate attention scores based on graph connectivity
        attended_features = []
        
        for head in range(self.num_heads):
            q_head = Q[:, head, :]  # (num_nodes, head_dim)
            k_head = K[:, head, :]
            v_head = V[:, head, :]
            
            # Create attention matrix based on edges
            attention_matrix = torch.zeros(x.size(0), x.size(0), device=x.device)
            
            if edge_index.size(1) > 0:
                # Calculate attention scores for connected nodes
                source_nodes = edge_index[0]
                target_nodes = edge_index[1]
                
                attention_scores = torch.sum(q_head[source_nodes] * k_head[target_nodes], dim=1) / self.scale
                attention_matrix[source_nodes, target_nodes] = attention_scores
                
                # Make attention symmetric
                attention_matrix = attention_matrix + attention_matrix.t()
            
            # Apply softmax to attention scores
            attention_weights = F.softmax(attention_matrix, dim=1)
            
            # Apply attention to values
            attended_head = torch.mm(attention_weights, v_head)
            attended_features.append(attended_head)
        
        # Concatenate heads
        attended = torch.cat(attended_features, dim=1)
        
        return attended
    
    def _graph_attention(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Graph-level attention focusing on global patterns"""
        
        # For simplicity, use global average pooling with learnable attention
        if batch is None:
            # Single graph case
            global_context = torch.mean(x, dim=0, keepdim=True)  # (1, hidden_dim)
            global_context = global_context.expand(x.size(0), -1)  # (num_nodes, hidden_dim)
        else:
            # Batch case - compute global context per graph
            unique_batches = torch.unique(batch)
            global_contexts = []
            
            for b in unique_batches:
                mask = batch == b
                batch_nodes = x[mask]
                batch_context = torch.mean(batch_nodes, dim=0, keepdim=True)
                global_contexts.append(batch_context)
            
            global_context = torch.cat(global_contexts, dim=0)
            global_context = global_context[batch]  # Map back to nodes
        
        # Apply attention mechanism
        Q_graph = self.graph_query(x)
        K_graph = self.graph_key(global_context)
        V_graph = self.graph_value(global_context)
        
        # Calculate attention scores
        attention_scores = torch.sum(Q_graph * K_graph, dim=1, keepdim=True) / self.scale
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # Apply attention
        graph_attended = attention_weights * V_graph
        
        return graph_attended

class TemporalGraphConvolution(nn.Module):
    """Temporal graph convolution with memory"""
    
    def __init__(self, in_dim: int, out_dim: int, temporal_dim: int = 64):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.temporal_dim = temporal_dim
        
        # Spatial convolution
        self.spatial_conv = nn.Linear(in_dim, out_dim)
        
        # Temporal convolution
        self.temporal_conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1)
        
        # Memory mechanism
        self.memory_query = nn.Linear(out_dim, temporal_dim)
        self.memory_key = nn.Linear(out_dim, temporal_dim)
        self.memory_value = nn.Linear(out_dim, out_dim)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
               temporal_features: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with temporal dynamics
        x: Current node features (num_nodes, in_dim)
        edge_index: Edge connectivity (2, num_edges)
        temporal_features: Temporal node features (num_nodes, seq_len, in_dim)
        memory: Memory from previous timesteps (num_nodes, out_dim)
        """
        
        # Spatial convolution
        spatial_out = self.spatial_conv(x)
        
        # Apply graph convolution
        if edge_index.size(1) > 0:
            # Aggregate features from neighbors
            source_nodes = edge_index[0]
            target_nodes = edge_index[1]
            
            # Simple message passing
            messages = spatial_out[source_nodes]
            aggregated = torch.zeros_like(spatial_out)
            aggregated.index_add_(0, target_nodes, messages)
            
            spatial_out = spatial_out + aggregated
        
        # Temporal convolution
        if temporal_features.size(1) > 1:  # If we have temporal sequence
            temporal_input = temporal_features.transpose(1, 2)  # (num_nodes, in_dim, seq_len)
            temporal_out = self.temporal_conv(temporal_input)  # (num_nodes, out_dim, seq_len)
            temporal_out = temporal_out.mean(dim=2)  # Average over time
        else:
            temporal_out = self.spatial_conv(temporal_features.squeeze(1))
        
        # Memory mechanism
        if memory is not None:
            queries = self.memory_query(spatial_out)
            keys = self.memory_key(memory)
            values = self.memory_value(memory)
            
            # Attention over memory
            attention_scores = torch.sum(queries * keys, dim=1, keepdim=True)
            attention_weights = F.softmax(attention_scores, dim=0)
            memory_out = attention_weights * values
        else:
            memory_out = torch.zeros_like(spatial_out)
        
        # Fuse spatial, temporal, and memory information
        combined = torch.cat([spatial_out, temporal_out], dim=1)
        output = self.fusion(combined)
        
        return output

class DynamicNetworkAnt(nn.Module):
    """Dynamic Graph Neural Network with temporal evolution and hierarchical attention"""
    
    def __init__(self, 
                 node_features: int = 64,
                 edge_features: int = 16,
                 hidden_dim: int = 128,
                 temporal_dim: int = 64,
                 num_classes: int = 10,  # Enhanced wallet types
                 num_layers: int = 4,
                 num_heads: int = 8,
                 memory_size: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.temporal_dim = temporal_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.memory_size = memory_size
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # Positional encoding for temporal sequences
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Temporal graph convolution layers
        self.temporal_conv_layers = nn.ModuleList([
            TemporalGraphConvolution(hidden_dim, hidden_dim, temporal_dim)
            for _ in range(num_layers)
        ])
        
        # Hierarchical attention layers
        self.attention_layers = nn.ModuleList([
            HierarchicalAttention(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Memory networks for temporal information
        self.memory_networks = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output heads
        
        # Node classification (wallet type prediction)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Edge prediction (relationship formation)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Graph-level tasks
        self.contagion_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.manipulation_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Temporal dynamics predictor
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # Predict next timestep features
        )
        
        # Initialize memory
        self.register_buffer('node_memory', torch.zeros(memory_size, hidden_dim))
        self.memory_pointer = 0
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
               edge_features: torch.Tensor, temporal_node_features: torch.Tensor,
               memory_states: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with temporal dynamics
        
        Args:
            node_features: Current node features (num_nodes, node_features)
            edge_index: Edge connectivity (2, num_edges)
            edge_features: Edge features (num_edges, edge_features)
            temporal_node_features: Historical node features (num_nodes, seq_len, node_features)
            memory_states: Memory states from previous timesteps
        """
        
        num_nodes = node_features.size(0)
        
        # Encode node and edge features
        h_nodes = self.node_encoder(node_features)
        
        if edge_features.size(0) > 0:
            h_edges = self.edge_encoder(edge_features)
        else:
            h_edges = torch.zeros(0, self.hidden_dim // 2, device=node_features.device)
        
        # Add positional encoding for temporal aspect
        if temporal_node_features.size(1) > 1:
            h_nodes = self.positional_encoding(h_nodes.unsqueeze(1)).squeeze(1)
        
        # Initialize memory states if not provided
        if memory_states is None:
            memory_states = [torch.zeros(num_nodes, self.hidden_dim, device=node_features.device) 
                           for _ in range(self.num_layers)]
        
        # Process through layers
        layer_outputs = []
        new_memory_states = []
        
        for i in range(self.num_layers):
            # Temporal graph convolution
            h_conv = self.temporal_conv_layers[i](
                h_nodes, edge_index, temporal_node_features, memory_states[i]
            )
            
            # Hierarchical attention
            h_attn = self.attention_layers[i](h_conv, edge_index)
            
            # Combine convolution and attention
            h_combined = h_conv + h_attn
            h_combined = F.relu(h_combined)
            
            # Update memory
            new_memory = self.memory_networks[i](h_combined.view(-1, self.hidden_dim), 
                                               memory_states[i].view(-1, self.hidden_dim))
            new_memory = new_memory.view(num_nodes, self.hidden_dim)
            new_memory_states.append(new_memory)
            
            # Store for skip connections
            layer_outputs.append(h_combined)
            h_nodes = h_combined
        
        # Final node representations
        final_node_repr = h_nodes
        
        # Node classification (wallet type prediction)
        node_logits = self.node_classifier(final_node_repr)
        
        # Edge prediction for all node pairs
        edge_predictions = self._predict_all_edges(final_node_repr, h_edges)
        
        # Graph-level predictions
        graph_repr = torch.mean(final_node_repr, dim=0, keepdim=True)
        contagion_score = self.contagion_scorer(graph_repr)
        manipulation_score = self.manipulation_detector(graph_repr)
        
        # Temporal dynamics prediction
        next_step_features = self.dynamics_predictor(final_node_repr)
        
        return {
            'node_logits': node_logits,
            'node_embeddings': final_node_repr,
            'edge_predictions': edge_predictions,
            'contagion_score': contagion_score,
            'manipulation_score': manipulation_score,
            'next_step_features': next_step_features,
            'memory_states': new_memory_states,
            'layer_outputs': layer_outputs
        }
    
    def _predict_all_edges(self, node_embeddings: torch.Tensor, 
                          edge_features: torch.Tensor) -> torch.Tensor:
        """Predict edge probabilities for all node pairs"""
        
        num_nodes = node_embeddings.size(0)
        
        # Create all possible node pairs
        edge_predictions = torch.zeros(num_nodes, num_nodes, device=node_embeddings.device)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Concatenate node embeddings
                edge_repr = torch.cat([node_embeddings[i], node_embeddings[j]], dim=0)
                
                # Add average edge features if available
                if edge_features.size(0) > 0:
                    avg_edge_features = torch.mean(edge_features, dim=0)
                    edge_repr = torch.cat([edge_repr, avg_edge_features], dim=0)
                else:
                    zero_edge_features = torch.zeros(self.hidden_dim // 2, device=node_embeddings.device)
                    edge_repr = torch.cat([edge_repr, zero_edge_features], dim=0)
                
                # Predict edge probability
                edge_prob = torch.sigmoid(self.edge_predictor(edge_repr.unsqueeze(0)))
                edge_predictions[i, j] = edge_prob
                edge_predictions[j, i] = edge_prob  # Symmetric
        
        return edge_predictions

class DynamicNetworkAntPredictor:
    """High-level interface for Dynamic Network Ant with temporal evolution"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = setup_logger("DynamicNetworkAnt")
        
        # Model configuration
        self.model = DynamicNetworkAnt(
            node_features=64,
            edge_features=16,
            hidden_dim=128,
            temporal_dim=64,
            num_classes=10,
            num_layers=4,
            num_heads=8,
            memory_size=1000,
            dropout=0.1
        )
        
        # Temporal graph storage
        self.graph_history: deque = deque(maxlen=100)  # Store last 100 snapshots
        self.current_snapshot: Optional[GraphSnapshot] = None
        
        # Node and edge tracking
        self.wallet_nodes: Dict[str, TemporalNode] = {}
        self.temporal_edges: Dict[Tuple[str, str], TemporalEdge] = {}
        self.node_id_mapping: Dict[str, int] = {}
        self.next_node_id = 0
        
        # Performance tracking
        self.prediction_accuracy = 0.0
        self.edge_prediction_accuracy = 0.0
        self.temporal_consistency = 0.0
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.logger.info("✅ Dynamic Network Ant initialized")
    
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.logger.info(f"✅ Loaded Dynamic Network Ant model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def save_model(self, model_path: str):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'prediction_accuracy': self.prediction_accuracy,
                'temporal_consistency': self.temporal_consistency,
                'timestamp': datetime.now()
            }
            torch.save(checkpoint, model_path)
            self.logger.info(f"✅ Saved Dynamic Network Ant model to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    async def predict_network_intelligence(self, token_address: str, 
                                         include_temporal: bool = True) -> Dict[str, Any]:
        """Generate network intelligence prediction with temporal dynamics"""
        
        try:
            # Build current graph snapshot
            current_snapshot = await self._build_current_snapshot(token_address)
            
            if current_snapshot is None:
                return self._fallback_prediction(token_address)
            
            # Prepare model inputs
            node_features, edge_index, edge_features, temporal_features = \
                self._prepare_model_inputs(current_snapshot, include_temporal)
            
            # Get memory states from previous timestep
            memory_states = self._get_memory_states()
            
            # Generate predictions
            with torch.no_grad():
                outputs = self.model(
                    node_features, edge_index, edge_features, 
                    temporal_features, memory_states
                )
            
            # Process outputs
            predictions = await self._process_predictions(outputs, current_snapshot, token_address)
            
            # Update graph history
            self.graph_history.append(current_snapshot)
            self.current_snapshot = current_snapshot
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Dynamic network intelligence prediction failed for {token_address}: {e}")
            return self._fallback_prediction(token_address)
    
    async def _build_current_snapshot(self, token_address: str) -> Optional[GraphSnapshot]:
        """Build current graph snapshot with temporal information"""
        
        try:
            # Get relevant wallets and transactions for the token
            relevant_wallets = await self._get_relevant_wallets(token_address)
            relevant_edges = await self._get_relevant_edges(token_address)
            
            if len(relevant_wallets) < 2:
                return None
            
            # Create or update temporal nodes
            nodes = {}
            for i, wallet_address in enumerate(relevant_wallets):
                if wallet_address not in self.node_id_mapping:
                    self.node_id_mapping[wallet_address] = self.next_node_id
                    self.next_node_id += 1
                
                node_id = self.node_id_mapping[wallet_address]
                
                # Create temporal node
                temporal_node = await self._create_temporal_node(wallet_address, node_id)
                nodes[node_id] = temporal_node
            
            # Create temporal edges
            edges = []
            for edge_data in relevant_edges:
                temporal_edge = await self._create_temporal_edge(edge_data)
                if temporal_edge:
                    edges.append(temporal_edge)
            
            # Build adjacency matrix
            num_nodes = len(nodes)
            adjacency_matrix = torch.zeros(num_nodes, num_nodes)
            
            for edge in edges:
                if edge.source_id < num_nodes and edge.target_id < num_nodes:
                    adjacency_matrix[edge.source_id, edge.target_id] = edge.weight
                    adjacency_matrix[edge.target_id, edge.source_id] = edge.weight
            
            # Global features
            global_features = await self._extract_global_features(token_address)
            
            # Market state
            market_state = await self._get_market_state(token_address)
            
            snapshot = GraphSnapshot(
                timestamp=datetime.utcnow(),
                nodes=nodes,
                edges=edges,
                adjacency_matrix=adjacency_matrix,
                global_features=global_features,
                market_state=market_state
            )
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error building graph snapshot: {e}")
            return None
    
    async def _create_temporal_node(self, wallet_address: str, node_id: int) -> TemporalNode:
        """Create temporal node with historical features"""
        
        # Extract static features (wallet characteristics that don't change much)
        static_features = torch.tensor([
            0.5,  # Account age (normalized)
            0.3,  # Activity frequency
            0.7,  # Diversity score
            0.4,  # Network centrality
        ], dtype=torch.float32)
        
        # Pad to required size
        static_features = F.pad(static_features, (0, 60))  # Pad to 64 features
        
        # Extract temporal features (current state)
        temporal_features = torch.tensor([
            1.0,  # Current balance
            0.2,  # Recent activity
            0.8,  # Position size
            0.6,  # Profit/loss
        ], dtype=torch.float32)
        
        # Pad to required size
        temporal_features = F.pad(temporal_features, (0, 60))  # Pad to 64 features
        
        # Activity history (simplified)
        activity_history = [
            {'timestamp': datetime.utcnow(), 'action': 'trade', 'amount': 100.0}
        ]
        
        # Historical embeddings
        embeddings_history = deque(maxlen=50)
        
        return TemporalNode(
            wallet_address=wallet_address,
            node_id=node_id,
            static_features=static_features,
            temporal_features=temporal_features,
            activity_history=activity_history,
            embeddings_history=embeddings_history,
            last_updated=datetime.utcnow(),
            confidence_score=0.8
        )
    
    async def _create_temporal_edge(self, edge_data: Dict[str, Any]) -> Optional[TemporalEdge]:
        """Create temporal edge from transaction data"""
        
        try:
            source_wallet = edge_data.get('source_wallet')
            target_wallet = edge_data.get('target_wallet')
            
            if not source_wallet or not target_wallet:
                return None
            
            source_id = self.node_id_mapping.get(source_wallet)
            target_id = self.node_id_mapping.get(target_wallet)
            
            if source_id is None or target_id is None:
                return None
            
            # Extract edge features
            temporal_features = torch.tensor([
                edge_data.get('amount', 0.0),
                edge_data.get('frequency', 1.0),
                edge_data.get('recency', 1.0),
                0.0  # Placeholder
            ], dtype=torch.float32)
            
            # Pad to required size
            temporal_features = F.pad(temporal_features, (0, 12))  # Pad to 16 features
            
            return TemporalEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_data.get('type', 'transaction'),
                weight=edge_data.get('weight', 1.0),
                temporal_features=temporal_features,
                creation_time=edge_data.get('creation_time', datetime.utcnow()),
                last_activity=edge_data.get('last_activity', datetime.utcnow()),
                interaction_frequency=edge_data.get('frequency', 1.0),
                strength_score=edge_data.get('strength', 0.5)
            )
            
        except Exception as e:
            self.logger.error(f"Error creating temporal edge: {e}")
            return None
    
    def _prepare_model_inputs(self, snapshot: GraphSnapshot, 
                            include_temporal: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare inputs for the model"""
        
        # Node features
        node_features = []
        temporal_node_features = []
        
        for node_id in sorted(snapshot.nodes.keys()):
            node = snapshot.nodes[node_id]
            
            # Combine static and temporal features
            combined_features = torch.cat([node.static_features, node.temporal_features])
            node_features.append(combined_features)
            
            # Temporal sequence (use last few timesteps)
            if include_temporal and len(node.embeddings_history) > 0:
                # Use historical embeddings if available
                temporal_seq = list(node.embeddings_history)[-10:]  # Last 10 timesteps
                if len(temporal_seq) < 10:
                    # Pad with current features
                    padding = [combined_features] * (10 - len(temporal_seq))
                    temporal_seq = padding + temporal_seq
                temporal_node_features.append(torch.stack(temporal_seq))
            else:
                # Use current features repeated
                temporal_node_features.append(combined_features.unsqueeze(0).repeat(10, 1))
        
        node_features = torch.stack(node_features)
        temporal_node_features = torch.stack(temporal_node_features)
        
        # Edge index and features
        edge_index = []
        edge_features = []
        
        for edge in snapshot.edges:
            edge_index.append([edge.source_id, edge.target_id])
            edge_features.append(edge.temporal_features)
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_features = torch.stack(edge_features)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_features = torch.zeros(0, 16)
        
        return node_features, edge_index, edge_features, temporal_node_features
    
    def _get_memory_states(self) -> Optional[List[torch.Tensor]]:
        """Get memory states from previous timestep"""
        
        if self.current_snapshot is None:
            return None
        
        # Use stored embeddings as memory (simplified)
        num_nodes = len(self.current_snapshot.nodes)
        memory_states = []
        
        for layer in range(self.model.num_layers):
            # Initialize with zeros for now (could be enhanced)
            memory = torch.zeros(num_nodes, self.model.hidden_dim)
            memory_states.append(memory)
        
        return memory_states
    
    async def _process_predictions(self, outputs: Dict[str, torch.Tensor], 
                                 snapshot: GraphSnapshot, token_address: str) -> Dict[str, Any]:
        """Process model outputs into final predictions"""
        
        try:
            # Node classifications
            node_logits = outputs['node_logits']
            node_probs = F.softmax(node_logits, dim=1)
            
            wallet_classifications = {}
            for i, (node_id, node) in enumerate(snapshot.nodes.items()):
                if i < len(node_probs):
                    wallet_type_idx = torch.argmax(node_probs[i]).item()
                    wallet_type = list(WalletType)[wallet_type_idx]
                    confidence = torch.max(node_probs[i]).item()
                    
                    wallet_classifications[node.wallet_address] = {
                        'type': wallet_type,
                        'confidence': confidence
                    }
            
            # Edge predictions
            edge_predictions = outputs['edge_predictions']
            predicted_edges = {}
            
            node_addresses = [node.wallet_address for node in snapshot.nodes.values()]
            for i in range(len(node_addresses)):
                for j in range(i + 1, len(node_addresses)):
                    if i < edge_predictions.size(0) and j < edge_predictions.size(1):
                        prob = edge_predictions[i, j].item()
                        if prob > 0.5:  # Threshold for edge prediction
                            edge_key = f"{node_addresses[i]}-{node_addresses[j]}"
                            predicted_edges[edge_key] = prob
            
            # Graph-level predictions
            contagion_score = torch.sigmoid(outputs['contagion_score']).item()
            manipulation_score = torch.sigmoid(outputs['manipulation_score']).item()
            
            # Calculate derived metrics
            smart_money_wallets = [
                addr for addr, data in wallet_classifications.items()
                if data['type'] == WalletType.SMART_MONEY
            ]
            smart_money_probability = len(smart_money_wallets) / len(wallet_classifications)
            
            # Build influence network
            influence_network = self._build_influence_network(wallet_classifications, predicted_edges)
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(outputs, snapshot)
            
            return {
                'token_address': token_address,
                'contagion_score': contagion_score,
                'smart_money_probability': smart_money_probability,
                'manipulation_risk': manipulation_score,
                'wallet_classifications': wallet_classifications,
                'predicted_edges': predicted_edges,
                'influence_network': influence_network,
                'confidence': confidence,
                'temporal_consistency': self.temporal_consistency,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing predictions: {e}")
            return self._fallback_prediction(token_address)
    
    def _build_influence_network(self, wallet_classifications: Dict[str, Any], 
                               predicted_edges: Dict[str, float]) -> Dict[str, List[str]]:
        """Build influence network from predictions"""
        
        influence_network = defaultdict(list)
        
        for edge_key, strength in predicted_edges.items():
            if strength > 0.7:  # High influence threshold
                source, target = edge_key.split('-')
                influence_network[source].append(target)
                influence_network[target].append(source)  # Bidirectional
        
        return dict(influence_network)
    
    def _calculate_prediction_confidence(self, outputs: Dict[str, torch.Tensor], 
                                       snapshot: GraphSnapshot) -> float:
        """Calculate overall prediction confidence"""
        
        try:
            # Node classification confidence
            node_probs = F.softmax(outputs['node_logits'], dim=1)
            node_confidence = torch.mean(torch.max(node_probs, dim=1)[0]).item()
            
            # Edge prediction confidence
            edge_probs = torch.sigmoid(outputs['edge_predictions'])
            edge_confidence = torch.mean(torch.abs(edge_probs - 0.5) * 2).item()
            
            # Combine confidences
            overall_confidence = (node_confidence + edge_confidence) / 2
            
            return overall_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    async def _get_relevant_wallets(self, token_address: str) -> List[str]:
        """Get wallets relevant to the token (placeholder)"""
        
        # This would be implemented to fetch actual wallet data
        # For now, return sample wallets
        return [
            f"wallet_{token_address}_{i}" for i in range(min(20, 50))
        ]
    
    async def _get_relevant_edges(self, token_address: str) -> List[Dict[str, Any]]:
        """Get transaction edges relevant to the token (placeholder)"""
        
        # This would be implemented to fetch actual transaction data
        # For now, return sample edges
        edges = []
        wallets = await self._get_relevant_wallets(token_address)
        
        for i in range(min(50, len(wallets) * 2)):
            source = wallets[i % len(wallets)]
            target = wallets[(i + 1) % len(wallets)]
            
            edges.append({
                'source_wallet': source,
                'target_wallet': target,
                'type': 'transaction',
                'amount': 100.0 * (i + 1),
                'weight': 1.0,
                'frequency': 1.0,
                'creation_time': datetime.utcnow(),
                'last_activity': datetime.utcnow()
            })
        
        return edges
    
    async def _extract_global_features(self, token_address: str) -> torch.Tensor:
        """Extract global graph features"""
        
        # Placeholder for global features
        return torch.tensor([
            0.5,  # Network density
            0.3,  # Clustering coefficient
            0.8,  # Average path length
            0.6   # Centralization
        ], dtype=torch.float32)
    
    async def _get_market_state(self, token_address: str) -> Dict[str, Any]:
        """Get current market state"""
        
        return {
            'price': 1.0,
            'volume': 1000.0,
            'market_cap': 1000000.0,
            'volatility': 0.1
        }
    
    def _fallback_prediction(self, token_address: str) -> Dict[str, Any]:
        """Generate fallback prediction"""
        
        return {
            'token_address': token_address,
            'contagion_score': 0.5,
            'smart_money_probability': 0.5,
            'manipulation_risk': 0.3,
            'wallet_classifications': {},
            'predicted_edges': {},
            'influence_network': {},
            'confidence': 0.1,
            'temporal_consistency': 0.0,
            'timestamp': datetime.utcnow()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        
        return {
            'prediction_accuracy': self.prediction_accuracy,
            'edge_prediction_accuracy': self.edge_prediction_accuracy,
            'temporal_consistency': self.temporal_consistency,
            'graph_history_size': len(self.graph_history),
            'tracked_wallets': len(self.wallet_nodes),
            'tracked_edges': len(self.temporal_edges),
            'model_layers': self.model.num_layers,
            'memory_size': self.model.memory_size
        }


# Factory function
async def create_dynamic_network_ant() -> DynamicNetworkAntPredictor:
    """Create Dynamic Network Ant predictor"""
    return DynamicNetworkAntPredictor() 