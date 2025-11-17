"""
Baseline models for financial time series prediction with uniform input structure.
All models use the same 82-feature input with L=5 historical window.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class BaselineModel(nn.Module):
    """Base class for all baseline models ensuring uniform input structure"""
    
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim  # N=82 features
        self.seq_len = seq_len      # L=5 historical window
        self.hidden_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        All models return (mean, log_variance) for probabilistic evaluation
        Args:
            x: (B, L, N) where B=batch, L=5, N=82
        Returns:
            mean: (B,) predicted returns
            log_var: (B,) prediction uncertainty
        """
        raise NotImplementedError


class LSTMBaseline(BaselineModel):
    """LSTM baseline with same input structure as MAMBA-BGNN"""
    
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__(input_dim, seq_len, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Probabilistic output head
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L=5, N=82)
        lstm_out, _ = self.lstm(x)  # (B, L, hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # (B, hidden_dim)
        
        mean = self.mean_head(last_hidden).squeeze(-1)  # (B,)
        log_var = self.logvar_head(last_hidden).squeeze(-1)  # (B,)
        
        return mean, log_var


class TransformerBaseline(BaselineModel):
    """Transformer baseline with same input structure"""
    
    def __init__(self, input_dim: int, seq_len: int, d_model: int = 64, nhead: int = 8, num_layers: int = 3):
        super().__init__(input_dim, seq_len, d_model)
        
        # Input projection to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_pos_encoding(seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Probabilistic output heads
        self.mean_head = nn.Linear(d_model, 1)
        self.logvar_head = nn.Linear(d_model, 1)
        
    def _create_pos_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L=5, N=82)
        B, L, N = x.shape
        
        # Project to d_model
        x = self.input_proj(x)  # (B, L, d_model)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :L, :].to(x.device)
        x = x + pos_enc
        
        # Transformer encoding
        encoded = self.transformer(x)  # (B, L, d_model)
        
        # Use last token for prediction
        last_token = encoded[:, -1, :]  # (B, d_model)
        
        mean = self.mean_head(last_token).squeeze(-1)  # (B,)
        log_var = self.logvar_head(last_token).squeeze(-1)  # (B,)
        
        return mean, log_var


class AGCRNBaseline(BaselineModel):
    """Adaptive Graph Convolution RNN baseline"""
    
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 64):
        super().__init__(input_dim, seq_len, hidden_dim)
        
        # Adaptive adjacency matrix (learnable)
        self.adaptive_adj = nn.Parameter(torch.randn(input_dim, input_dim) * 0.1)
        
        # GCN layer
        self.gcn = nn.Linear(input_dim, hidden_dim)
        
        # RNN for temporal modeling
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2
        )
        
        # Output heads
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L=5, N=82)
        B, L, N = x.shape
        
        # Apply adaptive graph convolution at each timestep
        gcn_out = []
        adj = torch.softmax(self.adaptive_adj, dim=1)  # Row-normalize
        
        for t in range(L):
            x_t = x[:, t, :]  # (B, N)
            # Graph convolution: X' = A * X * W
            h_t = torch.matmul(x_t, adj)  # (B, N)
            h_t = self.gcn(h_t)  # (B, hidden_dim)
            gcn_out.append(h_t)
        
        # Stack temporal features
        temporal_features = torch.stack(gcn_out, dim=1)  # (B, L, hidden_dim)
        
        # RNN processing
        rnn_out, _ = self.rnn(temporal_features)  # (B, L, hidden_dim)
        last_hidden = rnn_out[:, -1, :]  # (B, hidden_dim)
        
        mean = self.mean_head(last_hidden).squeeze(-1)  # (B,)
        log_var = self.logvar_head(last_hidden).squeeze(-1)  # (B,)
        
        return mean, log_var


class TemporalGNBaseline(BaselineModel):
    """Temporal Graph Network baseline"""
    
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 64):
        super().__init__(input_dim, seq_len, hidden_dim)
        
        # Node embeddings
        self.node_embedding = nn.Parameter(torch.randn(input_dim, hidden_dim))
        
        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Graph layers
        self.graph_conv1 = nn.Linear(input_dim, hidden_dim)
        self.graph_conv2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output heads
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L=5, N=82)
        B, L, N = x.shape
        
        # Graph convolution at each timestep
        graph_features = []
        for t in range(L):
            x_t = x[:, t, :]  # (B, N)
            
            # Two-layer graph convolution
            h1 = F.relu(self.graph_conv1(x_t))  # (B, hidden_dim)
            h2 = self.graph_conv2(h1)  # (B, hidden_dim)
            
            graph_features.append(h2)
        
        # Stack temporal graph features
        temporal_graph = torch.stack(graph_features, dim=1)  # (B, L, hidden_dim)
        
        # Temporal attention
        attn_out, _ = self.temporal_attn(
            temporal_graph, temporal_graph, temporal_graph
        )  # (B, L, hidden_dim)
        
        # Global temporal pooling
        pooled = torch.mean(attn_out, dim=1)  # (B, hidden_dim)
        
        mean = self.mean_head(pooled).squeeze(-1)  # (B,)
        log_var = self.logvar_head(pooled).squeeze(-1)  # (B,)
        
        return mean, log_var


class LinearBaseline(BaselineModel):
    """Simple linear baseline for comparison"""
    
    def __init__(self, input_dim: int, seq_len: int):
        super().__init__(input_dim, seq_len)
        
        # Flatten input and apply linear transformation
        self.linear = nn.Linear(input_dim * seq_len, 64)
        self.mean_head = nn.Linear(64, 1)
        self.logvar_head = nn.Linear(64, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L=5, N=82)
        B, L, N = x.shape
        
        # Flatten
        x_flat = x.view(B, -1)  # (B, L*N)
        
        # Linear transformation
        hidden = F.relu(self.linear(x_flat))  # (B, 64)
        
        mean = self.mean_head(hidden).squeeze(-1)  # (B,)
        log_var = self.logvar_head(hidden).squeeze(-1)  # (B,)
        
        return mean, log_var


def create_baseline_models(input_dim: int = 82, seq_len: int = 5) -> dict:
    """Create all baseline models with uniform input structure"""
    
    models = {
        'Linear': LinearBaseline(input_dim, seq_len),
        'LSTM': LSTMBaseline(input_dim, seq_len, hidden_dim=64, num_layers=2),
        'Transformer': TransformerBaseline(input_dim, seq_len, d_model=64, nhead=8, num_layers=3),
        'AGCRN': AGCRNBaseline(input_dim, seq_len, hidden_dim=64),
        'TemporalGN': TemporalGNBaseline(input_dim, seq_len, hidden_dim=64)
    }
    
    # Initialize parameters
    for name, model in models.items():
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif param.dim() == 1:
                nn.init.zeros_(param)
                
    return models


if __name__ == "__main__":
    # Test baseline models
    models = create_baseline_models()
    
    # Test input (batch_size=32, seq_len=5, input_dim=82)
    x = torch.randn(32, 5, 82)
    
    print("Testing baseline models with uniform input structure:")
    print(f"Input shape: {x.shape} (B, L=5, N=82)")
    print("-" * 60)
    
    for name, model in models.items():
        try:
            mean, log_var = model(x)
            print(f"{name:12} | Mean: {mean.shape} | LogVar: {log_var.shape} | Params: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"{name:12} | ERROR: {e}")