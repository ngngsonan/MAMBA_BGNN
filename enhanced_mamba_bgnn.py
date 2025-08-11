"""
Enhanced MAMBA-BGNN with directional prediction capability.
Addresses reviewer concern about directional accuracy focus.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_bgnn import (
    MambaBlock, BIMambaBlock, BayesianMAGAC, ModelArgs
)
from typing import Tuple


class DirectionalMambaBlock(MambaBlock):
    """Enhanced Mamba block with directional awareness"""
    
    def __init__(self, args: ModelArgs, directional_weight: float = 0.1):
        super().__init__(args)
        self.directional_weight = directional_weight
        
        # Add directional component
        self.directional_proj = nn.Linear(args.d_model, args.d_model // 4)
        self.directional_gate = nn.Linear(args.d_model // 4, args.d_model)
        
    def forward(self, x):
        # Original Mamba processing
        mamba_out = super().forward(x)
        
        # Directional enhancement
        batch_size, seq_len, _ = x.shape
        
        # Calculate directional features (return changes)
        if seq_len > 1:
            # Use price column (assumed to be first feature after normalization)
            price_changes = x[:, 1:, 0] - x[:, :-1, 0]  # (B, L-1)
            
            # Pad to match sequence length
            directional_features = F.pad(price_changes, (0, 1), mode='constant', value=0)  # (B, L)
            directional_features = directional_features.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # (B, L, N)
            
            # Process directional information
            dir_proj = torch.tanh(self.directional_proj(directional_features))  # (B, L, N//4)
            dir_gate = torch.sigmoid(self.directional_gate(dir_proj))  # (B, L, N)
            
            # Combine with original output
            enhanced_out = mamba_out + self.directional_weight * (mamba_out * dir_gate)
            return enhanced_out
        
        return mamba_out


class EnhancedBIMambaBlock(nn.Module):
    """Enhanced bidirectional Mamba with directional awareness"""
    
    def __init__(self, args: ModelArgs, R: int = 3, dropout: float = 0.1, 
                 directional_weight: float = 0.1):
        super().__init__()
        self.R = R
        
        # Use enhanced directional Mamba blocks
        self.f_mamba = nn.ModuleList([DirectionalMambaBlock(args, directional_weight) for _ in range(R)])
        self.b_mamba = nn.ModuleList([DirectionalMambaBlock(args, directional_weight) for _ in range(R)])
        
        self.norm1 = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(R)])
        self.ffn   = nn.ModuleList([self._build_ffn(args, dropout) for _ in range(R)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(R)])
        
    def _build_ffn(self, args: ModelArgs, dropout: float):
        return nn.Sequential(
            nn.Linear(args.d_model, args.d_proj_U if hasattr(args, 'd_proj_U') else 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(args.d_proj_U if hasattr(args, 'd_proj_U') else 32, args.d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        for i in range(self.R):
            # Forward and backward Mamba processing
            Y1 = self.f_mamba[i](x)
            
            x_rev = torch.flip(x, dims=[1]).contiguous()
            Y2_rev = self.b_mamba[i](x_rev)
            Y2 = torch.flip(Y2_rev, dims=[1]).contiguous()
            
            # Residual connections
            Y3 = self.norm1[i](x + Y1 + Y2)
            Yp = self.ffn[i](Y3)
            x = self.norm2[i](Yp + Y3)
            
        return x


class MultiTaskHead(nn.Module):
    """Multi-task head for both return prediction and directional classification"""
    
    def __init__(self, input_dim: int, directional_weight: float = 0.3):
        super().__init__()
        self.directional_weight = directional_weight
        
        # Shared feature extraction
        self.shared_layer = nn.Linear(input_dim, input_dim // 2)
        
        # Return prediction head (regression)
        self.return_mean = nn.Linear(input_dim // 2, 1)
        self.return_logvar = nn.Linear(input_dim // 2, 1)
        
        # Directional prediction head (classification)
        self.direction_logits = nn.Linear(input_dim // 2, 3)  # up, down, neutral
        
        # Cross-task attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim // 2,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor, return_dict: bool = False):
        """
        Args:
            x: (B, N) input features
            return_dict: whether to return detailed outputs
            
        Returns:
            If return_dict=False: (mean, log_var) for compatibility
            If return_dict=True: dict with all outputs
        """
        batch_size = x.shape[0]
        
        # Shared feature extraction
        shared_features = F.relu(self.shared_layer(x))  # (B, N//2)
        
        # Cross-task attention (treat each sample as a sequence of length 1)
        shared_expanded = shared_features.unsqueeze(1)  # (B, 1, N//2)
        attended_features, _ = self.cross_attn(
            shared_expanded, shared_expanded, shared_expanded
        )  # (B, 1, N//2)
        attended_features = attended_features.squeeze(1)  # (B, N//2)
        
        # Return prediction
        return_mean = self.return_mean(attended_features).squeeze(-1)  # (B,)
        return_logvar = self.return_logvar(attended_features).squeeze(-1)  # (B,)
        
        # Directional prediction
        direction_logits = self.direction_logits(attended_features)  # (B, 3)
        direction_probs = F.softmax(direction_logits, dim=-1)  # (B, 3)
        
        if return_dict:
            return {
                'return_mean': return_mean,
                'return_logvar': return_logvar,
                'direction_logits': direction_logits,
                'direction_probs': direction_probs,
                'shared_features': shared_features
            }
        else:
            # For compatibility with existing code
            return return_mean, return_logvar


class MAMBA_BGNN_Enhanced(nn.Module):
    """Enhanced MAMBA-BGNN with directional prediction capability"""
    
    def __init__(self, args: ModelArgs, R: int = 3, K: int = 3,
                 d_e: int = 10, heads=4,
                 mc_train=3, mc_eval=20, drop_edge_p=0.1, mc_dropout_p=0.2,
                 directional_weight: float = 0.3, use_multi_task: bool = True):
        super().__init__()
        
        self.use_multi_task = use_multi_task
        self.directional_weight = directional_weight
        
        # Enhanced bidirectional Mamba with directional awareness
        self.bi_mamba = EnhancedBIMambaBlock(
            args, R=R, directional_weight=directional_weight
        )
        
        # Bayesian graph convolution
        self.agc_bayes = BayesianMAGAC(
            args.d_model, args.seq_len, K, d_e, heads=heads,
            mc_train=mc_train, mc_eval=mc_eval,
            drop_edge_p=drop_edge_p, mc_dropout_p=mc_dropout_p
        )
        
        # Multi-task or single-task head
        if use_multi_task:
            self.head = MultiTaskHead(args.d_model, directional_weight)
        else:
            # Traditional single-task head
            self.head = nn.Linear(args.d_model, 1)
    
    def forward(self, x, return_detailed: bool = False):
        """
        Args:
            x: (B, L, N) input sequences
            return_detailed: whether to return detailed multi-task outputs
            
        Returns:
            If use_multi_task=False or return_detailed=False: (mean, log_var)
            If use_multi_task=True and return_detailed=True: detailed dict
        """
        # Temporal modeling with enhanced Mamba
        y_seq = self.bi_mamba(x)  # (B, L, N)
        
        # Spatial modeling with Bayesian graph convolution
        z_node = y_seq.transpose(1, 2).contiguous()  # (B, N, L)
        g_node, log_var_node = self.agc_bayes(z_node)  # both (B, N)
        
        if self.use_multi_task:
            # Multi-task prediction
            outputs = self.head(g_node, return_dict=return_detailed)
            
            if return_detailed:
                # Add graph uncertainty to return variance
                return_var = outputs['return_logvar'].exp()
                graph_uncertainty = torch.sum(log_var_node.exp(), dim=1)  # (B,)
                total_var = return_var + graph_uncertainty
                outputs['return_logvar'] = total_var.log()
                return outputs
            else:
                # Compatibility mode
                return_mean, return_logvar = outputs
                # Add graph uncertainty
                return_var = return_logvar.exp()
                graph_uncertainty = torch.sum(log_var_node.exp(), dim=1)
                total_var = return_var + graph_uncertainty + 1e-6
                return return_mean, total_var.log()
        else:
            # Traditional single-task prediction
            w = self.head.weight.squeeze(0)  # (N,)
            b = self.head.bias  # (1,)
            
            mu = torch.einsum('bn,n->b', g_node, w) + b
            var = torch.einsum('bn,n->b', log_var_node.exp(), w.pow(2)) + 1e-6
            log_var = var.log()
            
            return mu, log_var


class DirectionalLoss(nn.Module):
    """Combined loss for return prediction and directional classification"""
    
    def __init__(self, return_weight: float = 0.7, direction_weight: float = 0.3,
                 direction_threshold: float = 0.001):
        super().__init__()
        self.return_weight = return_weight
        self.direction_weight = direction_weight
        self.direction_threshold = direction_threshold
        
        self.return_loss = nn.GaussianNLLLoss(full=True, reduction='mean')
        self.direction_loss = nn.CrossEntropyLoss()
    
    def _get_direction_labels(self, returns: torch.Tensor) -> torch.Tensor:
        """Convert returns to directional labels: 0=down, 1=neutral, 2=up"""
        labels = torch.zeros_like(returns, dtype=torch.long)
        labels[returns > self.direction_threshold] = 2  # up
        labels[returns < -self.direction_threshold] = 0  # down
        labels[torch.abs(returns) <= self.direction_threshold] = 1  # neutral
        return labels
    
    def forward(self, outputs: dict, true_returns: torch.Tensor) -> dict:
        """
        Args:
            outputs: dict from enhanced model with return and direction predictions
            true_returns: (B,) true return values
            
        Returns:
            dict with individual losses and total loss
        """
        # Return prediction loss
        return_loss = self.return_loss(
            outputs['return_mean'], 
            true_returns, 
            outputs['return_logvar'].exp()
        )
        
        # Directional classification loss
        direction_labels = self._get_direction_labels(true_returns)
        direction_loss = self.direction_loss(outputs['direction_logits'], direction_labels)
        
        # Combined loss
        total_loss = (self.return_weight * return_loss + 
                     self.direction_weight * direction_loss)
        
        return {
            'total_loss': total_loss,
            'return_loss': return_loss,
            'direction_loss': direction_loss,
            'direction_accuracy': self._calculate_direction_accuracy(
                outputs['direction_logits'], direction_labels
            )
        }
    
    def _calculate_direction_accuracy(self, logits: torch.Tensor, 
                                    labels: torch.Tensor) -> float:
        """Calculate directional prediction accuracy"""
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).float()
        return correct.mean().item()


def create_enhanced_model(args: ModelArgs, **kwargs) -> MAMBA_BGNN_Enhanced:
    """Factory function to create enhanced model with proper initialization"""
    
    model = MAMBA_BGNN_Enhanced(args, **kwargs)
    
    # Initialize parameters
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif param.dim() == 1:
            nn.init.zeros_(param)
    
    return model


if __name__ == "__main__":
    # Test enhanced model
    print("Testing Enhanced MAMBA-BGNN with directional prediction...")
    
    # Test parameters
    batch_size, seq_len, num_features = 16, 5, 82
    
    args = ModelArgs(d_model=num_features, seq_len=seq_len, d_state=128)
    
    # Create models
    models = {
        'Enhanced_MultiTask': create_enhanced_model(
            args, use_multi_task=True, directional_weight=0.3
        ),
        'Enhanced_SingleTask': create_enhanced_model(
            args, use_multi_task=False, directional_weight=0.3
        )
    }
    
    # Test input
    x = torch.randn(batch_size, seq_len, num_features)
    y_true = torch.randn(batch_size) * 0.02  # Simulated returns
    
    print(f"Input shape: {x.shape}")
    print("-" * 60)
    
    for name, model in models.items():
        print(f"\nTesting {name}:")
        
        if name == 'Enhanced_MultiTask':
            # Test detailed output
            detailed_outputs = model(x, return_detailed=True)
            print(f"  Return mean shape: {detailed_outputs['return_mean'].shape}")
            print(f"  Return logvar shape: {detailed_outputs['return_logvar'].shape}")
            print(f"  Direction logits shape: {detailed_outputs['direction_logits'].shape}")
            print(f"  Direction probs shape: {detailed_outputs['direction_probs'].shape}")
            
            # Test loss
            loss_fn = DirectionalLoss()
            loss_dict = loss_fn(detailed_outputs, y_true)
            print(f"  Total loss: {loss_dict['total_loss']:.6f}")
            print(f"  Return loss: {loss_dict['return_loss']:.6f}")
            print(f"  Direction loss: {loss_dict['direction_loss']:.6f}")
            print(f"  Direction accuracy: {loss_dict['direction_accuracy']:.4f}")
            
            # Test compatibility mode
            mean, logvar = model(x, return_detailed=False)
            print(f"  Compatibility - Mean: {mean.shape}, LogVar: {logvar.shape}")
        
        else:
            # Single-task model
            mean, logvar = model(x)
            print(f"  Mean shape: {mean.shape}")
            print(f"  LogVar shape: {logvar.shape}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
    
    print("\nEnhanced model testing complete!")