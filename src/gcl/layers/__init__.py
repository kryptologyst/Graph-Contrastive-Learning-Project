"""Custom GNN layers for contrastive learning."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer with optional normalization and dropout."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_bn: bool = False,
        use_residual: bool = False,
    ):
        """Initialize GCN layer.
        
        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            dropout: Dropout probability.
            use_bn: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
        """
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.use_bn = use_bn
        self.use_residual = use_residual
        
        if use_bn:
            self.bn = nn.BatchNorm1d(out_channels)
        
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_weight: Optional edge weights.
            
        Returns:
            torch.Tensor: Output node features.
        """
        residual = x
        
        x = self.conv(x, edge_index, edge_weight)
        
        if self.use_bn:
            x = self.bn(x)
        
        x = F.relu(x)
        x = self.dropout(x)
        
        if self.use_residual:
            if residual.size(-1) != x.size(-1):
                residual = self.residual_proj(residual)
            x = x + residual
        
        return x


class GATLayer(nn.Module):
    """Graph Attention Network layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        use_bn: bool = False,
        use_residual: bool = False,
    ):
        """Initialize GAT layer.
        
        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            heads: Number of attention heads.
            dropout: Dropout probability.
            use_bn: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
        """
        super().__init__()
        self.conv = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.use_bn = use_bn
        self.use_residual = use_residual
        
        if use_bn:
            self.bn = nn.BatchNorm1d(out_channels * heads)
        
        if use_residual and in_channels != out_channels * heads:
            self.residual_proj = nn.Linear(in_channels, out_channels * heads)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_weight: Optional edge weights.
            
        Returns:
            torch.Tensor: Output node features.
        """
        residual = x
        
        x = self.conv(x, edge_index, edge_weight)
        
        if self.use_bn:
            x = self.bn(x)
        
        x = F.relu(x)
        x = self.dropout(x)
        
        if self.use_residual:
            if residual.size(-1) != x.size(-1):
                residual = self.residual_proj(residual)
            x = x + residual
        
        return x


class SAGELayer(nn.Module):
    """GraphSAGE layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_bn: bool = False,
        use_residual: bool = False,
    ):
        """Initialize SAGE layer.
        
        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            dropout: Dropout probability.
            use_bn: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
        """
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.use_bn = use_bn
        self.use_residual = use_residual
        
        if use_bn:
            self.bn = nn.BatchNorm1d(out_channels)
        
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_weight: Optional edge weights.
            
        Returns:
            torch.Tensor: Output node features.
        """
        residual = x
        
        x = self.conv(x, edge_index, edge_weight)
        
        if self.use_bn:
            x = self.bn(x)
        
        x = F.relu(x)
        x = self.dropout(x)
        
        if self.use_residual:
            if residual.size(-1) != x.size(-1):
                residual = self.residual_proj(residual)
            x = x + residual
        
        return x


class MLP(nn.Module):
    """Multi-layer perceptron with optional batch normalization and dropout."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_bn: bool = False,
        activation: str = "relu",
    ):
        """Initialize MLP.
        
        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output dimension.
            num_layers: Number of layers.
            dropout: Dropout probability.
            use_bn: Whether to use batch normalization.
            activation: Activation function name.
        """
        super().__init__()
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if i < num_layers - 1:  # Don't apply activation/norm to last layer
                if use_bn:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.layers(x)
