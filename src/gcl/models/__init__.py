"""Contrastive learning models for graphs."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, dropout_node

from .layers import GCNLayer, GATLayer, SAGELayer, MLP


class GraphEncoder(nn.Module):
    """Base graph encoder for contrastive learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        layer_type: str = "gcn",
        dropout: float = 0.5,
        use_bn: bool = False,
        use_residual: bool = False,
    ):
        """Initialize graph encoder.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output dimension.
            num_layers: Number of GNN layers.
            layer_type: Type of GNN layer ("gcn", "gat", "sage").
            dropout: Dropout probability.
            use_bn: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            if layer_type == "gcn":
                layer = GCNLayer(
                    dims[i], dims[i + 1], dropout, use_bn, use_residual
                )
            elif layer_type == "gat":
                layer = GATLayer(
                    dims[i], dims[i + 1], heads=8, dropout=dropout,
                    use_bn=use_bn, use_residual=use_residual
                )
            elif layer_type == "sage":
                layer = SAGELayer(
                    dims[i], dims[i + 1], dropout, use_bn, use_residual
                )
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
            self.layers.append(layer)
    
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
            torch.Tensor: Node embeddings.
        """
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        
        return x


class GraphCLModel(nn.Module):
    """Graph Contrastive Learning model."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        layer_type: str = "gcn",
        dropout: float = 0.5,
        use_bn: bool = False,
        use_residual: bool = False,
        projection_dim: int = 128,
        temperature: float = 0.5,
    ):
        """Initialize GraphCL model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output dimension.
            num_layers: Number of GNN layers.
            layer_type: Type of GNN layer.
            dropout: Dropout probability.
            use_bn: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
            projection_dim: Projection head dimension.
            temperature: Temperature for contrastive loss.
        """
        super().__init__()
        
        self.encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            layer_type=layer_type,
            dropout=dropout,
            use_bn=use_bn,
            use_residual=use_residual,
        )
        
        self.projection_head = MLP(
            input_dim=output_dim,
            hidden_dim=projection_dim,
            output_dim=projection_dim,
            num_layers=2,
            dropout=dropout,
            use_bn=use_bn,
        )
        
        self.temperature = temperature
    
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
            torch.Tensor: Node embeddings.
        """
        h = self.encoder(x, edge_index, edge_weight)
        z = self.projection_head(h)
        return F.normalize(z, dim=1)
    
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode nodes without projection head.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_weight: Optional edge weights.
            
        Returns:
            torch.Tensor: Node embeddings.
        """
        return self.encoder(x, edge_index, edge_weight)
    
    def augment(
        self,
        data: Data,
        node_drop_rate: float = 0.2,
        edge_drop_rate: float = 0.1,
        feature_mask_rate: float = 0.1,
    ) -> Data:
        """Apply graph augmentations.
        
        Args:
            data: Input graph data.
            node_drop_rate: Node dropout rate.
            edge_drop_rate: Edge dropout rate.
            feature_mask_rate: Feature masking rate.
            
        Returns:
            Data: Augmented graph data.
        """
        x, edge_index = data.x, data.edge_index
        
        # Node dropout
        if node_drop_rate > 0:
            x, edge_index, _, _ = dropout_node(
                x, edge_index, p=node_drop_rate, training=self.training
            )
        
        # Edge dropout
        if edge_drop_rate > 0:
            edge_index, _ = dropout_edge(
                edge_index, p=edge_drop_rate, training=self.training
            )
        
        # Feature masking
        if feature_mask_rate > 0 and self.training:
            mask = torch.rand(x.size(1)) < feature_mask_rate
            x = x.clone()
            x[:, mask] = 0
        
        return Data(x=x, edge_index=edge_index, y=data.y)


class GRACEModel(nn.Module):
    """GRACE (Graph Contrastive Learning with Adaptive Augmentation) model."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        layer_type: str = "gcn",
        dropout: float = 0.5,
        use_bn: bool = False,
        use_residual: bool = False,
        projection_dim: int = 128,
        temperature: float = 0.5,
    ):
        """Initialize GRACE model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output dimension.
            num_layers: Number of GNN layers.
            layer_type: Type of GNN layer.
            dropout: Dropout probability.
            use_bn: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
            projection_dim: Projection head dimension.
            temperature: Temperature for contrastive loss.
        """
        super().__init__()
        
        self.encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            layer_type=layer_type,
            dropout=dropout,
            use_bn=use_bn,
            use_residual=use_residual,
        )
        
        self.projection_head = MLP(
            input_dim=output_dim,
            hidden_dim=projection_dim,
            output_dim=projection_dim,
            num_layers=2,
            dropout=dropout,
            use_bn=use_bn,
        )
        
        self.temperature = temperature
    
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
            torch.Tensor: Node embeddings.
        """
        h = self.encoder(x, edge_index, edge_weight)
        z = self.projection_head(h)
        return F.normalize(z, dim=1)
    
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode nodes without projection head.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_weight: Optional edge weights.
            
        Returns:
            torch.Tensor: Node embeddings.
        """
        return self.encoder(x, edge_index, edge_weight)
    
    def augment(
        self,
        data: Data,
        edge_drop_rate: float = 0.1,
        feature_corruption_rate: float = 0.1,
    ) -> Data:
        """Apply GRACE-specific augmentations.
        
        Args:
            data: Input graph data.
            edge_drop_rate: Edge dropout rate.
            feature_corruption_rate: Feature corruption rate.
            
        Returns:
            Data: Augmented graph data.
        """
        x, edge_index = data.x, data.edge_index
        
        # Edge dropout
        if edge_drop_rate > 0:
            edge_index, _ = dropout_edge(
                edge_index, p=edge_drop_rate, training=self.training
            )
        
        # Feature corruption
        if feature_corruption_rate > 0 and self.training:
            x = x.clone()
            mask = torch.rand(x.size(0)) < feature_corruption_rate
            x[mask] = torch.randn_like(x[mask])
        
        return Data(x=x, edge_index=edge_index, y=data.y)


class BGRLModel(nn.Module):
    """Bootstrapped Graph Representation Learning model."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        layer_type: str = "gcn",
        dropout: float = 0.5,
        use_bn: bool = False,
        use_residual: bool = False,
        projection_dim: int = 128,
        temperature: float = 0.5,
        momentum: float = 0.999,
    ):
        """Initialize BGRL model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output dimension.
            num_layers: Number of GNN layers.
            layer_type: Type of GNN layer.
            dropout: Dropout probability.
            use_bn: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
            projection_dim: Projection head dimension.
            temperature: Temperature for contrastive loss.
            momentum: Momentum for EMA updates.
        """
        super().__init__()
        
        self.online_encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            layer_type=layer_type,
            dropout=dropout,
            use_bn=use_bn,
            use_residual=use_residual,
        )
        
        self.target_encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            layer_type=layer_type,
            dropout=dropout,
            use_bn=use_bn,
            use_residual=use_residual,
        )
        
        self.online_projection = MLP(
            input_dim=output_dim,
            hidden_dim=projection_dim,
            output_dim=projection_dim,
            num_layers=2,
            dropout=dropout,
            use_bn=use_bn,
        )
        
        self.target_projection = MLP(
            input_dim=output_dim,
            hidden_dim=projection_dim,
            output_dim=projection_dim,
            num_layers=2,
            dropout=dropout,
            use_bn=use_bn,
        )
        
        self.temperature = temperature
        self.momentum = momentum
        
        # Initialize target encoder with online encoder weights
        self._init_target_encoder()
    
    def _init_target_encoder(self) -> None:
        """Initialize target encoder with online encoder weights."""
        for target_param, online_param in zip(
            self.target_encoder.parameters(), self.online_encoder.parameters()
        ):
            target_param.data.copy_(online_param.data)
        
        for target_param, online_param in zip(
            self.target_projection.parameters(), self.online_projection.parameters()
        ):
            target_param.data.copy_(online_param.data)
    
    def _update_target_encoder(self) -> None:
        """Update target encoder with momentum."""
        for target_param, online_param in zip(
            self.target_encoder.parameters(), self.online_encoder.parameters()
        ):
            target_param.data = (
                self.momentum * target_param.data + (1 - self.momentum) * online_param.data
            )
        
        for target_param, online_param in zip(
            self.target_projection.parameters(), self.online_projection.parameters()
        ):
            target_param.data = (
                self.momentum * target_param.data + (1 - self.momentum) * online_param.data
            )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_weight: Optional edge weights.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Online and target embeddings.
        """
        # Online encoder
        h_online = self.online_encoder(x, edge_index, edge_weight)
        z_online = self.online_projection(h_online)
        z_online = F.normalize(z_online, dim=1)
        
        # Target encoder
        with torch.no_grad():
            h_target = self.target_encoder(x, edge_index, edge_weight)
            z_target = self.target_projection(h_target)
            z_target = F.normalize(z_target, dim=1)
        
        return z_online, z_target
    
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode nodes using online encoder.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_weight: Optional edge weights.
            
        Returns:
            torch.Tensor: Node embeddings.
        """
        return self.online_encoder(x, edge_index, edge_weight)
    
    def augment(
        self,
        data: Data,
        node_drop_rate: float = 0.2,
        edge_drop_rate: float = 0.1,
    ) -> Data:
        """Apply augmentations for BGRL.
        
        Args:
            data: Input graph data.
            node_drop_rate: Node dropout rate.
            edge_drop_rate: Edge dropout rate.
            
        Returns:
            Data: Augmented graph data.
        """
        x, edge_index = data.x, data.edge_index
        
        # Node dropout
        if node_drop_rate > 0:
            x, edge_index, _, _ = dropout_node(
                x, edge_index, p=node_drop_rate, training=self.training
            )
        
        # Edge dropout
        if edge_drop_rate > 0:
            edge_index, _ = dropout_edge(
                edge_index, p=edge_drop_rate, training=self.training
            )
        
        return Data(x=x, edge_index=edge_index, y=data.y)
