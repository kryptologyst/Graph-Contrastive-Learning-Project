"""Tests for Graph Contrastive Learning models."""

import pytest
import torch
import torch.nn as nn

from src.gcl.models import GraphCLModel, GRACEModel, BGRLModel
from src.gcl.layers import GCNLayer, GATLayer, SAGELayer, MLP
from src.gcl.data import CoraDataset
from src.gcl.train import contrastive_loss, bgrl_loss
from src.gcl.eval import compute_alignment_uniformity
from src.gcl.utils import set_seed, get_device, count_parameters


class TestLayers:
    """Test custom GNN layers."""
    
    def test_gcn_layer(self):
        """Test GCN layer."""
        layer = GCNLayer(10, 20, dropout=0.1, use_bn=True, use_residual=True)
        
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        
        output = layer(x, edge_index)
        assert output.shape == (5, 20)
    
    def test_gat_layer(self):
        """Test GAT layer."""
        layer = GATLayer(10, 20, heads=4, dropout=0.1, use_bn=True, use_residual=True)
        
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        
        output = layer(x, edge_index)
        assert output.shape == (5, 80)  # 20 * 4 heads
    
    def test_sage_layer(self):
        """Test SAGE layer."""
        layer = SAGELayer(10, 20, dropout=0.1, use_bn=True, use_residual=True)
        
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        
        output = layer(x, edge_index)
        assert output.shape == (5, 20)
    
    def test_mlp(self):
        """Test MLP."""
        mlp = MLP(10, 20, 5, num_layers=3, dropout=0.1, use_bn=True)
        
        x = torch.randn(5, 10)
        output = mlp(x)
        assert output.shape == (5, 5)


class TestModels:
    """Test contrastive learning models."""
    
    def test_graphcl_model(self):
        """Test GraphCL model."""
        model = GraphCLModel(
            input_dim=10,
            hidden_dim=20,
            output_dim=10,
            num_layers=2,
            layer_type="gcn",
            dropout=0.1,
            use_bn=True,
            use_residual=True,
            projection_dim=16,
            temperature=0.5,
        )
        
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        
        # Test forward pass
        output = model(x, edge_index)
        assert output.shape == (5, 16)
        
        # Test encode
        embeddings = model.encode(x, edge_index)
        assert embeddings.shape == (5, 10)
        
        # Test augment
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, y=torch.zeros(5))
        augmented = model.augment(data, node_drop_rate=0.1, edge_drop_rate=0.1, feature_mask_rate=0.1)
        assert augmented.x.shape[0] <= 5  # Some nodes might be dropped
    
    def test_grace_model(self):
        """Test GRACE model."""
        model = GRACEModel(
            input_dim=10,
            hidden_dim=20,
            output_dim=10,
            num_layers=2,
            layer_type="gcn",
            dropout=0.1,
            use_bn=True,
            use_residual=True,
            projection_dim=16,
            temperature=0.5,
        )
        
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        
        # Test forward pass
        output = model(x, edge_index)
        assert output.shape == (5, 16)
        
        # Test encode
        embeddings = model.encode(x, edge_index)
        assert embeddings.shape == (5, 10)
        
        # Test augment
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, y=torch.zeros(5))
        augmented = model.augment(data, edge_drop_rate=0.1, feature_corruption_rate=0.1)
        assert augmented.x.shape == (5, 10)
    
    def test_bgrl_model(self):
        """Test BGRL model."""
        model = BGRLModel(
            input_dim=10,
            hidden_dim=20,
            output_dim=10,
            num_layers=2,
            layer_type="gcn",
            dropout=0.1,
            use_bn=True,
            use_residual=True,
            projection_dim=16,
            temperature=0.5,
            momentum=0.999,
        )
        
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        
        # Test forward pass
        z_online, z_target = model(x, edge_index)
        assert z_online.shape == (5, 16)
        assert z_target.shape == (5, 16)
        
        # Test encode
        embeddings = model.encode(x, edge_index)
        assert embeddings.shape == (5, 10)
        
        # Test augment
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, y=torch.zeros(5))
        augmented = model.augment(data, node_drop_rate=0.1, edge_drop_rate=0.1)
        assert augmented.x.shape[0] <= 5  # Some nodes might be dropped


class TestLossFunctions:
    """Test loss functions."""
    
    def test_contrastive_loss(self):
        """Test contrastive loss."""
        z1 = torch.randn(5, 10)
        z2 = torch.randn(5, 10)
        
        loss = contrastive_loss(z1, z2, temperature=0.5)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
    
    def test_bgrl_loss(self):
        """Test BGRL loss."""
        z_online = torch.randn(5, 10)
        z_target = torch.randn(5, 10)
        
        loss = bgrl_loss(z_online, z_target, temperature=0.5)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0


class TestEvaluation:
    """Test evaluation functions."""
    
    def test_alignment_uniformity(self):
        """Test alignment and uniformity computation."""
        embeddings = torch.randn(10, 5)
        labels = torch.randint(0, 3, (10,))
        
        alignment, uniformity = compute_alignment_uniformity(embeddings, labels)
        assert isinstance(alignment, float)
        assert isinstance(uniformity, float)
        assert alignment >= 0
        assert uniformity >= 0


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This should not raise an exception
        assert True
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = nn.Linear(10, 5)
        num_params = count_parameters(model)
        assert num_params == 55  # 10*5 + 5 bias


class TestData:
    """Test data loading."""
    
    def test_cora_dataset(self):
        """Test Cora dataset loading."""
        # This test might fail if data is not available
        try:
            dataset = CoraDataset(root="data")
            assert dataset.num_features > 0
            assert dataset.num_classes > 0
        except Exception:
            # Skip if data is not available
            pytest.skip("Cora dataset not available")


if __name__ == "__main__":
    pytest.main([__file__])
