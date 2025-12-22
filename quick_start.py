#!/usr/bin/env python3
"""Quick start script for Graph Contrastive Learning project."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dropout_node

from gcl.models import GraphCLModel
from gcl.utils import set_seed, get_device


def create_demo_graph():
    """Create a simple demo graph."""
    # Create a small graph with 10 nodes
    num_nodes = 10
    num_features = 5
    num_classes = 3
    
    # Random node features
    x = torch.randn(num_nodes, num_features)
    
    # Random node labels
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # Create a simple ring graph
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    ])
    
    # Train/val/test masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[:6] = True
    val_mask[6:8] = True
    test_mask[8:] = True
    
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def main():
    """Main demo function."""
    print("Graph Contrastive Learning - Quick Start Demo")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create demo graph
    print("\nCreating demo graph...")
    data = create_demo_graph()
    data = data.to(device)
    
    print(f"Graph info:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {data.num_node_features}")
    print(f"  Classes: {int(data.y.max().item()) + 1}")
    
    # Initialize GraphCL model
    print("\nInitializing GraphCL model...")
    model = GraphCLModel(
        input_dim=data.num_node_features,
        hidden_dim=16,
        output_dim=16,
        num_layers=2,
        layer_type="gcn",
        dropout=0.1,
        use_bn=True,
        use_residual=True,
        projection_dim=8,
        temperature=0.5,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("\nTraining GraphCL model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        
        # Create two augmented views
        view1, _ = dropout_node(data, p=0.1)
        view2, _ = dropout_node(data, p=0.1)
        
        # Forward pass
        z1 = model(view1.x, view1.edge_index)
        z2 = model(view2.x, view2.edge_index)
        
        # Contrastive loss
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        sim_matrix = torch.matmul(z1, z2.T) / 0.5
        labels = torch.arange(z1.size(0), device=device)
        
        loss_12 = F.cross_entropy(sim_matrix, labels)
        loss_21 = F.cross_entropy(sim_matrix.T, labels)
        loss = (loss_12 + loss_21) / 2
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}, Loss: {loss.item():.4f}")
    
    # Evaluation
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Simple node classification evaluation
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    clf = LogisticRegression(random_state=42)
    clf.fit(embeddings[data.train_mask].cpu().numpy(), data.y[data.train_mask].cpu().numpy())
    
    train_pred = clf.predict(embeddings[data.train_mask].cpu().numpy())
    test_pred = clf.predict(embeddings[data.test_mask].cpu().numpy())
    
    train_acc = accuracy_score(data.y[data.train_mask].cpu().numpy(), train_pred)
    test_acc = accuracy_score(data.y[data.test_mask].cpu().numpy(), test_pred)
    
    print(f"Node Classification Results:")
    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    
    # Contrastive learning metrics
    print(f"\nContrastive Learning Metrics:")
    
    # Alignment: average similarity of positive pairs (same class)
    alignment_scores = []
    for i in range(len(data.y)):
        same_class_mask = data.y == data.y[i]
        same_class_mask[i] = False  # Exclude self
        if same_class_mask.any():
            sim = torch.cosine_similarity(embeddings[i:i+1], embeddings[same_class_mask]).mean()
            alignment_scores.append(sim.item())
    
    alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
    
    # Uniformity: negative log of average pairwise similarity
    sim_matrix = torch.matmul(embeddings, embeddings.T)
    mask = torch.eye(len(embeddings), dtype=torch.bool, device=device)
    off_diagonal_sim = sim_matrix[~mask]
    uniformity = -torch.log(torch.mean(torch.exp(off_diagonal_sim / 0.5))).item()
    
    print(f"  Alignment: {alignment:.3f}")
    print(f"  Uniformity: {uniformity:.3f}")
    
    print("\nDemo completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python scripts/demo.py --dataset synthetic --epochs 100 --eval' for a full demo")
    print("2. Run 'streamlit run demo/app.py' for the interactive web demo")
    print("3. Run 'python scripts/train.py --config configs/graphcl_cora.yaml --eval' to train on Cora dataset")


if __name__ == "__main__":
    main()
