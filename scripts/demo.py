#!/usr/bin/env python3
"""Modernized Graph Contrastive Learning implementation.

This script demonstrates a clean, production-ready implementation of Graph Contrastive Learning
using PyTorch Geometric. It includes proper error handling, type hints, and modern Python practices.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dropout_node

from src.gcl.models import GraphCLModel
from src.gcl.train import train_contrastive_model, evaluate_contrastive_model
from src.gcl.eval import evaluate_model, create_evaluation_report
from src.gcl.utils import set_seed, get_device, count_parameters


def create_synthetic_data(
    num_nodes: int = 100,
    num_features: int = 50,
    num_classes: int = 3,
    edge_prob: float = 0.1,
    seed: int = 42,
) -> Data:
    """Create synthetic graph data for demonstration.
    
    Args:
        num_nodes: Number of nodes in the graph.
        num_features: Number of features per node.
        num_classes: Number of classes for node classification.
        edge_prob: Probability of edge existence.
        seed: Random seed for reproducibility.
        
    Returns:
        Data: Synthetic graph data.
    """
    torch.manual_seed(seed)
    
    # Generate random node features
    x = torch.randn(num_nodes, num_features)
    
    # Generate random node labels
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # Generate random edges
    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if torch.rand(1) < edge_prob:
                edge_index.append([i, j])
                edge_index.append([j, i])  # Undirected graph
    
    edge_index = torch.tensor(edge_index).T if edge_index else torch.empty((2, 0), dtype=torch.long)
    
    # Create train/val/test masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Simple split: 60% train, 20% val, 20% test
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask[:train_size] = True
    val_mask[train_size:train_size + val_size] = True
    test_mask[train_size + val_size:] = True
    
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def load_cora_data(root: str = "data") -> Tuple[Data, int, int]:
    """Load Cora dataset.
    
    Args:
        root: Root directory for data storage.
        
    Returns:
        Tuple[Data, int, int]: Data, number of features, number of classes.
    """
    try:
        dataset = Planetoid(root=root, name="Cora")
        data = dataset[0]
        
        # Ensure undirected graph
        from torch_geometric.utils import to_undirected
        data.edge_index = to_undirected(data.edge_index)
        
        return data, dataset.num_node_features, dataset.num_classes
    except Exception as e:
        print(f"Failed to load Cora dataset: {e}")
        print("Using synthetic data instead...")
        return None, None, None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Graph Contrastive Learning Demo")
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "synthetic"], help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--output_dim", type=int, default=64, help="Output dimension")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for contrastive loss")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("--save_model", action="store_true", help="Save trained model")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    if args.dataset == "cora":
        data, num_features, num_classes = load_cora_data()
        if data is None:
            print("Creating synthetic data...")
            data = create_synthetic_data()
            num_features = data.num_node_features
            num_classes = int(data.y.max().item()) + 1
    else:
        print("Creating synthetic data...")
        data = create_synthetic_data()
        num_features = data.num_node_features
        num_classes = int(data.y.max().item()) + 1
    
    data = data.to(device)
    
    print(f"Dataset info:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {num_features}")
    print(f"  Classes: {num_classes}")
    
    # Initialize model
    model = GraphCLModel(
        input_dim=num_features,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=2,
        layer_type="gcn",
        dropout=0.5,
        use_bn=False,
        use_residual=False,
        projection_dim=128,
        temperature=args.temperature,
    )
    
    print(f"Model: GraphCL")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Training
    print("\nStarting training...")
    history = train_contrastive_model(
        model=model,
        data=data,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-4,
        temperature=args.temperature,
        device=device,
        save_path="checkpoints/graphcl_demo.pth" if args.save_model else None,
        log_interval=10,
        augmentation={
            "node_drop_rate": 0.2,
            "edge_drop_rate": 0.1,
            "feature_mask_rate": 0.1,
        },
    )
    
    # Evaluation
    if args.eval:
        print("\nRunning evaluation...")
        
        # Load best model if saved
        if args.save_model and os.path.exists("checkpoints/graphcl_demo.pth"):
            model.load_state_dict(torch.load("checkpoints/graphcl_demo.pth"))
            print("Loaded best model")
        
        # Evaluate on all tasks
        results = evaluate_model(
            model=model,
            data=data,
            tasks=["node_classification", "link_prediction", "clustering"],
            device=device,
        )
        
        # Create report
        report = create_evaluation_report(
            results=results,
            model_name="GraphCL",
            dataset_name=args.dataset.title(),
        )
        
        print(report)
        
        # Save results
        os.makedirs("results", exist_ok=True)
        with open("results/evaluation_report.txt", "w") as f:
            f.write(report)
        print("Results saved to results/evaluation_report.txt")
        
        # Save embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(data.x, data.edge_index)
        
        torch.save(embeddings.cpu(), "results/embeddings.pt")
        print("Embeddings saved to results/embeddings.pt")
        
        # Create visualization
        try:
            from src.gcl.eval import visualize_embeddings
            
            visualize_embeddings(
                embeddings=embeddings,
                labels=data.y,
                save_path="results/embeddings_visualization.png",
                title=f"GraphCL Embeddings on {args.dataset.title()}",
            )
            print("Visualization saved to results/embeddings_visualization.png")
        except ImportError:
            print("Matplotlib not available, skipping visualization")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
