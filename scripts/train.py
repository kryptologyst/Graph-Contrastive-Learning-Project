#!/usr/bin/env python3
"""Training script for Graph Contrastive Learning models."""

import argparse
import os
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from src.gcl.models import GraphCLModel, GRACEModel, BGRLModel
from src.gcl.data import load_dataset
from src.gcl.train import train_contrastive_model, evaluate_contrastive_model
from src.gcl.eval import evaluate_model, create_evaluation_report
from src.gcl.utils import set_seed, get_device


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Graph Contrastive Learning models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.device != "auto":
        config.device = args.device
    if args.seed != 42:
        config.seed = args.seed
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {config.data.name}")
    dataset = load_dataset(config.data.name, config.data.root)
    data = dataset.data.to(device)
    
    print(f"Dataset info:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {data.num_node_features}")
    print(f"  Classes: {dataset.num_classes}")
    
    # Initialize model
    model_config = config.model
    if model_config.name == "GraphCL":
        model = GraphCLModel(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            output_dim=model_config.output_dim,
            num_layers=model_config.num_layers,
            layer_type=model_config.layer_type,
            dropout=model_config.dropout,
            use_bn=model_config.use_bn,
            use_residual=model_config.use_residual,
            projection_dim=model_config.projection_dim,
            temperature=model_config.temperature,
        )
    elif model_config.name == "GRACE":
        model = GRACEModel(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            output_dim=model_config.output_dim,
            num_layers=model_config.num_layers,
            layer_type=model_config.layer_type,
            dropout=model_config.dropout,
            use_bn=model_config.use_bn,
            use_residual=model_config.use_residual,
            projection_dim=model_config.projection_dim,
            temperature=model_config.temperature,
        )
    elif model_config.name == "BGRL":
        model = BGRLModel(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            output_dim=model_config.output_dim,
            num_layers=model_config.num_layers,
            layer_type=model_config.layer_type,
            dropout=model_config.dropout,
            use_bn=model_config.use_bn,
            use_residual=model_config.use_residual,
            projection_dim=model_config.projection_dim,
            temperature=model_config.temperature,
            momentum=model_config.momentum,
        )
    else:
        raise ValueError(f"Unknown model: {model_config.name}")
    
    print(f"Model: {model_config.name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    print("\nStarting training...")
    history = train_contrastive_model(
        model=model,
        data=data,
        epochs=config.training.epochs,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        temperature=model_config.temperature,
        device=device,
        save_path=config.training.save_path,
        log_interval=config.training.log_interval,
        augmentation=config.data.augmentation,
    )
    
    # Evaluation
    if args.eval:
        print("\nRunning evaluation...")
        
        # Load best model
        if os.path.exists(config.training.save_path):
            model.load_state_dict(torch.load(config.training.save_path))
            print(f"Loaded best model from {config.training.save_path}")
        
        # Evaluate on all tasks
        results = evaluate_model(
            model=model,
            data=data,
            tasks=config.evaluation.tasks,
            device=device,
        )
        
        # Create report
        report = create_evaluation_report(
            results=results,
            model_name=model_config.name,
            dataset_name=config.data.name,
        )
        
        print(report)
        
        # Save results
        results_path = Path(config.training.save_path).parent / "evaluation_results.txt"
        with open(results_path, "w") as f:
            f.write(report)
        print(f"Results saved to {results_path}")
        
        # Save embeddings if requested
        if config.evaluation.save_embeddings:
            model.eval()
            with torch.no_grad():
                if isinstance(model, BGRLModel):
                    embeddings = model.encode(data.x, data.edge_index)
                else:
                    embeddings = model.encode(data.x, data.edge_index)
            
            embeddings_path = Path(config.training.save_path).parent / "embeddings.pt"
            torch.save(embeddings.cpu(), embeddings_path)
            print(f"Embeddings saved to {embeddings_path}")
        
        # Visualize embeddings if requested
        if config.evaluation.visualize:
            from src.gcl.eval import visualize_embeddings
            
            viz_path = Path(config.training.save_path).parent / "embeddings_visualization.png"
            visualize_embeddings(
                embeddings=embeddings,
                labels=data.y,
                save_path=str(viz_path),
                title=f"{model_config.name} Embeddings on {config.data.name}",
            )
            print(f"Visualization saved to {viz_path}")


if __name__ == "__main__":
    main()
