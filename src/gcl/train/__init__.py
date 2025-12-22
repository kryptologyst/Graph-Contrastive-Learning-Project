"""Training utilities for contrastive learning models."""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from .models import GraphCLModel, GRACEModel, BGRLModel
from .utils import get_device, count_parameters


def contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Compute contrastive loss between two views.
    
    Args:
        z1: Embeddings from first view.
        z2: Embeddings from second view.
        temperature: Temperature parameter.
        
    Returns:
        torch.Tensor: Contrastive loss.
    """
    batch_size = z1.size(0)
    device = z1.device
    
    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    
    # Positive pairs are on the diagonal
    labels = torch.arange(batch_size, device=device)
    
    # Symmetric loss
    loss_12 = F.cross_entropy(sim_matrix, labels)
    loss_21 = F.cross_entropy(sim_matrix.T, labels)
    
    return (loss_12 + loss_21) / 2


def bgrl_loss(
    z_online: torch.Tensor,
    z_target: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Compute BGRL loss.
    
    Args:
        z_online: Online embeddings.
        z_target: Target embeddings.
        temperature: Temperature parameter.
        
    Returns:
        torch.Tensor: BGRL loss.
    """
    batch_size = z_online.size(0)
    device = z_online.device
    
    # Normalize embeddings
    z_online = F.normalize(z_online, dim=1)
    z_target = F.normalize(z_target, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(z_online, z_target.T) / temperature
    
    # Positive pairs are on the diagonal
    labels = torch.arange(batch_size, device=device)
    
    return F.cross_entropy(sim_matrix, labels)


def train_contrastive_model(
    model: Union[GraphCLModel, GRACEModel, BGRLModel],
    data: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    temperature: float = 0.5,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    log_interval: int = 10,
    **kwargs,
) -> Dict[str, List[float]]:
    """Train a contrastive learning model.
    
    Args:
        model: Contrastive learning model.
        data: Graph data.
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay.
        temperature: Temperature for contrastive loss.
        device: Device to use.
        save_path: Path to save best model.
        log_interval: Logging interval.
        **kwargs: Additional training parameters.
        
    Returns:
        Dict[str, List[float]]: Training history.
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    data = data.to(device)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Setup scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    history = {"loss": [], "lr": []}
    
    # Best model tracking
    best_loss = float("inf")
    
    print(f"Training {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Generate two augmented views
        if isinstance(model, GraphCLModel):
            view1 = model.augment(data, **kwargs.get("augmentation", {}))
            view2 = model.augment(data, **kwargs.get("augmentation", {}))
            
            z1 = model(view1.x, view1.edge_index)
            z2 = model(view2.x, view2.edge_index)
            
            loss = contrastive_loss(z1, z2, temperature)
            
        elif isinstance(model, GRACEModel):
            view1 = model.augment(data, **kwargs.get("augmentation", {}))
            view2 = model.augment(data, **kwargs.get("augmentation", {}))
            
            z1 = model(view1.x, view1.edge_index)
            z2 = model(view2.x, view2.edge_index)
            
            loss = contrastive_loss(z1, z2, temperature)
            
        elif isinstance(model, BGRLModel):
            view1 = model.augment(data, **kwargs.get("augmentation", {}))
            view2 = model.augment(data, **kwargs.get("augmentation", {}))
            
            z1_online, z1_target = model(view1.x, view1.edge_index)
            z2_online, z2_target = model(view2.x, view2.edge_index)
            
            loss1 = bgrl_loss(z1_online, z2_target, temperature)
            loss2 = bgrl_loss(z2_online, z1_target, temperature)
            loss = (loss1 + loss2) / 2
            
            # Update target encoder
            model._update_target_encoder()
        
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update history
        history["loss"].append(loss.item())
        history["lr"].append(scheduler.get_last_lr()[0])
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        
        # Logging
        if epoch % log_interval == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
    
    print(f"Training completed. Best loss: {best_loss:.4f}")
    return history


def evaluate_contrastive_model(
    model: Union[GraphCLModel, GRACEModel, BGRLModel],
    data: torch.Tensor,
    task: str = "node_classification",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate contrastive learning model on downstream tasks.
    
    Args:
        model: Trained contrastive learning model.
        data: Graph data.
        task: Downstream task ("node_classification", "link_prediction").
        device: Device to use.
        
    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        if isinstance(model, BGRLModel):
            embeddings = model.encode(data.x, data.edge_index)
        else:
            embeddings = model.encode(data.x, data.edge_index)
    
    if task == "node_classification":
        return evaluate_node_classification(embeddings, data, device)
    elif task == "link_prediction":
        return evaluate_link_prediction(embeddings, data, device)
    else:
        raise ValueError(f"Unknown task: {task}")


def evaluate_node_classification(
    embeddings: torch.Tensor,
    data: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate embeddings on node classification task.
    
    Args:
        embeddings: Node embeddings.
        data: Graph data with labels.
        device: Device to use.
        
    Returns:
        Dict[str, float]: Classification metrics.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    
    # Get train/val/test masks
    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    
    # Get embeddings and labels
    X = embeddings.cpu().numpy()
    y = data.y.cpu().numpy()
    
    # Train classifier
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X[train_mask], y[train_mask])
    
    # Evaluate
    train_pred = clf.predict(X[train_mask])
    val_pred = clf.predict(X[val_mask])
    test_pred = clf.predict(X[test_mask])
    
    train_acc = accuracy_score(y[train_mask], train_pred)
    val_acc = accuracy_score(y[val_mask], val_pred)
    test_acc = accuracy_score(y[test_mask], test_pred)
    
    train_f1 = f1_score(y[train_mask], train_pred, average="macro")
    val_f1 = f1_score(y[val_mask], val_pred, average="macro")
    test_f1 = f1_score(y[test_mask], test_pred, average="macro")
    
    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "test_f1": test_f1,
    }


def evaluate_link_prediction(
    embeddings: torch.Tensor,
    data: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate embeddings on link prediction task.
    
    Args:
        embeddings: Node embeddings.
        data: Graph data.
        device: Device to use.
        
    Returns:
        Dict[str, float]: Link prediction metrics.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    # Get edge indices
    edge_index = data.edge_index.cpu().numpy()
    
    # Generate negative edges
    num_nodes = embeddings.size(0)
    num_edges = edge_index.shape[1]
    
    # Sample negative edges
    neg_edges = []
    while len(neg_edges) < num_edges:
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        if src != dst and (src, dst) not in zip(edge_index[0], edge_index[1]):
            neg_edges.append((src, dst))
    
    neg_edges = torch.tensor(neg_edges).T
    
    # Compute similarities
    pos_sim = torch.sum(embeddings[edge_index[0]] * embeddings[edge_index[1]], dim=1)
    neg_sim = torch.sum(embeddings[neg_edges[0]] * embeddings[neg_edges[1]], dim=1)
    
    # Create labels and scores
    labels = torch.cat([torch.ones(num_edges), torch.zeros(num_edges)])
    scores = torch.cat([pos_sim, neg_sim])
    
    # Compute metrics
    roc_auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
    ap = average_precision_score(labels.cpu().numpy(), scores.cpu().numpy())
    
    return {
        "roc_auc": roc_auc,
        "average_precision": ap,
    }
