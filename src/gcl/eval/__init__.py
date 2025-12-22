"""Evaluation utilities for contrastive learning models."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from .models import GraphCLModel, GRACEModel, BGRLModel


def compute_alignment_uniformity(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.5,
) -> Tuple[float, float]:
    """Compute alignment and uniformity metrics for contrastive learning.
    
    Args:
        embeddings: Node embeddings.
        labels: Node labels.
        temperature: Temperature parameter.
        
    Returns:
        Tuple[float, float]: Alignment and uniformity scores.
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)
    
    # Compute pairwise similarities
    sim_matrix = torch.matmul(embeddings, embeddings.T)
    
    # Alignment: average similarity of positive pairs (same class)
    alignment_scores = []
    for i in range(len(labels)):
        same_class_mask = labels == labels[i]
        same_class_mask[i] = False  # Exclude self
        if same_class_mask.any():
            alignment_scores.append(sim_matrix[i, same_class_mask].mean().item())
    
    alignment = np.mean(alignment_scores) if alignment_scores else 0.0
    
    # Uniformity: negative log of average pairwise similarity
    # Exclude diagonal (self-similarity)
    mask = torch.eye(len(embeddings), dtype=torch.bool)
    off_diagonal_sim = sim_matrix[~mask]
    uniformity = -torch.log(torch.mean(torch.exp(off_diagonal_sim / temperature))).item()
    
    return alignment, uniformity


def evaluate_node_classification(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate embeddings on node classification task.
    
    Args:
        embeddings: Node embeddings.
        labels: Node labels.
        train_mask: Training mask.
        val_mask: Validation mask.
        test_mask: Test mask.
        random_state: Random state for reproducibility.
        
    Returns:
        Dict[str, float]: Classification metrics.
    """
    # Convert to numpy
    X = embeddings.cpu().numpy()
    y = labels.cpu().numpy()
    train_mask = train_mask.cpu().numpy()
    val_mask = val_mask.cpu().numpy()
    test_mask = test_mask.cpu().numpy()
    
    # Train classifier
    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(X[train_mask], y[train_mask])
    
    # Evaluate on all splits
    splits = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }
    
    results = {}
    for split_name, mask in splits.items():
        if mask.any():
            pred = clf.predict(X[mask])
            true = y[mask]
            
            results[f"{split_name}_accuracy"] = accuracy_score(true, pred)
            results[f"{split_name}_f1_micro"] = f1_score(true, pred, average="micro")
            results[f"{split_name}_f1_macro"] = f1_score(true, pred, average="macro")
            
            # AUROC for binary classification
            if len(np.unique(true)) == 2:
                proba = clf.predict_proba(X[mask])[:, 1]
                results[f"{split_name}_auroc"] = roc_auc_score(true, proba)
    
    return results


def evaluate_link_prediction(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    num_neg_samples: int = 1000,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate embeddings on link prediction task.
    
    Args:
        embeddings: Node embeddings.
        edge_index: Edge indices.
        num_neg_samples: Number of negative samples.
        random_state: Random state for reproducibility.
        
    Returns:
        Dict[str, float]: Link prediction metrics.
    """
    np.random.seed(random_state)
    
    # Get positive edges
    pos_edges = edge_index.cpu().numpy()
    num_nodes = embeddings.size(0)
    
    # Generate negative edges
    neg_edges = []
    edge_set = set(zip(pos_edges[0], pos_edges[1]))
    
    while len(neg_edges) < num_neg_samples:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst and (src, dst) not in edge_set:
            neg_edges.append([src, dst])
    
    neg_edges = np.array(neg_edges).T
    
    # Compute similarities
    pos_sim = torch.sum(
        embeddings[pos_edges[0]] * embeddings[pos_edges[1]], dim=1
    ).cpu().numpy()
    neg_sim = torch.sum(
        embeddings[neg_edges[0]] * embeddings[neg_edges[1]], dim=1
    ).cpu().numpy()
    
    # Create labels and scores
    labels = np.concatenate([np.ones(len(pos_sim)), np.zeros(len(neg_sim))])
    scores = np.concatenate([pos_sim, neg_sim])
    
    # Compute metrics
    roc_auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    
    return {
        "roc_auc": roc_auc,
        "average_precision": ap,
    }


def evaluate_clustering(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate embeddings on clustering task.
    
    Args:
        embeddings: Node embeddings.
        labels: True cluster labels.
        n_clusters: Number of clusters (if None, use number of unique labels).
        random_state: Random state for reproducibility.
        
    Returns:
        Dict[str, float]: Clustering metrics.
    """
    if n_clusters is None:
        n_clusters = len(torch.unique(labels))
    
    # Convert to numpy
    X = embeddings.cpu().numpy()
    y_true = labels.cpu().numpy()
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    y_pred = kmeans.fit_predict(X)
    
    # Compute metrics
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    return {
        "nmi": nmi,
        "ari": ari,
    }


def visualize_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Node Embeddings",
) -> None:
    """Visualize node embeddings using t-SNE.
    
    Args:
        embeddings: Node embeddings.
        labels: Node labels.
        save_path: Path to save the plot.
        title: Plot title.
    """
    # Convert to numpy
    X = embeddings.cpu().numpy()
    y = labels.cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def evaluate_model(
    model: torch.nn.Module,
    data: torch.Tensor,
    tasks: List[str] = ["node_classification", "link_prediction"],
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """Comprehensive evaluation of a contrastive learning model.
    
    Args:
        model: Trained model.
        data: Graph data.
        tasks: List of tasks to evaluate.
        device: Device to use.
        
    Returns:
        Dict[str, Dict[str, float]]: Evaluation results for each task.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        if isinstance(model, BGRLModel):
            embeddings = model.encode(data.x, data.edge_index)
        else:
            embeddings = model.encode(data.x, data.edge_index)
    
    results = {}
    
    # Compute alignment and uniformity
    alignment, uniformity = compute_alignment_uniformity(embeddings, data.y)
    results["contrastive_metrics"] = {
        "alignment": alignment,
        "uniformity": uniformity,
    }
    
    # Evaluate on specified tasks
    if "node_classification" in tasks:
        results["node_classification"] = evaluate_node_classification(
            embeddings, data.y, data.train_mask, data.val_mask, data.test_mask
        )
    
    if "link_prediction" in tasks:
        results["link_prediction"] = evaluate_link_prediction(
            embeddings, data.edge_index
        )
    
    if "clustering" in tasks:
        results["clustering"] = evaluate_clustering(embeddings, data.y)
    
    return results


def create_evaluation_report(
    results: Dict[str, Dict[str, float]],
    model_name: str,
    dataset_name: str,
) -> str:
    """Create a formatted evaluation report.
    
    Args:
        results: Evaluation results.
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        
    Returns:
        str: Formatted report.
    """
    report = f"Evaluation Report\n"
    report += f"Model: {model_name}\n"
    report += f"Dataset: {dataset_name}\n"
    report += "=" * 50 + "\n\n"
    
    for task, metrics in results.items():
        report += f"{task.replace('_', ' ').title()}:\n"
        for metric, value in metrics.items():
            report += f"  {metric}: {value:.4f}\n"
        report += "\n"
    
    return report
