"""Graph Contrastive Learning Package.

A modern implementation of Graph Contrastive Learning methods including GraphCL,
GRACE, and BGRL using PyTorch Geometric.
"""

__version__ = "0.1.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"

from .models import GraphCLModel, GRACEModel, BGRLModel
from .data import CoraDataset, CiteSeerDataset, PubMedDataset
from .train import train_contrastive_model
from .eval import evaluate_model

__all__ = [
    "GraphCLModel",
    "GRACEModel", 
    "BGRLModel",
    "CoraDataset",
    "CiteSeerDataset",
    "PubMedDataset",
    "train_contrastive_model",
    "evaluate_model",
]
