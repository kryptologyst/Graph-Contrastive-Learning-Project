"""Data loading and preprocessing utilities."""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected


class BaseGraphDataset(Dataset):
    """Base class for graph datasets with standardized interface."""
    
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
    ):
        """Initialize dataset.
        
        Args:
            root: Root directory for dataset storage.
            name: Dataset name.
            transform: Optional transform to apply to data.
            pre_transform: Optional pre-transform to apply to data.
        """
        self.name = name
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names."""
        return ["nodes.csv", "edges.csv", "splits.json"]
    
    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names."""
        return [f"{self.name}.pt"]
    
    def download(self) -> None:
        """Download dataset if not present."""
        # Override in subclasses for specific datasets
        pass
    
    def process(self) -> None:
        """Process raw data into PyTorch Geometric format."""
        # Override in subclasses
        pass


class CoraDataset(BaseGraphDataset):
    """Cora citation network dataset."""
    
    def __init__(
        self,
        root: str = "data",
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
    ):
        """Initialize Cora dataset."""
        super().__init__(root, "Cora", transform, pre_transform)
        self.data = self.processed_data[0]
    
    def download(self) -> None:
        """Download Cora dataset."""
        dataset = Planetoid(root=self.root, name="Cora")
        self.data = dataset[0]
    
    def process(self) -> None:
        """Process Cora data."""
        dataset = Planetoid(root=self.root, name="Cora")
        data = dataset[0]
        
        # Ensure undirected graph
        data.edge_index = to_undirected(data.edge_index)
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        torch.save(data, self.processed_paths[0])
    
    @property
    def processed_data(self) -> List[Data]:
        """Return processed data."""
        return [torch.load(self.processed_paths[0])]
    
    @property
    def num_features(self) -> int:
        """Return number of node features."""
        return self.data.num_node_features
    
    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return int(self.data.y.max().item()) + 1


class CiteSeerDataset(BaseGraphDataset):
    """CiteSeer citation network dataset."""
    
    def __init__(
        self,
        root: str = "data",
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
    ):
        """Initialize CiteSeer dataset."""
        super().__init__(root, "CiteSeer", transform, pre_transform)
        self.data = self.processed_data[0]
    
    def download(self) -> None:
        """Download CiteSeer dataset."""
        dataset = Planetoid(root=self.root, name="CiteSeer")
        self.data = dataset[0]
    
    def process(self) -> None:
        """Process CiteSeer data."""
        dataset = Planetoid(root=self.root, name="CiteSeer")
        data = dataset[0]
        
        # Ensure undirected graph
        data.edge_index = to_undirected(data.edge_index)
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        torch.save(data, self.processed_paths[0])
    
    @property
    def processed_data(self) -> List[Data]:
        """Return processed data."""
        return [torch.load(self.processed_paths[0])]
    
    @property
    def num_features(self) -> int:
        """Return number of node features."""
        return self.data.num_node_features
    
    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return int(self.data.y.max().item()) + 1


class PubMedDataset(BaseGraphDataset):
    """PubMed citation network dataset."""
    
    def __init__(
        self,
        root: str = "data",
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
    ):
        """Initialize PubMed dataset."""
        super().__init__(root, "PubMed", transform, pre_transform)
        self.data = self.processed_data[0]
    
    def download(self) -> None:
        """Download PubMed dataset."""
        dataset = Planetoid(root=self.root, name="PubMed")
        self.data = dataset[0]
    
    def process(self) -> None:
        """Process PubMed data."""
        dataset = Planetoid(root=self.root, name="PubMed")
        data = dataset[0]
        
        # Ensure undirected graph
        data.edge_index = to_undirected(data.edge_index)
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        torch.save(data, self.processed_paths[0])
    
    @property
    def processed_data(self) -> List[Data]:
        """Return processed data."""
        return [torch.load(self.processed_paths[0])]
    
    @property
    def num_features(self) -> int:
        """Return number of node features."""
        return self.data.num_node_features
    
    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return int(self.data.y.max().item()) + 1


def load_dataset(name: str, root: str = "data") -> BaseGraphDataset:
    """Load dataset by name.
    
    Args:
        name: Dataset name.
        root: Root directory for data storage.
        
    Returns:
        BaseGraphDataset: Loaded dataset.
        
    Raises:
        ValueError: If dataset name is not supported.
    """
    datasets = {
        "cora": CoraDataset,
        "citeseer": CiteSeerDataset,
        "pubmed": PubMedDataset,
    }
    
    if name.lower() not in datasets:
        raise ValueError(f"Dataset {name} not supported. Available: {list(datasets.keys())}")
    
    return datasets[name.lower()](root=root)
