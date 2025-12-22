"""Streamlit demo for Graph Contrastive Learning."""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

from src.gcl.models import GraphCLModel, GRACEModel, BGRLModel
from src.gcl.data import load_dataset
from src.gcl.utils import get_device


# Page configuration
st.set_page_config(
    page_title="Graph Contrastive Learning Demo",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    color: #1f77b4;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Graph Contrastive Learning Demo</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_name = st.sidebar.selectbox(
    "Select Model",
    ["GraphCL", "GRACE", "BGRL"],
    help="Choose the contrastive learning model to explore"
)

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ["Cora", "CiteSeer", "PubMed"],
    help="Choose the dataset to analyze"
)

# Load model and data
@st.cache_data
def load_model_and_data(model_name: str, dataset_name: str) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
    """Load model and dataset."""
    try:
        # Load dataset
        dataset = load_dataset(dataset_name.lower(), "data")
        data = dataset.data
        
        # Model configuration based on dataset
        configs = {
            "Cora": {"input_dim": 1433, "hidden_dim": 64, "output_dim": 64},
            "CiteSeer": {"input_dim": 3703, "hidden_dim": 64, "output_dim": 64},
            "PubMed": {"input_dim": 500, "hidden_dim": 64, "output_dim": 64},
        }
        
        config = configs[dataset_name]
        
        # Initialize model
        if model_name == "GraphCL":
            model = GraphCLModel(**config)
        elif model_name == "GRACE":
            model = GRACEModel(**config)
        elif model_name == "BGRL":
            model = BGRLModel(**config, momentum=0.999)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load trained weights if available
        model_path = f"checkpoints/{model_name.lower()}_{dataset_name.lower()}.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            st.sidebar.success(f"Loaded trained model from {model_path}")
        else:
            st.sidebar.warning(f"No trained model found at {model_path}. Using random weights.")
        
        return model, data, {
            "num_nodes": data.num_nodes,
            "num_edges": data.num_edges,
            "num_features": data.num_node_features,
            "num_classes": int(data.y.max().item()) + 1,
        }
    
    except Exception as e:
        st.error(f"Error loading model and data: {e}")
        return None, None, {}


# Load data
model, data, dataset_info = load_model_and_data(model_name, dataset_name)

if model is None:
    st.error("Failed to load model and data. Please check the configuration.")
    st.stop()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Dataset: {dataset_name}")
    
    # Dataset statistics
    col1_1, col1_2, col1_3, col1_4 = st.columns(4)
    with col1_1:
        st.metric("Nodes", dataset_info["num_nodes"])
    with col1_2:
        st.metric("Edges", dataset_info["num_edges"])
    with col1_3:
        st.metric("Features", dataset_info["num_features"])
    with col1_4:
        st.metric("Classes", dataset_info["num_classes"])

with col2:
    st.subheader(f"Model: {model_name}")
    
    # Model statistics
    num_params = sum(p.numel() for p in model.parameters())
    st.metric("Parameters", f"{num_params:,}")

# Generate embeddings
@st.cache_data
def generate_embeddings(model: torch.nn.Module, data: torch.Tensor) -> np.ndarray:
    """Generate node embeddings."""
    model.eval()
    with torch.no_grad():
        if isinstance(model, BGRLModel):
            embeddings = model.encode(data.x, data.edge_index)
        else:
            embeddings = model.encode(data.x, data.edge_index)
    return embeddings.cpu().numpy()


# Generate embeddings
embeddings = generate_embeddings(model, data)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Embeddings", "Node Analysis", "Link Prediction", "Model Comparison"])

with tab1:
    st.subheader("Node Embeddings Visualization")
    
    # t-SNE visualization
    if st.button("Generate t-SNE Visualization"):
        with st.spinner("Computing t-SNE..."):
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Create DataFrame
            df = pd.DataFrame({
                "x": embeddings_2d[:, 0],
                "y": embeddings_2d[:, 1],
                "label": data.y.cpu().numpy(),
                "node_id": range(len(embeddings_2d))
            })
            
            # Create plot
            fig = px.scatter(
                df, x="x", y="y", color="label",
                title=f"{model_name} Embeddings on {dataset_name}",
                labels={"x": "t-SNE 1", "y": "t-SNE 2"},
                hover_data=["node_id"]
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Embedding statistics
    st.subheader("Embedding Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Embedding Dimension", embeddings.shape[1])
    with col2:
        st.metric("Mean L2 Norm", f"{np.linalg.norm(embeddings, axis=1).mean():.3f}")
    with col3:
        st.metric("Std L2 Norm", f"{np.linalg.norm(embeddings, axis=1).std():.3f}")

with tab2:
    st.subheader("Node Classification Analysis")
    
    # Train classifier
    if st.button("Evaluate Node Classification"):
        with st.spinner("Training classifier..."):
            # Get masks
            train_mask = data.train_mask.cpu().numpy()
            val_mask = data.val_mask.cpu().numpy()
            test_mask = data.test_mask.cpu().numpy()
            
            # Train classifier
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(embeddings[train_mask], data.y[train_mask].cpu().numpy())
            
            # Evaluate
            train_pred = clf.predict(embeddings[train_mask])
            val_pred = clf.predict(embeddings[val_mask])
            test_pred = clf.predict(embeddings[test_mask])
            
            train_acc = accuracy_score(data.y[train_mask].cpu().numpy(), train_pred)
            val_acc = accuracy_score(data.y[val_mask].cpu().numpy(), val_pred)
            test_acc = accuracy_score(data.y[test_mask].cpu().numpy(), test_pred)
            
            train_f1 = f1_score(data.y[train_mask].cpu().numpy(), train_pred, average="macro")
            val_f1 = f1_score(data.y[val_mask].cpu().numpy(), val_pred, average="macro")
            test_f1 = f1_score(data.y[test_mask].cpu().numpy(), test_pred, average="macro")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Train Accuracy", f"{train_acc:.3f}")
                st.metric("Train F1", f"{train_f1:.3f}")
            with col2:
                st.metric("Val Accuracy", f"{val_acc:.3f}")
                st.metric("Val F1", f"{val_f1:.3f}")
            with col3:
                st.metric("Test Accuracy", f"{test_acc:.3f}")
                st.metric("Test F1", f"{test_f1:.3f}")
    
    # Node similarity analysis
    st.subheader("Node Similarity Analysis")
    
    node_id = st.number_input("Select Node ID", min_value=0, max_value=len(embeddings)-1, value=0)
    
    if st.button("Analyze Node Similarities"):
        # Compute similarities
        node_embedding = embeddings[node_id]
        similarities = np.dot(embeddings, node_embedding)
        
        # Get top similar nodes
        top_indices = np.argsort(similarities)[::-1][:10]
        
        # Create DataFrame
        df = pd.DataFrame({
            "Node ID": top_indices,
            "Similarity": similarities[top_indices],
            "Label": data.y[top_indices].cpu().numpy(),
            "Same Label": data.y[top_indices].cpu().numpy() == data.y[node_id].cpu().numpy()
        })
        
        st.dataframe(df)
        
        # Visualization
        fig = go.Figure(data=go.Scatter(
            x=range(len(top_indices)),
            y=similarities[top_indices],
            mode='markers',
            marker=dict(
                color=df["Same Label"],
                colorscale="RdYlBu",
                size=10
            ),
            text=df["Node ID"],
            hovertemplate="Node: %{text}<br>Similarity: %{y:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Top 10 Most Similar Nodes to Node {node_id}",
            xaxis_title="Rank",
            yaxis_title="Cosine Similarity"
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Link Prediction Analysis")
    
    if st.button("Evaluate Link Prediction"):
        with st.spinner("Evaluating link prediction..."):
            # Get positive edges
            pos_edges = data.edge_index.cpu().numpy()
            num_pos = pos_edges.shape[1]
            
            # Generate negative edges
            num_nodes = len(embeddings)
            neg_edges = []
            edge_set = set(zip(pos_edges[0], pos_edges[1]))
            
            while len(neg_edges) < num_pos:
                src = np.random.randint(0, num_nodes)
                dst = np.random.randint(0, num_nodes)
                if src != dst and (src, dst) not in edge_set:
                    neg_edges.append([src, dst])
            
            neg_edges = np.array(neg_edges).T
            
            # Compute similarities
            pos_sim = np.sum(embeddings[pos_edges[0]] * embeddings[pos_edges[1]], axis=1)
            neg_sim = np.sum(embeddings[neg_edges[0]] * embeddings[neg_edges[1]], axis=1)
            
            # Create labels and scores
            labels = np.concatenate([np.ones(num_pos), np.zeros(num_pos)])
            scores = np.concatenate([pos_sim, neg_sim])
            
            # Compute metrics
            from sklearn.metrics import roc_auc_score, average_precision_score
            
            roc_auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ROC-AUC", f"{roc_auc:.3f}")
            with col2:
                st.metric("Average Precision", f"{ap:.3f}")
            
            # Visualization
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=pos_sim,
                name="Positive Edges",
                opacity=0.7,
                nbinsx=50
            ))
            
            fig.add_trace(go.Histogram(
                x=neg_sim,
                name="Negative Edges",
                opacity=0.7,
                nbinsx=50
            ))
            
            fig.update_layout(
                title="Similarity Score Distribution",
                xaxis_title="Cosine Similarity",
                yaxis_title="Count",
                barmode="overlay"
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Model Comparison")
    
    # Compare different models
    models_to_compare = st.multiselect(
        "Select Models to Compare",
        ["GraphCL", "GRACE", "BGRL"],
        default=["GraphCL", "GRACE"]
    )
    
    if st.button("Compare Models"):
        comparison_results = {}
        
        for model_name_comp in models_to_compare:
            with st.spinner(f"Evaluating {model_name_comp}..."):
                # Load model
                configs = {
                    "Cora": {"input_dim": 1433, "hidden_dim": 64, "output_dim": 64},
                    "CiteSeer": {"input_dim": 3703, "hidden_dim": 64, "output_dim": 64},
                    "PubMed": {"input_dim": 500, "hidden_dim": 64, "output_dim": 64},
                }
                
                config = configs[dataset_name]
                
                if model_name_comp == "GraphCL":
                    model_comp = GraphCLModel(**config)
                elif model_name_comp == "GRACE":
                    model_comp = GRACEModel(**config)
                elif model_name_comp == "BGRL":
                    model_comp = BGRLModel(**config, momentum=0.999)
                
                # Load weights if available
                model_path = f"checkpoints/{model_name_comp.lower()}_{dataset_name.lower()}.pth"
                if os.path.exists(model_path):
                    model_comp.load_state_dict(torch.load(model_path, map_location="cpu"))
                
                # Generate embeddings
                model_comp.eval()
                with torch.no_grad():
                    if isinstance(model_comp, BGRLModel):
                        embeddings_comp = model_comp.encode(data.x, data.edge_index)
                    else:
                        embeddings_comp = model_comp.encode(data.x, data.edge_index)
                
                embeddings_comp = embeddings_comp.cpu().numpy()
                
                # Evaluate node classification
                train_mask = data.train_mask.cpu().numpy()
                val_mask = data.val_mask.cpu().numpy()
                test_mask = data.test_mask.cpu().numpy()
                
                clf = LogisticRegression(random_state=42, max_iter=1000)
                clf.fit(embeddings_comp[train_mask], data.y[train_mask].cpu().numpy())
                
                test_pred = clf.predict(embeddings_comp[test_mask])
                test_acc = accuracy_score(data.y[test_mask].cpu().numpy(), test_pred)
                test_f1 = f1_score(data.y[test_mask].cpu().numpy(), test_pred, average="macro")
                
                comparison_results[model_name_comp] = {
                    "Test Accuracy": test_acc,
                    "Test F1": test_f1
                }
        
        # Display comparison
        if comparison_results:
            df_comparison = pd.DataFrame(comparison_results).T
            st.dataframe(df_comparison)
            
            # Visualization
            fig = go.Figure()
            
            for model_name_comp in comparison_results:
                fig.add_trace(go.Bar(
                    name=model_name_comp,
                    x=["Test Accuracy", "Test F1"],
                    y=[comparison_results[model_name_comp]["Test Accuracy"],
                       comparison_results[model_name_comp]["Test F1"]]
                ))
            
            fig.update_layout(
                title="Model Comparison",
                xaxis_title="Metrics",
                yaxis_title="Score",
                barmode="group"
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Graph Contrastive Learning Demo** - Explore different contrastive learning methods on graph data")
