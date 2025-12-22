#!/usr/bin/env python3
"""Setup script for Graph Contrastive Learning project."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Graph Contrastive Learning Project Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Create necessary directories
    print("\nCreating project directories...")
    directories = [
        "data/raw",
        "data/processed", 
        "checkpoints",
        "results",
        "assets",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    
    # Install base dependencies
    if not run_command("pip install -e .", "Installing base dependencies"):
        print("Failed to install base dependencies")
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        print("Warning: Failed to install development dependencies")
    
    # Install pre-commit hooks
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        print("Warning: Failed to install pre-commit hooks")
    
    # Run tests
    print("\nRunning tests...")
    if not run_command("python -m pytest tests/ -v", "Running unit tests"):
        print("Warning: Some tests failed")
    
    # Create a simple demo
    print("\nRunning quick start demo...")
    if not run_command("python quick_start.py", "Running quick start demo"):
        print("Warning: Quick start demo failed")
    
    print("\n" + "=" * 50)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Run 'python quick_start.py' for a quick demo")
    print("2. Run 'python scripts/demo.py --dataset synthetic --epochs 100 --eval' for full demo")
    print("3. Run 'streamlit run demo/app.py' for interactive web demo")
    print("4. Run 'python scripts/train.py --config configs/graphcl_cora.yaml --eval' to train on Cora")
    print("\nFor development:")
    print("- Run 'pre-commit run --all-files' to check code quality")
    print("- Run 'python -m pytest tests/ -v' to run tests")
    print("- Run 'black src/ tests/ scripts/' to format code")
    print("- Run 'ruff check src/ tests/ scripts/' to lint code")


if __name__ == "__main__":
    main()
