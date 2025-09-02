#!/bin/bash

# TAM (Token Activation Map) - Conda Environment Setup Script
# This script sets up a complete conda environment for TAM

echo "🚀 Setting up TAM conda environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  - Anaconda: https://www.anaconda.com/products/distribution"
    echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment with Python 3.9
echo "📦 Creating conda environment 'TAM' with Python 3.9..."
conda create -n TAM python=3.9 -y

if [ $? -ne 0 ]; then
    echo "❌ Failed to create conda environment"
    exit 1
fi

echo "✅ Conda environment 'TAM' created successfully"

# Activate environment
echo "🔄 Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate TAM

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment"
    exit 1
fi

echo "✅ Environment activated"

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python packages from requirements.txt..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements"
    echo "💡 Try installing packages individually if some fail"
    exit 1
fi

# Remove conflicting fitz package if it was installed
echo "🔧 Removing conflicting fitz package (if present)..."
pip uninstall fitz -y 2>/dev/null || true

echo "✅ Python packages installed successfully"

# Verify critical imports
echo "🧪 Testing critical imports..."
python -c "
import torch
import cv2
import transformers
import fitz  # from PyMuPDF
import numpy as np
import matplotlib.pyplot as plt
print('✅ All critical packages imported successfully')
"

if [ $? -ne 0 ]; then
    echo "❌ Package import test failed"
    echo "💡 Please check the error messages above"
    exit 1
fi

echo ""
echo "🎉 TAM environment setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "  1. Activate the environment: conda activate TAM"
echo "  2. Run the demo: python demo.py"
echo ""
echo "📚 Environment info:"
echo "  - Environment name: TAM"
echo "  - Python version: $(python --version)"
echo "  - PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  - OpenCV version: $(python -c 'import cv2; print(cv2.__version__)')"
echo ""
echo "🔧 To deactivate: conda deactivate"
echo "🗑️  To remove environment: conda env remove -n TAM"
