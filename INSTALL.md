# TAM Installation Guide

Complete installation guide for Token Activation Map (TAM) on various platforms.

## Quick Setup (Recommended)

### Option 1: Automated Setup Script

```bash
# Clone the repository
git clone https://github.com/adityanagachandra/TAM.git
cd TAM

# Run the automated setup script
./setup_conda_env.sh
```

### Option 2: Manual Setup

```bash
# Create conda environment
conda create -n TAM python=3.9 -y

# Activate environment
conda activate TAM

# Install dependencies
pip install -r requirements.txt

# Remove conflicting package (if needed)
pip uninstall fitz -y
```

## Detailed Platform-Specific Instructions

### Prerequisites

1. **Conda/Anaconda**: Install from [anaconda.com](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. **Git**: For cloning the repository
3. **CUDA-compatible GPU**: Recommended for model inference (8GB+ VRAM)

### Step-by-Step Installation

#### 1. Environment Creation

```bash
# Create a new conda environment with Python 3.9
conda create -n TAM python=3.9 -y

# Activate the environment
conda activate TAM

# Verify Python version
python --version  # Should show Python 3.9.x
```

#### 2. Install Core Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

#### 3. Handle Known Issues

```bash
# Remove conflicting fitz package (PyMuPDF provides the correct one)
pip uninstall fitz -y

# Verify critical imports work
python -c "import torch, cv2, transformers, fitz; print('✅ All imports successful')"
```

#### 4. Optional: LaTeX for Text Visualization

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install texlive-xetex
```

**macOS:**
```bash
# Using Homebrew
brew install --cask mactex
# OR for smaller installation
brew install --cask basictex
```

**Windows:**
- Install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

## Package Details

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `transformers` | 4.52.1 | Hugging Face transformers library |
| `torch` | Latest | PyTorch deep learning framework |
| `opencv-python` | Latest | Computer vision operations |
| `accelerate` | Latest | Model loading and device mapping |
| `pymupdf` | Latest | PDF processing (provides `fitz`) |

### Full Dependencies List

```
# Core ML and Transformers
transformers==4.52.1
accelerate
torch
torchvision
safetensors

# Computer Vision
opencv-python
pillow

# PDF Processing
pymupdf

# Scientific Computing
numpy
scipy
matplotlib

# NLP
nltk
rouge

# Utilities
pathlib
pyyaml
tqdm
requests
huggingface-hub
```

## Verification

### Test Installation

```bash
# Activate environment
conda activate TAM

# Test critical imports
python -c "
import torch
import cv2
import transformers
import fitz
import numpy as np
print('✅ Installation successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Run Demo

```bash
# Test with demo (will download ~4GB model on first run)
python demo.py
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'cv2'**
   ```bash
   pip install opencv-python
   ```

2. **Accelerate package missing**
   ```bash
   pip install accelerate
   ```

3. **Fitz module conflicts**
   ```bash
   pip uninstall fitz -y
   # PyMuPDF provides the correct fitz module
   ```

4. **CUDA out of memory**
   - Reduce batch size
   - Use smaller model variant
   - Ensure GPU has 8GB+ VRAM

5. **Model download interrupted**
   - Ensure stable internet connection
   - Download will resume automatically on restart

### Environment Management

```bash
# List all conda environments
conda env list

# Activate TAM environment
conda activate TAM

# Deactivate environment
conda deactivate

# Remove environment (if needed)
conda env remove -n TAM

# Export environment (for sharing)
conda env export > environment.yml

# Create from exported environment
conda env create -f environment.yml
```

## System Requirements

### Minimum Requirements
- **CPU**: Multi-core processor
- **RAM**: 8GB+
- **Storage**: 15GB free space
- **OS**: Linux, macOS, or Windows 10/11

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: 8GB+ VRAM (NVIDIA with CUDA support)
- **Storage**: 25GB+ free space (for models and datasets)
- **OS**: Linux (Ubuntu 20.04+) or macOS

### Cloud Platform Requirements

**Lambda Labs:**
- Instance type: A10 or better
- Storage: 50GB+ SSD
- Pre-installed CUDA drivers

**AWS/Google Cloud/Azure:**
- GPU instances with 8GB+ VRAM
- CUDA 11.8+ support
- Ubuntu 20.04+ or similar Linux distribution

## Next Steps

After successful installation:

1. **Run Demo**: `python demo.py`
2. **Download Datasets**: For evaluation (see main README)
3. **Customize**: Adapt for your specific models and use cases

For more information, see the main [README.md](README.md) file.
