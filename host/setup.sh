#!/bin/bash
# Pi3X + SAM2 Host Environment Setup Script
# This script sets up the venv environment for GPU inference on macOS (MPS)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

echo "======================================"
echo "Pi3X + SAM2 Host Environment Setup"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "[1/5] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo ""
    echo "[1/5] Virtual environment already exists, skipping..."
fi

# Activate venv
echo ""
echo "[2/5] Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "Python path: $(which python)"

# Upgrade pip
pip install --upgrade pip

# Clone external repositories
echo ""
echo "[3/5] Setting up external repositories..."
mkdir -p "$PROJECT_ROOT/repos"

# SAM2
if [ ! -d "$PROJECT_ROOT/repos/sam2" ]; then
    echo "Cloning SAM2..."
    git clone https://github.com/facebookresearch/sam2.git "$PROJECT_ROOT/repos/sam2"
else
    echo "SAM2 already cloned, pulling latest..."
    cd "$PROJECT_ROOT/repos/sam2" && git pull
fi

# Pi3
if [ ! -d "$PROJECT_ROOT/repos/pi3" ]; then
    echo "Cloning Pi3..."
    git clone https://github.com/yyfz/Pi3.git "$PROJECT_ROOT/repos/pi3"
else
    echo "Pi3 already cloned, pulling latest..."
    cd "$PROJECT_ROOT/repos/pi3" && git pull
fi

cd "$PROJECT_ROOT"

# Install dependencies
echo ""
echo "[4/5] Installing dependencies..."

# Host requirements
pip install -r "$PROJECT_ROOT/host/requirements.txt"

# SAM2 (editable install)
echo "Installing SAM2..."
cd "$PROJECT_ROOT/repos/sam2"
pip install -e .

# Pi3 requirements
echo "Installing Pi3 requirements..."
if [ -f "$PROJECT_ROOT/repos/pi3/requirements.txt" ]; then
    pip install -r "$PROJECT_ROOT/repos/pi3/requirements.txt"
fi

cd "$PROJECT_ROOT"

# Download model checkpoints
echo ""
echo "[5/5] Setting up model checkpoints..."
mkdir -p "$PROJECT_ROOT/models/sam2" "$PROJECT_ROOT/models/pi3x"

# SAM2 checkpoints
SAM2_CKPT_DIR="$PROJECT_ROOT/repos/sam2/checkpoints"
if [ ! -f "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt" ]; then
    echo "Downloading SAM2 checkpoints..."
    cd "$SAM2_CKPT_DIR"
    if [ -f "download_ckpts.sh" ]; then
        bash download_ckpts.sh
    else
        # Manual download if script not available
        echo "Downloading SAM2.1 Hiera Large checkpoint..."
        curl -L -o sam2.1_hiera_large.pt \
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    fi
else
    echo "SAM2 checkpoint already exists."
fi

cd "$PROJECT_ROOT"

# Pi3X checkpoint (will be downloaded automatically by HuggingFace on first run)
echo "Pi3X checkpoint will be downloaded automatically on first inference."

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To start the inference server:"
echo "  python host/inference_server.py"
echo ""
