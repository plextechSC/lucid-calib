#!/bin/bash
#
# Setup script for lucid-calib pipeline
#
# This script:
# 1. Creates a Python virtual environment
# 2. Installs Python dependencies
# 3. Clones and builds CalibAnything (C++ calibration)
# 4. Clones lucid-sam and sets up its environment with PyTorch
# 5. Downloads SAM model checkpoints
#
# Usage:
#   ./setup.sh
#   source .venv/bin/activate  # Activate the environment after setup
#

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "lucid-calib Pipeline Setup"
echo "=============================================="
echo ""

# Configuration
VENV_DIR=".venv"
EXTERNAL_DIR="external"
CALIBANYTHING_REPO="https://github.com/plextechSC/CalibAnything.git"
LUCID_SAM_REPO="https://github.com/plextechSC/lucid-sam.git"

# Create external directory
mkdir -p "$EXTERNAL_DIR"

# =============================================================================
# Step 1: Create Python virtual environment for lucid-calib
# =============================================================================
echo "[1/6] Creating Python virtual environment..."

if [ -d "$VENV_DIR" ]; then
    echo "  Virtual environment already exists at $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    echo "  Created virtual environment at $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# =============================================================================
# Step 2: Install Python dependencies for lucid-calib
# =============================================================================
echo ""
echo "[2/6] Installing Python dependencies..."

pip install --upgrade pip
pip install -r requirements.txt

echo "  Dependencies installed successfully"

# =============================================================================
# Step 3: Clone CalibAnything repository
# =============================================================================
echo ""
echo "[3/6] Setting up CalibAnything..."

CALIBANYTHING_DIR="$EXTERNAL_DIR/CalibAnything"

if [ -d "$CALIBANYTHING_DIR" ]; then
    echo "  CalibAnything already exists at $CALIBANYTHING_DIR"
else
    echo "  Cloning CalibAnything repository..."
    git clone "$CALIBANYTHING_REPO" "$CALIBANYTHING_DIR"
    echo "  Successfully cloned CalibAnything"
fi

# =============================================================================
# Step 4: Build CalibAnything C++ binary
# =============================================================================
echo ""
echo "[4/6] Building CalibAnything..."

CALIBANYTHING_BIN="$CALIBANYTHING_DIR/bin/run_lidar2camera"

if [ -f "$CALIBANYTHING_BIN" ]; then
    echo "  CalibAnything binary already exists"
else
    # Fix outdated CMakeLists.txt if needed
    CMAKELISTS="$CALIBANYTHING_DIR/CMakeLists.txt"
    if grep -q "cmake_minimum_required(VERSION 2" "$CMAKELISTS" 2>/dev/null; then
        echo "  Fixing outdated CMakeLists.txt..."
        sed -i.bak 's/cmake_minimum_required(VERSION 2[^)]*)/cmake_minimum_required(VERSION 3.10)/' "$CMAKELISTS"
    fi

    # Check if cmake is available
    if ! command -v cmake &> /dev/null; then
        echo "  ERROR: cmake not found!"
        echo "  Please install cmake first:"
        echo "    macOS: brew install cmake"
        echo "    Ubuntu: sudo apt install cmake"
        exit 1
    fi

    # Check for required dependencies
    echo "  Checking dependencies..."
    MISSING_DEPS=""

    # Check for PCL
    if ! pkg-config --exists pcl_common 2>/dev/null && [ ! -d "/usr/local/include/pcl" ] && [ ! -d "/opt/homebrew/include/pcl" ]; then
        MISSING_DEPS="$MISSING_DEPS pcl"
    fi

    # Check for OpenCV
    if ! pkg-config --exists opencv4 2>/dev/null && ! pkg-config --exists opencv 2>/dev/null; then
        MISSING_DEPS="$MISSING_DEPS opencv"
    fi

    if [ -n "$MISSING_DEPS" ]; then
        echo "  WARNING: Some dependencies may be missing:$MISSING_DEPS"
        echo "  Install with:"
        echo "    macOS: brew install pcl opencv eigen boost jsoncpp"
        echo "    Ubuntu: sudo apt install libpcl-dev libopencv-dev libeigen3-dev libboost-all-dev libjsoncpp-dev"
        echo ""
        echo "  Attempting build anyway..."
    fi

    echo "  Building CalibAnything with cmake..."
    cd "$CALIBANYTHING_DIR"
    rm -rf build  # Clean build directory
    mkdir -p build
    cd build

    # Run cmake and make
    if cmake .. && make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4); then
        echo "  Successfully built CalibAnything"
    else
        echo ""
        echo "  ERROR: Failed to build CalibAnything!"
        echo ""
        echo "  Please install the required dependencies:"
        echo "    macOS: brew install pcl opencv eigen boost jsoncpp"
        echo "    Ubuntu: sudo apt install libpcl-dev libopencv-dev libeigen3-dev libboost-all-dev libjsoncpp-dev"
        echo ""
        echo "  Then re-run: ./setup.sh"
        exit 1
    fi

    cd "$SCRIPT_DIR"

    # Verify the binary was created
    if [ ! -f "$CALIBANYTHING_BIN" ]; then
        echo "  ERROR: Build succeeded but binary not found at $CALIBANYTHING_BIN"
        exit 1
    fi
fi

# =============================================================================
# Step 5: Clone and set up lucid-sam
# =============================================================================
echo ""
echo "[5/6] Setting up lucid-sam..."

LUCID_SAM_DIR="$EXTERNAL_DIR/lucid-sam"
LUCID_SAM_VENV="$LUCID_SAM_DIR/.venv"

if [ -d "$LUCID_SAM_DIR" ]; then
    echo "  lucid-sam already exists at $LUCID_SAM_DIR"
else
    echo "  Cloning lucid-sam repository..."
    git clone "$LUCID_SAM_REPO" "$LUCID_SAM_DIR"
    echo "  Successfully cloned lucid-sam"
fi

# Create and set up lucid-sam venv with all dependencies
echo "  Setting up lucid-sam Python environment..."

# Create venv if it doesn't exist
if [ ! -d "$LUCID_SAM_VENV" ]; then
    python3 -m venv "$LUCID_SAM_VENV"
    echo "  Created lucid-sam venv"
fi

# Install dependencies in lucid-sam venv
echo "  Installing lucid-sam dependencies (including PyTorch)..."
"$LUCID_SAM_VENV/bin/pip" install --upgrade pip

# Install PyTorch first (the requirements.txt excludes it for Lambda AI systems)
# Detect platform and install appropriate torch version
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS - install with MPS support
    "$LUCID_SAM_VENV/bin/pip" install torch torchvision
else
    # Linux - try CUDA first, fall back to CPU
    if command -v nvidia-smi &> /dev/null; then
        echo "  CUDA detected, installing PyTorch with CUDA support..."
        "$LUCID_SAM_VENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "  No CUDA detected, installing CPU-only PyTorch..."
        "$LUCID_SAM_VENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# Install rest of requirements
"$LUCID_SAM_VENV/bin/pip" install -r "$LUCID_SAM_DIR/requirements.txt"

echo "  lucid-sam dependencies installed"

# =============================================================================
# Step 6: Download SAM model checkpoints
# =============================================================================
echo ""
echo "[6/6] Downloading SAM model checkpoints..."

CKPT_DIR="$LUCID_SAM_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

# SAM 1 checkpoint URLs
SAM_VIT_H_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_VIT_L_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
SAM_VIT_B_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

download_checkpoint() {
    local url="$1"
    local filename="$(basename "$url")"
    local filepath="$CKPT_DIR/$filename"

    if [ -f "$filepath" ]; then
        echo "  Checkpoint $filename already exists"
    else
        echo "  Downloading $filename..."
        curl -L -o "$filepath" "$url"
        echo "  Downloaded $filename"
    fi
}

# Download SAM checkpoints (Large model is required, others optional)
download_checkpoint "$SAM_VIT_L_URL"

echo "  SAM checkpoints ready"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Directory structure:"
echo "  lucid-calib/"
echo "  ├── .venv/                    # Python virtual environment"
echo "  ├── scripts/                  # Pipeline scripts"
echo "  ├── external/"
echo "  │   ├── CalibAnything/        # Calibration C++ code"
echo "  │   └── lucid-sam/            # SAM mask generation"
echo "  └── requirements.txt"
echo ""

# Check what's ready
echo "Status:"
if [ -f "$CALIBANYTHING_BIN" ]; then
    echo "  [OK] CalibAnything binary built"
else
    echo "  [!!] CalibAnything binary NOT built - calibration step will fail"
fi

if [ -f "$LUCID_SAM_VENV/bin/python" ]; then
    echo "  [OK] lucid-sam environment ready"
else
    echo "  [!!] lucid-sam environment NOT ready"
fi

if [ -f "$CKPT_DIR/sam_vit_l_0b3195.pth" ]; then
    echo "  [OK] SAM checkpoint downloaded"
else
    echo "  [!!] SAM checkpoint NOT downloaded"
fi

echo ""
echo "Next steps:"
echo "  1. Activate the environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Run the pipeline:"
echo "     python scripts/run_pipeline.py -i <input_data> -o <output_dir>"
echo ""
echo "  3. For help:"
echo "     python scripts/run_pipeline.py --help"
echo ""
