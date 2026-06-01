#!/usr/bin/env bash
# CHAIN-MAC + SMAC server setup script
# Tested on Ubuntu 20.04/22.04 with CUDA 11.8+
# Usage: bash scripts/setup_server.sh [--conda-env chain-mac]

set -euo pipefail

CONDA_ENV="${1:-chain-mac}"
SC2_VERSION="4.10"
SC2_URL="https://blz.nosdn.127.net/2/SC2/${SC2_VERSION}/SC2.${SC2_VERSION}.zip"
SC2_MIRROR="https://github.com/Blizzard/s2client-proto/releases/download/v${SC2_VERSION}/SC2.${SC2_VERSION}.zip"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== CHAIN-MAC + SMAC Server Setup ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Conda env:   ${CONDA_ENV}"
echo ""

# ── 1. System dependencies ──────────────────────────────────────────
echo ">>> Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    curl wget unzip bzip2 \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    libgomp1 \
    || echo "    (some packages may be unavailable, continuing)"

# For headless rendering (Xvfb for pysc2)
sudo apt-get install -y -qq xvfb || true
echo ""

# ── 2. Conda environment ────────────────────────────────────────────
echo ">>> Setting up Conda environment '${CONDA_ENV}'..."
if ! command -v conda &>/dev/null; then
    echo "    Conda not found. Install Miniconda first:"
    echo "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "    bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3"
    echo "    ~/miniconda3/bin/conda init"
    exit 1
fi

# Create or update env
if conda info --envs | grep -q "^${CONDA_ENV} "; then
    echo "    Env '${CONDA_ENV}' already exists, updating..."
    conda install -y -n "${CONDA_ENV}" python=3.10 -c conda-forge -qq
else
    conda create -y -n "${CONDA_ENV}" python=3.10 -c conda-forge -qq
fi

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
echo ""

# ── 3. Python dependencies ──────────────────────────────────────────
echo ">>> Installing Python dependencies..."
pip install --upgrade pip -q

# Core ML
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q

# SC2 + SMAC
pip install pysc2 -q
pip install git+https://github.com/oxwhirl/smac.git -q

# Support
pip install \
    numpy scikit-learn \
    matplotlib tensorboard \
    -q

echo ""

# ── 4. StarCraft II ─────────────────────────────────────────────────
SC2_DIR="${HOME}/StarCraftII"
MAP_DIR="${SC2_DIR}/Maps/SMAC_Maps"

if [ ! -d "${SC2_DIR}/Versions" ]; then
    echo ">>> Downloading StarCraft II (${SC2_VERSION})..."
    echo "    This may take several minutes (~4GB)."

    mkdir -p "${SC2_DIR}"
    TMP_ZIP="/tmp/SC2.${SC2_VERSION}.zip"

    # Try Blizzard CDN first, then GitHub mirror
    if curl -fSL "${SC2_URL}" -o "${TMP_ZIP}" 2>/dev/null; then
        echo "    Downloaded from official CDN."
    elif curl -fSL "${SC2_MIRROR}" -o "${TMP_ZIP}" 2>/dev/null; then
        echo "    Downloaded from GitHub mirror."
    else
        echo "    WARNING: Could not download SC2 automatically."
        echo "    Manual download: https://github.com/Blizzard/s2client-proto#downloads"
        echo "    Place it at: ${SC2_DIR}"
    fi

    if [ -f "${TMP_ZIP}" ]; then
        unzip -q "${TMP_ZIP}" -d "${HOME}/"
        rm "${TMP_ZIP}"
        echo "    StarCraft II extracted to ${SC2_DIR}"
    fi
else
    echo ">>> StarCraft II already installed at ${SC2_DIR}"
fi
echo ""

# ── 5. SMAC maps ────────────────────────────────────────────────────
if [ ! -d "${MAP_DIR}" ]; then
    echo ">>> Downloading SMAC maps..."
    mkdir -p "${MAP_DIR}"
    TMP_MAPS="/tmp/smac_maps.zip"

    curl -fSL "https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip" \
         -o "${TMP_MAPS}" 2>/dev/null && {
        unzip -q "${TMP_MAPS}" -d "${MAP_DIR}/.."
        rm "${TMP_MAPS}"
        echo "    SMAC maps extracted to ${MAP_DIR}"
    } || {
        echo "    WARNING: Could not download SMAC maps automatically."
        echo "    Manual: https://github.com/oxwhirl/smac/releases"
    }
else
    echo ">>> SMAC maps already installed at ${MAP_DIR}"
fi
echo ""

# ── 6. Environment variables ───────────────────────────────────────
echo ">>> Setting up environment variables..."
RC_FILE="${HOME}/.bashrc"

# SC2PATH
if ! grep -q "SC2PATH" "${RC_FILE}" 2>/dev/null; then
    echo "export SC2PATH=${SC2_DIR}" >> "${RC_FILE}"
    echo "    Added SC2PATH=${SC2_DIR} to ~/.bashrc"
else
    echo "    SC2PATH already set in ~/.bashrc"
fi

# Xvfb for headless rendering
if ! grep -q "DISPLAY" "${RC_FILE}" 2>/dev/null; then
    echo "export DISPLAY=:0" >> "${RC_FILE}"
    echo "    Added DISPLAY=:0 to ~/.bashrc"
fi

# Source now
export SC2PATH="${SC2_DIR}"
export DISPLAY=":0"
echo ""

# ── 7. Verify ──────────────────────────────────────────────────────
echo ">>> Verifying installation..."
python -c "
import sys
print(f'  Python {sys.version}')

import numpy as np
print(f'  numpy {np.__version__}')

import torch
print(f'  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')

import pysc2
print(f'  pysc2 {pysc2.__version__ if hasattr(pysc2, \"__version__\") else \"ok\"}')

try:
    from smac.env import StarCraft2Env
    print('  smac ok')
except Exception as e:
    print(f'  smac import failed: {e}')
" 2>&1

echo ""
echo "=== Setup complete ==="
echo ""
echo "To train:"
echo "  conda activate ${CONDA_ENV}"
echo "  cd ${PROJECT_DIR}"
echo "  python scripts/train_smac.py --map 3m --steps 2000000"
echo ""
echo "Available maps: 3m, 8m, 2s3z, 3s_vs_5z, 5m_vs_6m, MMM2"
