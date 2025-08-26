#!/bin/bash

# Simple setup script with conda installation

# Detect OS and set miniconda installer
if uname -s | grep -iq Darwin; then
    INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
else
    INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
fi

# Set paths
MINICONDA_PATH="$HOME/miniconda3"
DOWNLOAD_URL="https://repo.anaconda.com/miniconda/$INSTALLER"

# Check if conda exists
if command -v conda >/dev/null 2>&1; then
    echo "Conda already installed"
else
    echo "Installing conda..."
    # Download and install miniconda
    curl -LO "$DOWNLOAD_URL"
    bash "$INSTALLER" -b -p "$MINICONDA_PATH"
    rm "$INSTALLER"
    
    # Initialize conda
    "$MINICONDA_PATH/bin/conda" init bash
    "$MINICONDA_PATH/bin/conda" config --set auto_activate_base false
    
    echo "Conda installed. Please restart your shell or run: source ~/.bashrc"
fi

# Create/update environment
ENV_NAME="conda-$USER"
echo "Setting up environment: $ENV_NAME"

if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment exists, updating..."
else
    echo "Creating new environment..."
    conda create --name "$ENV_NAME" python=3.11 numpy pandas matplotlib -y
fi

# Install/update requirements
conda run -n "$ENV_NAME" pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    conda run -n "$ENV_NAME" pip install -r requirements.txt
fi

echo "Setup complete! Activate with: conda activate $ENV_NAME"

