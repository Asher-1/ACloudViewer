#!/bin/bash
# Configure pip timeout and retry settings for robust package installation
# This script should be run inside Docker containers or development environments

echo "=== Configuring pip timeout and retry settings ==="

# Set environment variables for pip
export PIP_DEFAULT_TIMEOUT=1000
export PIP_RETRIES=5
export PIP_TIMEOUT=1000

# Configure pip global settings
echo "Setting pip global configuration..."
pip config set global.timeout 1000
pip config set global.retries 5
pip config set install.trusted-host "pypi.org files.pythonhosted.org pypi.python.org download.pytorch.org"

# Optional: Set Chinese mirror if needed (uncomment if in China)
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

echo "Current pip configuration:"
pip config list

echo "=== Pip configuration completed ==="

# Function to install PyTorch with robust settings
install_pytorch_robust() {
    local torch_version=${1:-"2.2.2+cu118"}
    local cuda_version=${2:-"cu118"}
    
    echo "Installing PyTorch ${torch_version} with CUDA ${cuda_version}..."
    
    pip install \
        --timeout=1000 \
        --retries=5 \
        --index-url "https://download.pytorch.org/whl/${cuda_version}" \
        --extra-index-url https://pypi.org/simple \
        --no-cache-dir \
        --verbose \
        "torch==${torch_version}"
}

# If script is called with arguments, install PyTorch
if [ $# -gt 0 ]; then
    install_pytorch_robust "$@"
fi
