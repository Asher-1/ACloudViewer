#!/bin/bash
# Robust PyTorch installation script with timeout and retry handling
# Usage: ./install_pytorch_robust.sh [torch_version] [cuda_version]

set -e

# Default values
TORCH_VERSION=${1:-"2.2.2+cu118"}
CUDA_VERSION=${2:-"cu118"}
MAX_RETRIES=3
TIMEOUT=1200  # 20 minutes
RETRY_DELAY=30  # 30 seconds between retries

echo "=== Robust PyTorch Installation Script ==="
echo "PyTorch version: ${TORCH_VERSION}"
echo "CUDA version: ${CUDA_VERSION}"
echo "Max retries: ${MAX_RETRIES}"
echo "Timeout: ${TIMEOUT} seconds"

# Configure pip for better timeout handling
echo "Configuring pip settings..."
pip config set global.timeout ${TIMEOUT}
pip config set global.retries ${MAX_RETRIES}
pip config set install.trusted-host "pypi.org files.pythonhosted.org pypi.python.org download.pytorch.org"

# Set PyTorch index URL
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_VERSION}"
echo "PyTorch index URL: ${PYTORCH_INDEX_URL}"

# Function to install with retries
install_with_retry() {
    local package=$1
    local attempt=1
    
    while [ $attempt -le $MAX_RETRIES ]; do
        echo "Attempt $attempt/$MAX_RETRIES: Installing $package..."
        
        if timeout ${TIMEOUT} pip install \
            --timeout=${TIMEOUT} \
            --retries=${MAX_RETRIES} \
            --index-url ${PYTORCH_INDEX_URL} \
            --extra-index-url https://pypi.org/simple \
            --no-cache-dir \
            --verbose \
            "$package"; then
            echo "Successfully installed $package"
            return 0
        else
            echo "Failed to install $package (attempt $attempt/$MAX_RETRIES)"
            if [ $attempt -lt $MAX_RETRIES ]; then
                echo "Waiting ${RETRY_DELAY} seconds before retry..."
                sleep ${RETRY_DELAY}
            fi
            ((attempt++))
        fi
    done
    
    echo "ERROR: Failed to install $package after $MAX_RETRIES attempts"
    return 1
}

# Function to check network connectivity
check_network() {
    echo "Checking network connectivity..."
    if ! curl -s --connect-timeout 10 https://pypi.org > /dev/null; then
        echo "WARNING: Network connectivity issues detected"
        echo "Checking proxy settings..."
        env | grep -i proxy || echo "No proxy environment variables found"
    else
        echo "Network connectivity OK"
    fi
}

# Function to clear pip cache
clear_pip_cache() {
    echo "Clearing pip cache..."
    pip cache purge || true
    rm -rf ~/.cache/pip/* || true
}

# Main installation process
main() {
    echo "Starting robust PyTorch installation..."
    
    # Check network
    check_network
    
    # Clear cache to avoid corrupted files
    clear_pip_cache
    
    # Install PyTorch with retries
    if install_with_retry "torch==${TORCH_VERSION}"; then
        echo "✅ PyTorch installation completed successfully!"
        
        # Verify installation
        echo "Verifying PyTorch installation..."
        python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
        
    else
        echo "❌ PyTorch installation failed after all retries"
        echo ""
        echo "=== Troubleshooting suggestions ==="
        echo "1. Check your internet connection"
        echo "2. Verify proxy settings if using a proxy"
        echo "3. Try using a different PyTorch index URL"
        echo "4. Consider using conda instead of pip"
        echo "5. Try installing without CUDA support first: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        exit 1
    fi
}

# Alternative installation methods
install_alternative() {
    echo ""
    echo "=== Alternative Installation Methods ==="
    echo ""
    
    echo "Method 1: Install CPU version first, then CUDA"
    echo "pip install torch --index-url https://download.pytorch.org/whl/cpu"
    echo "pip install torch==${TORCH_VERSION} --index-url ${PYTORCH_INDEX_URL}"
    echo ""
    
    echo "Method 2: Use conda instead of pip"
    echo "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    echo ""
    
    echo "Method 3: Download wheels manually"
    echo "wget https://download.pytorch.org/whl/${CUDA_VERSION}/torch-${TORCH_VERSION}-cp39-cp39-linux_x86_64.whl"
    echo "pip install torch-${TORCH_VERSION}-cp39-cp39-linux_x86_64.whl"
    echo ""
    
    echo "Method 4: Use Chinese mirror (if in China)"
    echo "pip install torch==${TORCH_VERSION} -i https://pypi.tuna.tsinghua.edu.cn/simple/"
}

# Show help
show_help() {
    echo "Usage: $0 [torch_version] [cuda_version]"
    echo ""
    echo "Examples:"
    echo "  $0                          # Install default version (2.2.2+cu118)"
    echo "  $0 2.1.0+cu118 cu118       # Install specific version"
    echo "  $0 --help                  # Show this help"
    echo "  $0 --alternatives          # Show alternative installation methods"
}

# Parse arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --alternatives|-a)
        install_alternative
        exit 0
        ;;
    *)
        main
        ;;
esac
