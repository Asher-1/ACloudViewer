#!/bin/bash
# Test CloudViewer wheel in runtime container
# Usage: test_wheel_runtime.sh <wheel_dir> <BUILD_CUDA_MODULE> [python_version] [docker_image]
#   wheel_dir: Directory containing wheel files
#   BUILD_CUDA_MODULE: ON for CUDA wheel, OFF for CPU wheel
#   python_version: Python version (default: 3.12)
#   docker_image: Docker image to use (optional, will use current image if not provided)
#
# This script can be run in two modes:
# 1. Inside Docker container: test_wheel_runtime.sh /opt/wheels ON 3.12
# 2. From host: docker run --gpus all -v ... test_wheel_runtime.sh /opt/wheels ON 3.12

set -euo pipefail

WHEEL_DIR="${1:-}"
BUILD_CUDA_MODULE="${2:-ON}"
PYTHON_VERSION="${3:-3.12}"
DOCKER_IMAGE="${4:-}"

# Function to print usage
usage() {
    echo "Usage: $0 <wheel_dir> <BUILD_CUDA_MODULE> [python_version] [docker_image]"
    echo ""
    echo "Arguments:"
    echo "  wheel_dir       : Directory containing wheel files"
    echo "  BUILD_CUDA_MODULE: ON for CUDA wheel, OFF for CPU wheel"
    echo "  python_version  : Python version (default: 3.12)"
    echo "  docker_image    : Docker image name (optional, for host mode)"
    echo ""
    echo "Examples:"
    echo "  # Test CUDA wheel inside container:"
    echo "  $0 /opt/wheels ON 3.12"
    echo ""
    echo "  # Test CPU wheel inside container:"
    echo "  $0 /opt/wheels OFF 3.12"
    echo ""
    echo "  # Test from host (requires docker run --gpus all):"
    echo "  docker run --gpus all -v \$(pwd):/opt/wheels -v /path/to/ACloudViewer:/root/ACloudViewer \\"
    echo "    --rm cloudviewer-ci:wheel \\"
    echo "    bash /root/ACloudViewer/docker/test_wheel_runtime.sh /opt/wheels ON 3.12"
    exit 1
}

# Check if running from host (need to run docker command)
if [ -n "$DOCKER_IMAGE" ] && [ ! -f "/.dockerenv" ]; then
    # Running from host, execute in Docker container
    CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
    
    # Determine GPU flag based on BUILD_CUDA_MODULE
    GPU_FLAG=""
    if [ "$BUILD_CUDA_MODULE" = "ON" ]; then
        GPU_FLAG="--gpus all"
    fi
    
    echo "=========================================="
    echo "Running wheel test in Docker container"
    echo "Wheel directory: $WHEEL_DIR"
    echo "BUILD_CUDA_MODULE: $BUILD_CUDA_MODULE"
    echo "Python version: $PYTHON_VERSION"
    echo "Docker image: $DOCKER_IMAGE"
    echo "=========================================="
    
    docker run $GPU_FLAG \
        --network host \
        -v "${CLOUDVIEWER_SOURCE_ROOT}:/root/ACloudViewer" \
        -v "${WHEEL_DIR}:/opt/wheels" \
        --rm "$DOCKER_IMAGE" \
        bash -c "bash /root/ACloudViewer/docker/test_wheel_runtime.sh /opt/wheels $BUILD_CUDA_MODULE $PYTHON_VERSION"
    exit $?
fi

# Running inside container
if [ -z "$WHEEL_DIR" ]; then
    echo "Error: wheel directory is required"
    usage
fi

if [ ! -d "$WHEEL_DIR" ]; then
    echo "Error: wheel directory not found: $WHEEL_DIR"
    exit 1
fi

# Convert Python version to wheel tag format (e.g., 3.12 -> cp312)
PYTHON_TAG=$(echo "$PYTHON_VERSION" | tr -d '.' | sed 's/^/cp/')

# Determine wheel file pattern based on BUILD_CUDA_MODULE and Python version
if [ "$BUILD_CUDA_MODULE" = "ON" ]; then
    WHEEL_PATTERN="cloudviewer-*-${PYTHON_TAG}-*.whl"
    WHEEL_TYPE="CUDA"
    # Check GPU availability for CUDA builds
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found. CUDA wheel may not work properly."
        echo "Make sure to run with --gpus all if testing CUDA wheel."
    fi
else
    WHEEL_PATTERN="cloudviewer_cpu-*-${PYTHON_TAG}-*.whl"
    WHEEL_TYPE="CPU"
fi

# Find wheel file
WHEEL_FILE=$(find "$WHEEL_DIR" -maxdepth 1 -name "$WHEEL_PATTERN" | head -n 1)

if [ -z "$WHEEL_FILE" ] || [ ! -f "$WHEEL_FILE" ]; then
    echo "Error: Could not find $WHEEL_TYPE wheel file matching pattern: $WHEEL_PATTERN"
    echo "Searched in: $WHEEL_DIR"
    echo "Available files:"
    ls -la "$WHEEL_DIR"/*.whl 2>/dev/null || echo "  (no .whl files found)"
    exit 1
fi

echo "=========================================="
echo "Testing $WHEEL_TYPE wheel in runtime container"
echo "Wheel: $WHEEL_FILE"
echo "Python version: $PYTHON_VERSION"
echo "BUILD_CUDA_MODULE: $BUILD_CUDA_MODULE"
echo "=========================================="

# Setup Python environment
export PYENV_ROOT=/root/.pyenv
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH"

# Check if pyenv is installed, if not, install it
if [ ! -d "$PYENV_ROOT" ]; then
    echo "Installing pyenv..."
    curl -fsSL https://pyenv.run | bash || true
    export PATH="$PYENV_ROOT/bin:$PATH"
fi

# Install Python version if not already installed
if [ ! -d "$PYENV_ROOT/versions/$PYTHON_VERSION" ]; then
    echo "Installing Python $PYTHON_VERSION..."
    pyenv install -s "$PYTHON_VERSION" || true
    pyenv global "$PYTHON_VERSION"
    pyenv rehash
fi

# Set up Python path
if [ -d "$PYENV_ROOT/versions/$PYTHON_VERSION" ]; then
    export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH"
    # Create symlink if needed
    if [ ! -L "$PYENV_ROOT/versions/$PYTHON_VERSION" ]; then
        ln -sf "$PYENV_ROOT/versions/${PYTHON_VERSION}"* "$PYENV_ROOT/versions/${PYTHON_VERSION}" 2>/dev/null || true
    fi
fi

python --version || {
    echo "Error: Python $PYTHON_VERSION not available"
    exit 1
}

# Check GPU availability (only for CUDA builds)
if [ "$BUILD_CUDA_MODULE" = "ON" ]; then
    echo ""
    echo "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi || echo "Warning: nvidia-smi failed, but continuing..."
    else
        echo "Warning: nvidia-smi not found. GPU may not be available."
    fi
fi

# Get source root (assuming script is in docker/ directory)
CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"

# Set environment variables that might be needed by ci_utils.sh
export CLOUDVIEWER_SOURCE_ROOT="${CLOUDVIEWER_SOURCE_ROOT}"
export CLOUDVIEWER_ML_ROOT="${CLOUDVIEWER_ML_ROOT:-${CLOUDVIEWER_SOURCE_ROOT}/../CloudViewer-ML}"
export BUILD_CUDA_MODULE="${BUILD_CUDA_MODULE}"

# Source ci_utils.sh to get test_wheel function
if [ -f "$CLOUDVIEWER_SOURCE_ROOT/util/ci_utils.sh" ]; then
    source "$CLOUDVIEWER_SOURCE_ROOT/util/ci_utils.sh"
else
    echo "Error: ci_utils.sh not found at $CLOUDVIEWER_SOURCE_ROOT/util/ci_utils.sh"
    exit 1
fi

# Change to a temporary directory for testing
TEST_DIR=$(mktemp -d)
ORIGINAL_DIR=$(pwd)

# Cleanup function to safely remove test directory
cleanup_test_dir() {
    local exit_code=$?
    # Return to original directory first
    cd "$ORIGINAL_DIR" 2>/dev/null || cd / 2>/dev/null || true
    # Only remove if TEST_DIR is set, exists, is a directory, and is under /tmp
    if [ -n "${TEST_DIR:-}" ] && [ -d "${TEST_DIR}" ] && [[ "${TEST_DIR}" == /tmp/* ]]; then
        # Double-check it's really a temp directory
        if [[ "${TEST_DIR}" =~ ^/tmp/tmp\.[a-zA-Z0-9]+$ ]] || [[ "${TEST_DIR}" =~ ^/tmp/tmp-[a-zA-Z0-9]+$ ]]; then
            rm -rf "${TEST_DIR}" 2>/dev/null || true
        fi
    fi
    # Exit with the original exit code
    exit ${exit_code}
}

# Set trap to ensure cleanup happens even on error
trap cleanup_test_dir EXIT INT TERM

cd "$TEST_DIR" || {
    echo "Error: Failed to change to test directory: $TEST_DIR"
    exit 1
}
echo "Testing in directory: $TEST_DIR"

# Run the test
echo ""
echo "Starting $WHEEL_TYPE wheel test..."
test_wheel "$WHEEL_FILE"
TEST_EXIT_CODE=$?

# Explicit cleanup before exit (trap will also handle it)
cd "$ORIGINAL_DIR" 2>/dev/null || cd / 2>/dev/null || true
if [ -n "$TEST_DIR" ] && [ -d "$TEST_DIR" ] && [[ "$TEST_DIR" == /tmp/* ]]; then
    # Double-check it's really a temp directory
    if [[ "$TEST_DIR" =~ ^/tmp/tmp\.[a-zA-Z0-9]+$ ]] || [[ "$TEST_DIR" =~ ^/tmp/tmp-[a-zA-Z0-9]+$ ]]; then
        rm -rf "$TEST_DIR" 2>/dev/null || true
    fi
fi

# Clear trap before normal exit
trap - EXIT INT TERM

# Exit with test result
if [ $TEST_EXIT_CODE -ne 0 ]; then
    exit $TEST_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "$WHEEL_TYPE wheel test completed successfully!"
echo "=========================================="
