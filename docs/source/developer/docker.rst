Docker Development
==================

ACloudViewer provides comprehensive Docker support for development, testing, and deployment.

Overview
--------

Docker images are available for:

- **Documentation Building**: Generate docs in isolated environment
- **CI/CD Testing**: Automated builds and tests
- **Wheel Building**: Create Python wheels for distribution
- **Development**: Consistent development environment
- **GPU Support**: CUDA-enabled builds and testing

Prerequisites
-------------

Docker Installation
^^^^^^^^^^^^^^^^^^^

Install Docker Engine:

.. code-block:: bash

   # Ubuntu
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Verify installation
   docker run --rm hello-world

**Post-installation** (Linux):

.. code-block:: bash

   # Add user to docker group (run without sudo)
   sudo usermod -aG docker $USER
   newgrp docker
   
   # Verify
   docker run --rm hello-world  # Should work without sudo

NVIDIA Docker (GPU Support)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Required for CUDA builds and GPU testing:

.. code-block:: bash

   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
       sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   
   # Verify GPU access
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

ARM64 Support (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^

For building ARM64 images on x86-64 hosts:

.. code-block:: bash

   # Install QEMU
   sudo apt-get install -y qemu binfmt-support qemu-user-static
   
   # Register ARM64 interpreter
   docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
   
   # Verify ARM64 support
   docker run --rm arm64v8/ubuntu:22.04 uname -m
   # Should output: aarch64

Available Docker Images
-----------------------

Documentation Build Image
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Dockerfile**: ``docker/Dockerfile.docs``

**Purpose**: Build comprehensive documentation (Sphinx + Doxygen) in an isolated environment.

**Features**:

- Ubuntu 22.04 base
- Python 3.11 with all doc dependencies
- Qt5 support (for GUI documentation)
- CloudViewer-ML integration
- Doxygen, Pandoc, Sphinx
- Automatic Python module detection and build

**Build**:

.. code-block:: bash

   # Development build (with git hash)
   docker build \
       --build-arg DEVELOPER_BUILD=ON \
       -t acloudviewer-ci:docs \
       -f docker/Dockerfile.docs .
   
   # Release build (with version number)
   docker build \
       --build-arg DEVELOPER_BUILD=OFF \
       -t acloudviewer-ci:docs \
       -f docker/Dockerfile.docs .

**Extract Documentation**:

.. code-block:: bash

   # Extract documentation tarball
   docker run -v $(pwd):/opt/mount --rm acloudviewer-ci:docs \
       bash -c "cp /root/ACloudViewer/acloudviewer-*-docs.tar.gz /opt/mount/"
   
   # Unpack and preview
   mkdir -p docs-preview
   tar -xzf acloudviewer-*-docs.tar.gz -C ./docs-preview/
   cd docs-preview && python3 -m http.server 8080
   # Open http://localhost:8080

**What It Builds**:

1. Python module (if not present)
2. C++ API documentation (Doxygen)
3. Python API documentation (Sphinx autodoc)
4. Jupyter notebook tutorials (nbsphinx)
5. Unified HTML documentation

CI/CD Build Image
^^^^^^^^^^^^^^^^^^

**Dockerfile**: ``docker/Dockerfile.ci``

**Purpose**: Automated CI/CD builds and testing.

**Variants**:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Variant
     - Base Image
     - Use Case
   * - CPU Static
     - ``ubuntu:22.04``
     - Static libraries, minimal dependencies
   * - CPU Shared
     - ``ubuntu:22.04``
     - Shared libraries with ML support
   * - CUDA Shared
     - ``nvidia/cuda:12.1.0-devel-ubuntu22.04``
     - GPU-accelerated builds
   * - Qt6 CPU
     - ``ubuntu:24.04``
     - Qt6-based GUI builds
   * - Qt6 CUDA
     - ``nvidia/cuda:12.1.0-devel-ubuntu24.04``
     - Qt6 + GPU builds

**Build Examples**:

.. code-block:: bash

   # CPU static build
   docker build \
       -f docker/Dockerfile.ci \
       --build-arg BASE_IMAGE=ubuntu:22.04 \
       --build-arg DEVELOPER_BUILD=ON \
       --build-arg BUILD_SHARED_LIBS=OFF \
       -t acloudviewer-ci:cpu-static .
   
   # CUDA shared build with ML
   docker build \
       -f docker/Dockerfile.ci \
       --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-devel-ubuntu22.04 \
       --build-arg DEVELOPER_BUILD=ON \
       --build-arg BUILD_SHARED_LIBS=ON \
       --build-arg BUILD_CUDA_MODULE=ON \
       --build-arg BUILD_PYTORCH_OPS=ON \
       -t acloudviewer-ci:cuda-ml .

**Using Build Scripts**:

.. code-block:: bash

   cd docker
   
   # See all available configurations
   ./docker_build.sh --help
   
   # Build CPU variant
   ./docker_build.sh cpu-static
   
   # Build CUDA variant with ML
   ./docker_build.sh cuda-ml-shared-jammy
   
   # Build Qt6 variant
   ./docker_build_qt6.sh qt6-cpu-static

Qt6 Build Image
^^^^^^^^^^^^^^^

**Dockerfile**: ``docker/Dockerfile.ci.qt6``

**Purpose**: Build with Qt6 support (Ubuntu 24.04+).

**Features**:

- Qt6 base libraries
- Qt6 WebSockets
- Qt6 networking
- Modern CMake (3.28+)

**Build**:

.. code-block:: bash

   docker build \
       -f docker/Dockerfile.ci.qt6 \
       --build-arg BASE_IMAGE=ubuntu:24.04 \
       --build-arg DEVELOPER_BUILD=ON \
       -t acloudviewer-ci:qt6 .

Wheel Building Image
^^^^^^^^^^^^^^^^^^^^

**Dockerfiles**: 

- ``docker/Dockerfile.wheel`` (Qt5, Ubuntu 20.04/22.04)
- ``docker/Dockerfile.wheel.qt6`` (Qt6, Ubuntu 24.04)

**Purpose**: Build manylinux-compatible Python wheels.

**Python Versions**: 3.10, 3.11, 3.12, 3.13

**Build**:

.. code-block:: bash

   # Build wheel with CUDA support
   docker build \
       -f docker/Dockerfile.wheel \
       --build-arg BASE_IMAGE=ubuntu:22.04 \
       --build-arg PYTHON_VERSION=3.11 \
       --build-arg BUILD_CUDA_MODULE=ON \
       --build-arg BUILD_PYTORCH_OPS=ON \
       -t acloudviewer-wheel:cuda-py311 .
   
   # Extract wheel
   docker run -v $(pwd):/opt/mount --rm acloudviewer-wheel:cuda-py311 \
       bash -c "cp /root/ACloudViewer/build/lib/python_package/pip_package/*.whl /opt/mount/"

**Using Build Scripts**:

.. code-block:: bash

   cd docker
   
   # Build wheel for Python 3.11
   ./build_cloudviewer_whl.sh 3.11
   
   # Build all Python versions
   ./build-all.sh
   
   # Build release wheels
   ./build-release.sh

Using Docker Images
-------------------

Interactive Development
^^^^^^^^^^^^^^^^^^^^^^^

Launch an interactive shell for development:

.. code-block:: bash

   # CPU environment
   docker run -it --rm \
       -v $(pwd):/workspace \
       -w /workspace \
       acloudviewer-ci:cpu-static \
       bash
   
   # Inside container
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   ctest --output-on-failure

**With GPU support**:

.. code-block:: bash

   docker run -it --rm --gpus all \
       -v $(pwd):/workspace \
       -w /workspace \
       acloudviewer-ci:cuda-ml \
       bash
   
   # Build with CUDA
   mkdir build && cd build
   cmake .. \
       -DCMAKE_BUILD_TYPE=Release \
       -DBUILD_CUDA_MODULE=ON \
       -DBUILD_PYTORCH_OPS=ON
   make -j$(nproc)

Running Tests
^^^^^^^^^^^^^

Run the complete test suite in Docker:

**C++ Unit Tests**:

.. code-block:: bash

   docker run --rm \
       -v $(pwd):/workspace \
       -w /workspace/build \
       acloudviewer-ci:cpu-static \
       bash -c "ctest --output-on-failure"

**Python Tests**:

.. code-block:: bash

   docker run --rm \
       -v $(pwd):/workspace \
       -w /workspace \
       acloudviewer-ci:cpu-static \
       bash -c "source util/ci_utils.sh && run_python_tests path/to/wheel.whl"

**GPU Tests**:

.. code-block:: bash

   docker run --rm --gpus all \
       -v $(pwd):/workspace \
       -w /workspace/build \
       acloudviewer-ci:cuda-ml \
       bash -c "ctest --output-on-failure -R GPU"

**Using Test Script**:

.. code-block:: bash

   cd docker
   
   # Test CPU build
   ./docker_test.sh cpu-static
   
   # Test CUDA build
   ./docker_test.sh cuda-ml-shared-jammy
   
   # Test Qt6 build
   ./docker_test.sh qt6-cpu-static

Building Wheels
^^^^^^^^^^^^^^^

Build Python wheels inside Docker:

.. code-block:: bash

   # Using ci_utils.sh
   docker run --rm \
       -v $(pwd):/workspace \
       -w /workspace \
       acloudviewer-wheel:cuda-py311 \
       bash -c "source util/ci_utils.sh && build_pip_package"
   
   # Extract wheel
   docker cp $(docker ps -lq):/root/ACloudViewer/dist ./

**Test wheel installation**:

.. code-block:: bash

   docker run --rm \
       -v $(pwd):/workspace \
       -w /workspace \
       python:3.11-slim \
       bash -c "pip install dist/*.whl && python -c 'import cloudViewer; print(cloudViewer.__version__)'"

Build Arguments
---------------

Common Build Arguments
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Argument
     - Default
     - Description
   * - ``BASE_IMAGE``
     - ``ubuntu:22.04``
     - Base Docker image
   * - ``DEVELOPER_BUILD``
     - ``ON``
     - Enable dev features (git hash in version)
   * - ``PYTHON_VERSION``
     - ``3.11``
     - Python version (3.10-3.13)
   * - ``CMAKE_VERSION``
     - ``3.28.1``
     - CMake version
   * - ``BUILD_SHARED_LIBS``
     - ``OFF``
     - Build shared libraries
   * - ``BUILD_CUDA_MODULE``
     - ``OFF``
     - Enable CUDA support
   * - ``BUILD_PYTORCH_OPS``
     - ``OFF``
     - Build PyTorch operators
   * - ``BUILD_TENSORFLOW_OPS``
     - ``OFF``
     - Build TensorFlow operators
   * - ``BUILD_GUI``
     - ``ON``
     - Build GUI components
   * - ``CCACHE_TAR_NAME``
     - (none)
     - ccache archive for faster rebuilds

**Example with all arguments**:

.. code-block:: bash

   docker build \
       --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-devel-ubuntu22.04 \
       --build-arg DEVELOPER_BUILD=OFF \
       --build-arg PYTHON_VERSION=3.11 \
       --build-arg CMAKE_VERSION=3.28.1 \
       --build-arg BUILD_SHARED_LIBS=ON \
       --build-arg BUILD_CUDA_MODULE=ON \
       --build-arg BUILD_PYTORCH_OPS=ON \
       --build-arg BUILD_TENSORFLOW_OPS=OFF \
       --build-arg BUILD_GUI=ON \
       -t acloudviewer:custom \
       -f docker/Dockerfile.ci .

Dockerfile Structure
--------------------

Typical Multi-Stage Dockerfile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ACloudViewer Dockerfiles use multi-stage builds for efficiency:

.. code-block:: dockerfile

   # Stage 1: Base dependencies
   FROM ubuntu:22.04 AS base
   RUN apt-get update && apt-get install -y \
       build-essential cmake git
   
   # Stage 2: Build stage
   FROM base AS builder
   COPY . /workspace
   WORKDIR /workspace
   RUN mkdir build && cd build && \
       cmake .. -DCMAKE_BUILD_TYPE=Release && \
       make -j$(nproc)
   
   # Stage 3: Runtime stage (smaller)
   FROM ubuntu:22.04 AS runtime
   COPY --from=builder /workspace/build/lib /usr/local/lib
   COPY --from=builder /workspace/build/bin /usr/local/bin
   RUN ldconfig

**Benefits**:

- Smaller final image (no build tools)
- Faster subsequent builds (cached layers)
- Reproducible builds

Layer Optimization
^^^^^^^^^^^^^^^^^^

Order Dockerfile commands from least to most frequently changing:

.. code-block:: dockerfile

   # 1. System packages (rarely change)
   RUN apt-get update && apt-get install -y \
       build-essential cmake git
   
   # 2. Python dependencies (occasional changes)
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   # 3. Third-party dependencies (infrequent changes)
   COPY 3rdparty/ /workspace/3rdparty/
   RUN cd /workspace/3rdparty && ./build_deps.sh
   
   # 4. Source code (frequent changes)
   COPY . /workspace

Caching Strategies
------------------

BuildKit Cache
^^^^^^^^^^^^^^

Use Docker BuildKit for advanced caching:

.. code-block:: bash

   # Enable BuildKit
   export DOCKER_BUILDKIT=1
   
   # Build with cache
   docker build \
       --cache-from acloudviewer-ci:cpu-static \
       -t acloudviewer-ci:cpu-static \
       -f docker/Dockerfile.ci .
   
   # Use GitHub Actions cache
   docker build \
       --cache-from type=gha \
       --cache-to type=gha,mode=max \
       -t acloudviewer-ci:cpu-static \
       -f docker/Dockerfile.ci .

ccache Integration
^^^^^^^^^^^^^^^^^^

Leverage ccache for C++ compilation:

.. code-block:: bash

   # Create ccache volume
   docker volume create ccache-vol
   
   # Run with ccache volume
   docker run --rm \
       -v ccache-vol:/root/.cache/ccache \
       -v $(pwd):/workspace \
       acloudviewer-ci:cpu-static \
       bash -c "ccache -M 5G && cd /workspace/build && make -j$(nproc)"
   
   # Check ccache statistics
   docker run --rm \
       -v ccache-vol:/root/.cache/ccache \
       acloudviewer-ci:cpu-static \
       ccache -s

**Benefits**:

- First build: ~45 min
- Cached build: ~8 min (5.6x speedup)

Docker Compose
--------------

For multi-container development setups:

**docker-compose.yml**:

.. code-block:: yaml

   version: '3.8'
   
   services:
     acloudviewer-dev:
       build:
         context: .
         dockerfile: docker/Dockerfile.ci
         args:
           BASE_IMAGE: ubuntu:22.04
           DEVELOPER_BUILD: 'ON'
           BUILD_SHARED_LIBS: 'OFF'
       volumes:
         - .:/workspace
         - ccache-vol:/root/.cache/ccache
       working_dir: /workspace
       command: bash
       stdin_open: true
       tty: true
     
     acloudviewer-cuda:
       build:
         context: .
         dockerfile: docker/Dockerfile.ci
         args:
           BASE_IMAGE: nvidia/cuda:12.1.0-devel-ubuntu22.04
           DEVELOPER_BUILD: 'ON'
           BUILD_CUDA_MODULE: 'ON'
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: all
                 capabilities: [gpu]
       volumes:
         - .:/workspace
       working_dir: /workspace
       command: bash
       stdin_open: true
       tty: true
   
   volumes:
     ccache-vol:

**Usage**:

.. code-block:: bash

   # Start services
   docker-compose up -d
   
   # Enter CPU container
   docker-compose exec acloudviewer-dev bash
   
   # Enter CUDA container
   docker-compose exec acloudviewer-cuda bash
   
   # Stop services
   docker-compose down

Testing with Docker
-------------------

Complete Test Suite
^^^^^^^^^^^^^^^^^^^

The ``docker_test.sh`` script runs comprehensive tests:

.. code-block:: bash

   cd docker
   
   # List available test configurations
   ./docker_test.sh
   
   # Run CPU tests
   ./docker_test.sh cpu-static
   
   # Run CUDA tests
   ./docker_test.sh cuda-ml-shared-jammy
   
   # Run Qt6 tests
   ./docker_test.sh qt6-cpu-static

**Test Workflow**:

1. Build C++ unit tests
2. Run C++ tests with gtest
3. Build Python package
4. Install Python package
5. Run Python tests with pytest
6. Test command-line tools
7. Test CMake integration
8. Test uninstall process

Custom Test Runs
^^^^^^^^^^^^^^^^

Run specific test suites:

.. code-block:: bash

   # C++ tests only
   docker run --rm acloudviewer-ci:cpu-static \
       bash -c "cd build && ./bin/tests"
   
   # Python tests only
   docker run --rm acloudviewer-ci:cpu-static \
       bash -c "cd python/test && pytest -v"
   
   # Specific test file
   docker run --rm acloudviewer-ci:cpu-static \
       bash -c "cd python/test && pytest test_pointcloud.py -v"
   
   # With GPU
   docker run --rm --gpus all acloudviewer-ci:cuda-ml \
       bash -c "cd build && ./bin/tests --gtest_filter=*CUDA*"

Troubleshooting
---------------

Permission Issues
^^^^^^^^^^^^^^^^^

Run Docker as current user to avoid permission issues:

.. code-block:: bash

   docker run --rm \
       --user $(id -u):$(id -g) \
       -v $(pwd):/workspace \
       acloudviewer-ci:cpu-static

**Fix existing permission issues**:

.. code-block:: bash

   # Fix ownership of build artifacts
   docker run --rm \
       -v $(pwd):/workspace \
       ubuntu:22.04 \
       chown -R $(id -u):$(id -g) /workspace/build

Out of Memory
^^^^^^^^^^^^^

Limit parallel builds and memory usage:

.. code-block:: bash

   # Limit to 4 parallel jobs
   docker run --rm \
       -m 8g \
       --cpus 4 \
       -v $(pwd):/workspace \
       acloudviewer-ci:cpu-static \
       bash -c "make -j4"
   
   # Monitor memory usage
   docker stats

Out of Disk Space
^^^^^^^^^^^^^^^^^

Clean up Docker resources:

.. code-block:: bash

   # Remove unused containers
   docker container prune -f
   
   # Remove unused images
   docker image prune -a -f
   
   # Remove build cache
   docker builder prune -a -f
   
   # Remove volumes
   docker volume prune -f
   
   # Clean everything
   docker system prune -a --volumes -f

**Check disk usage**:

.. code-block:: bash

   # Docker disk usage
   docker system df
   
   # Detailed breakdown
   docker system df -v

Network Issues
^^^^^^^^^^^^^^

Use host network for better connectivity:

.. code-block:: bash

   docker build --network=host \
       -t acloudviewer-ci:cpu-static \
       -f docker/Dockerfile.ci .

**Proxy configuration**:

.. code-block:: bash

   docker build \
       --build-arg http_proxy=$http_proxy \
       --build-arg https_proxy=$https_proxy \
       --build-arg no_proxy=$no_proxy \
       -t acloudviewer-ci:cpu-static \
       -f docker/Dockerfile.ci .

GPU Not Available
^^^^^^^^^^^^^^^^^

**Verify NVIDIA Docker**:

.. code-block:: bash

   # Check NVIDIA driver
   nvidia-smi
   
   # Check NVIDIA Docker runtime
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   
   # Restart Docker daemon
   sudo systemctl restart docker

**Common fixes**:

.. code-block:: bash

   # Update NVIDIA Container Toolkit
   sudo apt-get update
   sudo apt-get install --reinstall nvidia-container-toolkit
   sudo systemctl restart docker
   
   # Configure runtime
   sudo tee /etc/docker/daemon.json <<EOF
   {
       "runtimes": {
           "nvidia": {
               "path": "nvidia-container-runtime",
               "runtimeArgs": []
           }
       }
   }
   EOF
   sudo systemctl restart docker

Best Practices
--------------

1. **Use Specific Base Images**
   
   Pin versions for reproducibility:
   
   .. code-block:: dockerfile
   
      FROM ubuntu:22.04  # Good
      # FROM ubuntu:latest  # Bad (unpredictable)

2. **Minimize Layers**
   
   Combine related RUN commands:
   
   .. code-block:: dockerfile
   
      # Good
      RUN apt-get update && \
          apt-get install -y pkg1 pkg2 && \
          rm -rf /var/lib/apt/lists/*
      
      # Bad
      RUN apt-get update
      RUN apt-get install -y pkg1
      RUN apt-get install -y pkg2

3. **Clean Up in Same Layer**
   
   Remove temporary files before layer commit:
   
   .. code-block:: dockerfile
   
      RUN apt-get update && \
          apt-get install -y build-essential && \
          # ... build steps ... && \
          apt-get purge -y build-essential && \
          apt-get autoremove -y && \
          rm -rf /var/lib/apt/lists/*

4. **Use .dockerignore**
   
   Exclude unnecessary files from build context:
   
   .. code-block:: text
   
      # .dockerignore
      **/.git
      **/build
      **/build_*
      **/__pycache__
      **/*.pyc
      **/.pytest_cache
      **/docs/_out
      **/node_modules
      **/.vscode
      **/.idea

5. **Multi-Stage Builds**
   
   Separate build and runtime stages:
   
   .. code-block:: dockerfile
   
      FROM ubuntu:22.04 AS builder
      # ... build steps ...
      
      FROM ubuntu:22.04 AS runtime
      COPY --from=builder /build/output /usr/local

6. **Health Checks**
   
   Add health checks for long-running containers:
   
   .. code-block:: dockerfile
   
      HEALTHCHECK --interval=30s --timeout=3s \
        CMD python -c "import cloudViewer" || exit 1

7. **Non-Root User**
   
   Run as non-root for security:
   
   .. code-block:: dockerfile
   
      RUN useradd -m -s /bin/bash acloudviewer
      USER acloudviewer

Resources
---------

- **Docker Documentation**: https://docs.docker.com/
- **Best Practices**: https://docs.docker.com/develop/dev-best-practices/
- **BuildKit**: https://github.com/moby/buildkit
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- :doc:`ci_cd` - CI/CD integration
- :doc:`contributing` - Contributing guidelines
- `docker/ Directory <https://github.com/Asher-1/ACloudViewer/tree/main/docker>`_ - Docker scripts and Dockerfiles

Quick Reference
---------------

**Build Images**:

.. code-block:: bash

   # Documentation
   docker build -t acloudviewer-ci:docs -f docker/Dockerfile.docs .
   
   # CPU CI
   docker build -t acloudviewer-ci:cpu -f docker/Dockerfile.ci .
   
   # CUDA CI
   docker build -t acloudviewer-ci:cuda \
       --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-devel-ubuntu22.04 \
       -f docker/Dockerfile.ci .
   
   # Qt6
   docker build -t acloudviewer-ci:qt6 -f docker/Dockerfile.ci.qt6 .
   
   # Wheel
   docker build -t acloudviewer-wheel:py311 \
       --build-arg PYTHON_VERSION=3.11 \
       -f docker/Dockerfile.wheel .

**Run Tests**:

.. code-block:: bash

   # Using test script
   cd docker && ./docker_test.sh cpu-static
   
   # Manual
   docker run --rm acloudviewer-ci:cpu \
       bash -c "cd build && ctest --output-on-failure"

**Build Wheels**:

.. code-block:: bash

   # Using build script
   cd docker && ./build_cloudviewer_whl.sh 3.11
   
   # Manual
   docker run --rm -v $(pwd):/opt/mount acloudviewer-wheel:py311 \
       bash -c "cp /root/ACloudViewer/dist/*.whl /opt/mount/"

**Clean Up**:

.. code-block:: bash

   # Remove all
   docker system prune -a --volumes -f
   
   # Check space
   docker system df
