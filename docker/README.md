# CloudViewer Docker

## Dependencies

### Docker dependencies

- [Install Docker](https://docs.docker.com/get-docker/).
- [Post-installation steps for linux](https://docs.docker.com/engine/install/linux-postinstall/).
  Make sure that `docker` can be executed without root privileges.

To verify that Docker is working, run:

```bash
# You should be able to run this without sudo.
docker run --rm hello-world
```

### Nvidia Docker

You don't need to install Nvidia Docker to build CUDA container. You will need
to install Nvidia Docker to run the CUDA container.

- [Install Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).
  This is required for testing CUDA builds.

To verify that the Nvidia Docker is working, run:

```bash
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### ARM64 Docker

You can build and run ARM64 docker. This works on an ARM64 host including Apple
Silicon. However, if your host is x86-64, you will need to install QEMU:

```bash
sudo apt-get --yes install qemu binfmt-support qemu-user-static

# Run the registering scripts
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

To verify that the ARM64 environment is working, run:

```bash
# This shall print "aarch64".
# The following warning message is expected: "WARNING: The requested image's
# platform (linux/arm64/v8) does not match the detected host platform
# (linux/amd64) and no specific platform was requested aarch64."
docker run --rm arm64v8/ubuntu:24.04 uname -p
```

## Build and test Docker

For example:
```bash
# build for publish release version
./docker/build-release.sh

# build all platforms
./docker/build-all.sh
```

```bash
cd docker

# Build Docker.
./docker_build.sh cuda_wheel_py312_dev

# Test Docker image.
./docker_test.sh cuda_wheel_py312_dev
```

See `./docker_build.sh` and `./docker_test.sh` for all available options.

### Testing Docker Images

The `docker_test.sh` script runs comprehensive tests on built Docker images:

- **C++ Unit Tests**: Runs Google Test suite
- **Python Unit Tests**: Runs pytest with automatic ML ops detection
  - ML ops tests are automatically included/excluded based on the installed `cloudViewer` package's build configuration
  - The script dynamically checks `cloudViewer._build_config['BUILD_PYTORCH_OPS']` and `cloudViewer._build_config['BUILD_TENSORFLOW_OPS']` to determine if ML ops tests should run
- **Command-line Tools Test**: Tests CLI availability
- **C++ Linking Test**: Tests CMake find_package integration
- **Uninstall Test**: Tests package uninstallation

Example test commands:

```bash
# Test CUDA wheel builds (includes ML ops)
./docker/docker_test.sh cuda_wheel_py312_dev
./docker/docker_test.sh cuda_wheel_py312

# Test CI builds (matches GitHub Actions CI_CONFIG)
./docker/docker_test.sh cpu-jammy
./docker/docker_test.sh cuda-jammy
./docker/docker_test.sh cpu-jammy-release
./docker/docker_test.sh cuda-jammy-release

# Test Qt6 builds
./docker/docker_test.sh qt6-cpu
./docker/docker_test.sh qt6-cuda
```

**Note**: The test script automatically detects whether ML ops are available in the Docker image by checking the installed `cloudViewer` package's build configuration. You don't need to manually specify whether ML ops are enabled.


## Building for Linux under Windows

You can build and test  for Linux in a Docker container under Windows using the provided scripts thanks to **[Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)** and **[WSL](https://learn.microsoft.com/en-us/windows/wsl/about)**.

This guide walks you through installing Docker Desktop, setting up Windows Subsystem for Linux (WSL), configuring Docker integration with WSL, and building ACloudViewer for Linux including its documentation and the Python wheel, and testing it, using the provided scripts (respectively `docker_build.sh` and `docker_test.sh`).

### Step 1: Install Docker Desktop

1. **Download and Install**: [Download Docker Desktop](https://www.docker.com/products/docker-desktop) and follow the on-screen prompts to install.
2. **Launch Docker Desktop**: After installation, open Docker Desktop to ensure it is running.

### Step 2: Install and Set Up WSL

1. **Enable WSL**: Open PowerShell as Administrator and install WSL:

   ```powershell
   wsl --install
   ```

2. **Install a Linux Distribution** (e.g., Ubuntu-24.04):

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

3. **Restart** your system if prompted.

### Step 3: Enable Docker Integration with WSL

1. **Open Docker Desktop**.
2. **Go to Settings** > **Resources** > **WSL Integration**.
3. **Enable Integration** for your Linux distribution (e.g., Ubuntu-24.04).
4. If necessary, **restart Docker Desktop** to apply the changes.

### Step 4: Clone and check out ACloudViewer repository

1. **Open a terminal** within WSL.
2. **Clone and check out** ACloudViewer repository into the folder of your choice:

   ```bash
   git clone https://github.com/Asher-1/ACloudViewer /path/to/ACloudViewer
   ```

### Step 5: Build ACloudViewer for Linux in WSL using the provided script

1. **Open your WSL terminal**.
2. **Navigate** to the Docker folder in the ACloudViewer repository:

   ```bash
   cd /path/to/ACloudViewer/docker
   ```

3. **Disable PyTorch and TensorFlow ops** if not needed:

   ```bash
   export BUILD_PYTORCH_OPS=OFF
   export BUILD_TENSORFLOW_OPS=OFF
   ```

4. **Run the Docker build script**:

   Available build options:
   - **CPU builds**: `cpu-focal`, `cpu-jammy`, `cpu-noble` (Ubuntu 20.04, 22.04, 24.04)
   - **CUDA builds**: `cuda-focal`, `cuda-jammy`, `cuda-noble` (requires NVIDIA GPU support)
   - **CUDA wheels**: `cuda_wheel_py310_dev`, `cuda_wheel_py311_dev`, `cuda_wheel_py312_dev`, `cuda_wheel_py313_dev`
   - Add `-release` suffix for release mode builds (e.g., `cpu-jammy-release`)

   Example for CPU build:

   ```bash
   ./docker_build.sh cpu-jammy
   ```

   Example for CUDA wheel build (ML ops are automatically enabled for wheel builds):

   ```bash
   ./docker_build.sh cuda_wheel_py312_dev
   ```

   Check the log of the build. After the build completes, you will have an ACloudViewer Docker image ready to use, and the artifacts (binaries, documentation and Python package) will have been copied back to the host.

5. **Run tests within the built Docker image**:

   Use the same option name as the build command:

   ```bash
   ./docker_test.sh cpu-jammy
   ```

   Or for CUDA builds:

   ```bash
   ./docker_test.sh cuda-jammy
   ```

   The test script will automatically detect whether ML ops are available and run the appropriate tests.

## Build Documentation Docker

Build the documentation in Docker container:

```bash
# Build documentation Docker image
docker build --network=host -t acloudviewer-ci:docs -f docker/Dockerfile.docs . > docker_docs_build.log 2>&1

# Extract the generated documentation package
docker run -v $(pwd):/opt/mount --rm acloudviewer-ci:docs \
  bash -c "cp /root/ACloudViewer/acloudviewer-*-docs.tar.gz /opt/mount/"

# Extract and preview the documentation
mkdir -p docs-output
tar -xzf acloudviewer-*-docs.tar.gz -C ./docs-output/
cd docs-output && python3 -m http.server 8080
# Open http://localhost:8080 in your browser
```

The documentation Docker image:
- Builds Python module (if not already built)
- Generates C++ API documentation (Doxygen)
- Generates Python API documentation (Sphinx)
- Packages everything into a tarball for deployment