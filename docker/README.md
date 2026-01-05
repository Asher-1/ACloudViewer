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
# Note: select Ubuntu18.04 for building OS due to compatibility
./docker/build-release.sh

# build all platforms
./docker/build-all.sh
```

```bash
cd docker

# Build Docker.
./docker/docker_build.sh cuda_wheel_py312_dev

See `./docker_build.sh` for all available options.

Note: only Ubuntu24.04+ support QT6