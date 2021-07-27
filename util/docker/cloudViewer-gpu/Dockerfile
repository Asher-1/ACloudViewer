# Example build command:
# docker build \
#   --build-arg UBUNTU_VERSION=focal \
#   -t cloudViewer-gpu-ci-focal:latest \
#   -f ./util/docker/cloudViewer-gpu/Dockerfile \
#    .
# Example run command:
#    docker run --rm --gpus all \
#        --env NPROC=4 \
#        --env SHARED=OFF \
#        --env BUILD_CUDA_MODULE=ON \
#        --env BUILD_RPC_INTERFACE=ON \
#        --env BUILD_TENSORFLOW_OPS=ON \
#        --env BUILD_PYTORCH_OPS=ON \
#        --env OPEN3D_ML_ROOT=/root/CloudViewer/CloudViewer-ML \
#        cloudViewer-gpu-ci-focal:latest


ARG UBUNTU_VERSION
FROM ubuntu:${UBUNTU_VERSION} AS cloudViewer-gpu
# Persist ARG for the rest of the build
ARG UBUNTU_VERSION
LABEL name="cloudViewer-dev/cloudViewer-gpu-ci-${UBUNTU_VERSION}"
LABEL vendor="cloudViewer.org"
LABEL architecture="x86_64"
LABEL os="linux"
LABEL version="0.13.0"
LABEL maintainer="cloudViewer-info@osvf.org"

ENV DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles SUDO=command
# Install Python 3, cmake>=3.19 and nvidia drivers (only if not installed)
COPY ./util/docker/cloudViewer-gpu/scripts/env-setup.sh /root/CloudViewer/util/docker/cloudViewer-gpu/scripts/env-setup.sh
RUN /root/CloudViewer/util/docker/cloudViewer-gpu/scripts/env-setup.sh

# Install dependencies with apt-get and pip
COPY ./util/install_deps_ubuntu.sh ./util/ci_utils.sh /root/CloudViewer/util/
RUN /root/CloudViewer/util/install_deps_ubuntu.sh assume-yes

COPY ./util/ci_utils.sh /root/CloudViewer/util/
RUN bash -o errexit -c "source /root/CloudViewer/util/ci_utils.sh && \
        install_cuda_toolkit with-cudnn purge-cache"

# Persist PATH for cuda, cudnn and set requirements for host and container
ENV PATH=/usr/local/cuda/bin:$PATH \
         LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
         NVIDIA_VISIBLE_DEVICES=all \
         NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics" \
         NVIDIA_REQUIRE_CUDA="cuda>=11.0"

COPY . /root/CloudViewer

WORKDIR /root/CloudViewer
ENTRYPOINT /bin/bash