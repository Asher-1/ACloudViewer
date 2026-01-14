Docker Development
==================

ACloudViewer provides Docker images for consistent development and deployment environments.

Available Docker Images
-----------------------

Documentation Build
^^^^^^^^^^^^^^^^^^^

Build documentation in an isolated environment:

.. code-block:: bash

   # Build documentation Docker image
   docker build -t acloudviewer-ci:docs \
       -f docker/Dockerfile.docs .
   
   # Extract generated documentation
   docker run -v $(pwd):/opt/mount --rm acloudviewer-ci:docs \
       bash -c "cp /root/ACloudViewer/acloudviewer-*-docs.tar.gz /opt/mount/"

CI/CD Build
^^^^^^^^^^^

Build wheels and run tests:

.. code-block:: bash

   # Build CI image
   docker build -t acloudviewer-ci:cpu \
       -f docker/Dockerfile.ci \
       --build-arg BASE_IMAGE=ubuntu:22.04 \
       --build-arg DEVELOPER_BUILD=ON .

Wheel Building
^^^^^^^^^^^^^^

Build Python wheels for distribution:

.. code-block:: bash

   # Build wheel image
   docker build -t acloudviewer-wheel:cpu \
       -f docker/Dockerfile.wheel \
       --build-arg BASE_IMAGE=ubuntu:22.04 .

GPU Support
^^^^^^^^^^^

Build with CUDA support:

.. code-block:: bash

   # CUDA 12.1 image
   docker build -t acloudviewer-ci:cuda \
       -f docker/Dockerfile.ci \
       --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-devel-ubuntu22.04 \
       --build-arg DEVELOPER_BUILD=ON .

Using Docker Images
-------------------

Interactive Development
^^^^^^^^^^^^^^^^^^^^^^^

Launch an interactive shell:

.. code-block:: bash

   docker run -it --rm \
       -v $(pwd):/workspace \
       acloudviewer-ci:cpu \
       bash

With GPU:

.. code-block:: bash

   docker run -it --rm --gpus all \
       -v $(pwd):/workspace \
       acloudviewer-ci:cuda \
       bash

Running Tests
^^^^^^^^^^^^^

Run the complete test suite:

.. code-block:: bash

   docker run --rm \
       -v $(pwd):/workspace \
       acloudviewer-ci:cpu \
       bash -c "cd /workspace/build && ctest"

Building Wheels
^^^^^^^^^^^^^^^

Build Python wheels inside Docker:

.. code-block:: bash

   docker run --rm \
       -v $(pwd):/workspace \
       acloudviewer-wheel:cpu \
       bash -c "cd /workspace && util/ci_utils.sh build_pip_package"

Docker Compose
--------------

For multi-container setups, use ``docker-compose``:

.. code-block:: yaml

   version: '3'
   services:
     acloudviewer-dev:
       build:
         context: .
         dockerfile: docker/Dockerfile.ci
         args:
           BASE_IMAGE: ubuntu:22.04
           DEVELOPER_BUILD: 'ON'
       volumes:
         - .:/workspace
       command: bash

Usage:

.. code-block:: bash

   docker-compose up -d
   docker-compose exec acloudviewer-dev bash

Build Arguments
---------------

Common Build Arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``BASE_IMAGE``: Base Docker image (default: ``ubuntu:22.04``)
- ``DEVELOPER_BUILD``: Enable development features (``ON``/``OFF``)
- ``PYTHON_VERSION``: Python version (``3.10``, ``3.11``, ``3.12``)
- ``CMAKE_VERSION``: CMake version
- ``CCACHE_TAR_NAME``: ccache archive name for faster rebuilds

Example:

.. code-block:: bash

   docker build \
       --build-arg BASE_IMAGE=ubuntu:22.04 \
       --build-arg DEVELOPER_BUILD=OFF \
       --build-arg PYTHON_VERSION=3.11 \
       -t acloudviewer:custom \
       -f docker/Dockerfile.ci .

Dockerfile Structure
--------------------

Typical Dockerfile structure:

.. code-block:: dockerfile

   FROM ubuntu:22.04
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       cmake \
       git \
       python3-dev
   
   # Install Python dependencies
   COPY requirements.txt .
   RUN pip3 install -r requirements.txt
   
   # Copy source
   COPY . /workspace
   WORKDIR /workspace
   
   # Build ACloudViewer
   RUN mkdir build && cd build && \
       cmake .. && \
       make -j$(nproc)

Multi-Stage Builds
^^^^^^^^^^^^^^^^^^

Optimize image size with multi-stage builds:

.. code-block:: dockerfile

   # Stage 1: Build
   FROM ubuntu:22.04 as builder
   RUN apt-get update && apt-get install -y build-essential cmake
   COPY . /workspace
   RUN cd /workspace && mkdir build && cd build && \
       cmake .. && make -j$(nproc)
   
   # Stage 2: Runtime
   FROM ubuntu:22.04
   COPY --from=builder /workspace/build/lib /usr/local/lib
   COPY --from=builder /workspace/build/bin /usr/local/bin

Caching Strategies
------------------

Speed up builds with effective caching:

Layer Optimization
^^^^^^^^^^^^^^^^^^

Order Dockerfile commands from least to most frequently changing:

.. code-block:: dockerfile

   # 1. System packages (rarely change)
   RUN apt-get update && apt-get install -y ...
   
   # 2. Python requirements (occasional changes)
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   # 3. Source code (frequent changes)
   COPY . /workspace

Build Cache
^^^^^^^^^^^

Use BuildKit for advanced caching:

.. code-block:: bash

   DOCKER_BUILDKIT=1 docker build \
       --cache-from acloudviewer-ci:cpu \
       -t acloudviewer-ci:cpu \
       -f docker/Dockerfile.ci .

ccache Integration
^^^^^^^^^^^^^^^^^^

Leverage ccache for C++ compilation:

.. code-block:: bash

   docker run --rm \
       -v ccache-vol:/root/.cache/ccache \
       -v $(pwd):/workspace \
       acloudviewer-ci:cpu \
       bash -c "cd /workspace/build && make -j$(nproc)"

Troubleshooting
---------------

Permission Issues
^^^^^^^^^^^^^^^^^

Run as current user:

.. code-block:: bash

   docker run --rm \
       --user $(id -u):$(id -g) \
       -v $(pwd):/workspace \
       acloudviewer-ci:cpu

Out of Memory
^^^^^^^^^^^^^

Limit parallel builds:

.. code-block:: bash

   docker run --rm \
       -m 8g \
       -v $(pwd):/workspace \
       acloudviewer-ci:cpu \
       bash -c "make -j4"  # Limit to 4 jobs

Network Issues
^^^^^^^^^^^^^^

Use host network:

.. code-block:: bash

   docker build --network=host \
       -t acloudviewer-ci:cpu \
       -f docker/Dockerfile.ci .

Best Practices
--------------

1. **Use specific base images**: Pin versions for reproducibility
2. **Minimize layers**: Combine RUN commands
3. **Clean up in same layer**: Remove temporary files before layer commit
4. **Use .dockerignore**: Exclude unnecessary files
5. **Multi-stage builds**: Separate build and runtime environments
6. **Health checks**: Add health check for long-running containers
7. **Logging**: Configure appropriate logging drivers

Example .dockerignore:

.. code-block:: text

   **/.git
   **/build
   **/build_*
   **/__pycache__
   **/*.pyc
   **/.pytest_cache
   **/docs/_out
   **/docs/_build

Continuous Integration
----------------------

GitHub Actions
^^^^^^^^^^^^^^

See :doc:`ci_cd` for GitHub Actions integration.

GitLab CI
^^^^^^^^^

Example ``.gitlab-ci.yml``:

.. code-block:: yaml

   stages:
     - build
     - test
   
   build:
     stage: build
     image: docker:latest
     services:
       - docker:dind
     script:
       - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
       - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

Resources
---------

- `Docker Documentation <https://docs.docker.com/>`_
- `Docker Best Practices <https://docs.docker.com/develop/dev-best-practices/>`_
- `BuildKit <https://github.com/moby/buildkit>`_
- :doc:`ci_cd` - CI/CD integration
- :doc:`contributing` - Contributing guidelines
