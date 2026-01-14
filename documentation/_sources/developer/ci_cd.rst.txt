CI/CD Pipeline
==============

ACloudViewer uses automated CI/CD pipelines for continuous integration, testing, and deployment through GitHub Actions.

Overview
--------

The CI/CD system automates:

- **Building**: Compile across multiple platforms (Ubuntu, macOS, Windows)
- **Testing**: Run comprehensive test suites (C++ unit tests, Python tests)
- **Documentation**: Generate and deploy docs to GitHub Pages
- **Packaging**: Create Python wheels for multiple Python versions
- **Deployment**: Publish to GitHub Releases

GitHub Actions Workflows
-------------------------

Main Workflows
^^^^^^^^^^^^^^

The project uses several GitHub Actions workflows located in ``.github/workflows/``:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Workflow
     - Trigger
     - Purpose
   * - ``ubuntu.yml``
     - Push, PR, Release
     - Build and test on Ubuntu (CPU/CUDA variants)
   * - ``ubuntu-wheel.yml``
     - Push, PR, Release
     - Build Python wheels with CUDA support
   * - ``macos.yml``
     - Push, PR
     - Build and test on macOS
   * - ``windows.yml``
     - Push, PR
     - Build and test on Windows
   * - ``documentation.yml``
     - Push to main, PR, Manual
     - Build and deploy documentation
   * - ``ubuntu-cuda.yml``
     - Push, PR
     - Build and test with CUDA
   * - ``style.yml``
     - PR
     - Check code style (clang-format)
   * - ``webrtc.yml``
     - Push, PR
     - Build WebRTC dependencies
   * - ``vtk_packages.yml``
     - Manual
     - Build VTK packages

Documentation Workflow
^^^^^^^^^^^^^^^^^^^^^^

The documentation workflow (``documentation.yml``) is automatically triggered when:

- Changes are pushed to the ``main`` branch
- Pull requests are opened, reopened, or synchronized
- Manually triggered via "workflow_dispatch"

**Workflow Steps:**

1. **Setup Python 3.11**: For running documentation scripts
2. **Generate downloads_data.json**: Dynamically scan GitHub Releases
3. **Maximize build space**: Clean up disk space (remove dotnet, ghc, boost)
4. **Set up Docker Buildx**: Prepare multi-platform builds
5. **Build Docker image**: Create documentation environment (``Dockerfile.docs``)
6. **Extract documentation**: Extract generated HTML tarball from Docker
7. **Upload artifact**: Store documentation as GitHub artifact (30 days retention)
8. **Update main-devel release**: Upload to main-devel release (main branch only)
9. **Prepare website**: Merge main website with documentation
10. **Deploy to GitHub Pages**: Publish unified deployment

**Manual Trigger:**

.. code-block:: bash

   # Via GitHub UI:
   # 1. Go to Actions → Documentation
   # 2. Click "Run workflow"
   # 3. Select branch
   # 4. Set DEVELOPER_BUILD option (ON/OFF)

**Build Arguments:**

- ``DEVELOPER_BUILD``: ``ON`` (dev build with git hash) or ``OFF`` (release with version)

**Deployment Structure:**

.. code-block:: text

   https://asher-1.github.io/ACloudViewer/
   ├── index.html               # Main website
   ├── downloads_data.json      # Release information
   └── documentation/           # API documentation
       ├── index.html
       ├── python_api/          # Python API docs
       ├── cpp_api/             # C++ API docs (Doxygen)
       └── tutorial/            # Jupyter notebooks

Ubuntu Build Workflow
^^^^^^^^^^^^^^^^^^^^^^

**File**: ``ubuntu.yml``

**Matrix Strategy:**

.. code-block:: yaml

   matrix:
     include:
       - CI_CONFIG: cpu-focal      # Ubuntu 20.04 CPU
       - CI_CONFIG: cpu-jammy      # Ubuntu 22.04 CPU
       - CI_CONFIG: cpu-noble      # Ubuntu 24.04 CPU

**Key Features:**

- **Concurrency control**: Cancels previous runs on new pushes
- **Skip check**: Runs only on official repository (not forks)
- **Commit message check**: Skip with ``[skip ci]`` in commit message
- **Docker-based build**: Uses ``Dockerfile.ci``
- **Artifact upload**: Uploads build logs on failure

Ubuntu Wheel Workflow
^^^^^^^^^^^^^^^^^^^^^^

**File**: ``ubuntu-wheel.yml``

**Matrix Strategy:**

.. code-block:: yaml

   matrix:
     python_version: ['3.10', '3.11', '3.12', '3.13']
     # Full matrix only on main branch
     # PR builds use Python 3.13 only

**Build Configuration:**

.. code-block:: yaml

   env:
     BUILD_CUDA_MODULE: 'ON'
     BUILD_PYTORCH_OPS: 'ON'
     BUILD_TENSORFLOW_OPS: 'OFF'  # cxx11_abi compatibility

**Key Steps:**

1. **Maximize build space**: Remove unnecessary packages (~10GB)
2. **Setup ccache**: Restore compilation cache (~5GB)
3. **Build Docker image**: ``Dockerfile.wheel``
4. **Build wheel**: Inside Docker container
5. **Test wheel**: Run pytest with installed wheel
6. **Upload to release**: On main branch or release tags

macOS Build Workflow
^^^^^^^^^^^^^^^^^^^^

**File**: ``macos.yml``

**Matrix Strategy:**

.. code-block:: yaml

   matrix:
     os: [macos-12, macos-13, macos-14]
     include:
       - python_version: '3.11'

**Build Process:**

1. Install dependencies via ``brew``
2. Build with CMake (Release mode)
3. Run tests with ``ctest``
4. Upload build artifacts

Windows Build Workflow
^^^^^^^^^^^^^^^^^^^^^^^

**File**: ``windows.yml``

**Build Configuration:**

.. code-block:: yaml

   strategy:
     matrix:
       python_version: ['3.10', '3.11', '3.12']
       build_type: [Release]

**Key Features:**

- Visual Studio 2022 compiler
- CUDA 12.1 support
- Python wheel building
- Automated testing

Local CI Testing
----------------

Test CI workflows locally using Docker:

Docker-based Testing
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Build CPU test environment
   docker build -t acloudviewer-ci:cpu \
       -f docker/Dockerfile.ci \
       --build-arg BASE_IMAGE=ubuntu:22.04 \
       --build-arg DEVELOPER_BUILD=ON .
   
   # Run tests
   docker run --rm acloudviewer-ci:cpu \
       bash -c "cd build && ctest --output-on-failure"

   # Build CUDA test environment
   docker build -t acloudviewer-ci:cuda \
       -f docker/Dockerfile.ci \
       --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-devel-ubuntu22.04 \
       --build-arg DEVELOPER_BUILD=ON .
   
   # Run GPU tests
   docker run --rm --gpus all acloudviewer-ci:cuda \
       bash -c "cd build && ctest --output-on-failure"

Act (GitHub Actions Local Simulator)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Install act (macOS)
   brew install act
   
   # Install act (Linux)
   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
   
   # List available workflows
   act -l
   
   # Run specific job
   act -j build  # Run 'build' job
   
   # Simulate pull request
   act pull_request
   
   # Use specific Docker image
   act -P ubuntu-latest=catthehacker/ubuntu:act-latest

CI/CD Utilities
---------------

The ``util/ci_utils.sh`` script provides reusable CI functions:

Available Functions
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Install documentation dependencies
   install_docs_dependencies "${CLOUDVIEWER_ML_ROOT}"
   
   # Build documentation (ON=dev, OFF=release)
   build_docs "$DEVELOPER_BUILD"
   
   # Build Python wheel
   build_pip_package
   
   # Test wheel installation
   test_wheel path/to/wheel.whl
   
   # Run C++ unit tests
   run_cpp_unit_tests
   
   # Run Python unit tests
   run_python_tests
   
   # Maximize GitHub Actions build space (~10GB)
   maximize_ubuntu_github_actions_build_space
   
   # Purge cache
   purge_cache

Usage Example
^^^^^^^^^^^^^

.. code-block:: bash

   #!/bin/bash
   source util/ci_utils.sh
   
   # Setup environment
   export DEVELOPER_BUILD=ON
   export NPROC=$(nproc)
   export CLOUDVIEWER_SOURCE_ROOT=$(pwd)
   
   # Build documentation
   build_docs "$DEVELOPER_BUILD"
   
   # Build and test wheel
   build_pip_package
   test_wheel dist/*.whl

Caching Strategies
------------------

ccache (C++ Compilation)
^^^^^^^^^^^^^^^^^^^^^^^^^

Speed up C++ compilation with ccache:

.. code-block:: yaml

   - name: Restore ccache
     uses: actions/cache@v4
     with:
       path: /home/runner/.cache/ccache
       key: ${{ runner.os }}-ccache-${{ hashFiles('**/CMakeLists.txt') }}
       restore-keys: |
         ${{ runner.os }}-ccache-
   
   - name: Configure ccache
     run: |
       ccache -M 5G
       ccache -s
   
   - name: Build with ccache
     run: |
       mkdir build && cd build
       cmake -DCMAKE_C_COMPILER_LAUNCHER=ccache \
             -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
       make -j$(nproc)
       ccache -s

**Cache Benefits:**

- First build: ~45 min
- Cached build: ~10 min (4.5x speedup)

pip Cache (Python Dependencies)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cache Python packages:

.. code-block:: yaml

   - name: Setup Python
     uses: actions/setup-python@v5
     with:
       python-version: '3.11'
       cache: 'pip'
       cache-dependency-path: |
         python/requirements*.txt
         docs/requirements.txt
   
   - name: Install dependencies
     run: |
       pip install -U pip
       pip install -r python/requirements.txt

Docker Layer Caching
^^^^^^^^^^^^^^^^^^^^

Cache Docker layers with BuildKit:

.. code-block:: yaml

   - name: Set up Docker Buildx
     uses: docker/setup-buildx-action@v3
   
   - name: Build Docker image
     uses: docker/build-push-action@v5
     with:
       context: .
       file: docker/Dockerfile.ci
       cache-from: type=gha
       cache-to: type=gha,mode=max
       tags: acloudviewer-ci:cpu

**Cache Benefits:**

- First build: ~60 min
- Cached build: ~5 min (12x speedup)

Artifact Management
-------------------

Upload Artifacts
^^^^^^^^^^^^^^^^

.. code-block:: yaml

   - name: Upload documentation
     uses: actions/upload-artifact@v4
     with:
       name: acloudviewer-${{ github.sha }}-docs
       path: acloudviewer-*-docs.tar.gz
       retention-days: 30
       compression-level: 0  # Already compressed

   - name: Upload wheel
     uses: actions/upload-artifact@v4
     with:
       name: wheel-py${{ matrix.python_version }}
       path: dist/*.whl
       retention-days: 90

Download Artifacts
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   - name: Download documentation artifact
     uses: actions/download-artifact@v4
     with:
       name: acloudviewer-${{ github.sha }}-docs
       path: ./docs-download

Release Management
------------------

Automated Releases
^^^^^^^^^^^^^^^^^^

Releases are triggered by pushing a tag:

.. code-block:: bash

   # Create and push release tag
   git tag -a v3.9.3 -m "Release version 3.9.3"
   git push origin v3.9.3

**Release Workflow Actions:**

1. **Detect release**: Triggered on ``release: types: [released]``
2. **Build wheels**: All Python versions, all platforms
3. **Build GUI apps**: Platform-specific binaries
4. **Generate changelog**: From commit messages
5. **Upload to GitHub Releases**: Attach all artifacts
6. **Set DEVELOPER_BUILD=OFF**: Production builds

**Release Assets:**

.. code-block:: text

   v3.9.3/
   ├── cloudViewer-3.9.3-cp310-cp310-manylinux_2_27_x86_64.whl
   ├── cloudViewer-3.9.3-cp311-cp311-manylinux_2_27_x86_64.whl
   ├── cloudViewer-3.9.3-cp312-cp312-manylinux_2_27_x86_64.whl
   ├── cloudViewer-3.9.3-cp313-cp313-manylinux_2_27_x86_64.whl
   ├── ACloudViewer-3.9.3-Windows.exe
   ├── ACloudViewer-3.9.3-macOS.dmg
   └── acloudviewer-3.9.3-docs.tar.gz

Pre-release Workflow
^^^^^^^^^^^^^^^^^^^^

For beta/rc releases:

.. code-block:: bash

   # Create pre-release
   git tag -a v3.9.4-beta1 -m "Beta release 3.9.4-beta1"
   git push origin v3.9.4-beta1

**Pre-release Configuration:**

.. code-block:: yaml

   on:
     push:
       tags:
         - 'v*-beta*'
         - 'v*-rc*'
   
   jobs:
     release:
       steps:
         - name: Create Pre-release
           uses: softprops/action-gh-release@v1
           with:
             prerelease: true
             generate_release_notes: true

Deployment Targets
------------------

GitHub Pages
^^^^^^^^^^^^

**URL Structure:**

- **Main Website**: ``https://asher-1.github.io/ACloudViewer/``
- **API Documentation**: ``https://asher-1.github.io/ACloudViewer/documentation/``
- **Python API**: ``https://asher-1.github.io/ACloudViewer/documentation/python_api/``
- **C++ API**: ``https://asher-1.github.io/ACloudViewer/documentation/cpp_api/``
- **Tutorials**: ``https://asher-1.github.io/ACloudViewer/documentation/tutorial/``

**Deployment Process:**

.. code-block:: yaml

   - name: Deploy to GitHub Pages
     uses: peaceiris/actions-gh-pages@v3
     with:
       github_token: ${{ secrets.GITHUB_TOKEN }}
       publish_dir: ./unified-deploy
       keep_files: false
       enable_jekyll: false
       force_orphan: true

**Deployment Frequency:**

- Automatic on every push to ``main``
- Manual via workflow dispatch
- Average deployment time: ~5 minutes

GitHub Releases
^^^^^^^^^^^^^^^

Release artifacts are published to:

``https://github.com/Asher-1/ACloudViewer/releases``

**Asset Types:**

- Python wheels (``.whl``)
- Documentation (``.tar.gz``)
- GUI applications (``.exe``, ``.dmg``, ``.AppImage``)
- Source archives (``.tar.gz``, ``.zip``)

Monitoring & Notifications
--------------------------

Build Status Badges
^^^^^^^^^^^^^^^^^^^

Add status badges to README:

.. code-block:: markdown

   ![Ubuntu](https://github.com/Asher-1/ACloudViewer/actions/workflows/ubuntu.yml/badge.svg)
   ![macOS](https://github.com/Asher-1/ACloudViewer/actions/workflows/macos.yml/badge.svg)
   ![Windows](https://github.com/Asher-1/ACloudViewer/actions/workflows/windows.yml/badge.svg)
   ![Documentation](https://github.com/Asher-1/ACloudViewer/actions/workflows/documentation.yml/badge.svg)

Notifications
^^^^^^^^^^^^^

Configure GitHub notifications:

1. **Repository Settings** → **Notifications**
2. **Watch** → **Custom**
3. Select:
   
   - ✅ Releases
   - ✅ Actions (workflow failures)
   - ❌ All Activity (too noisy)

**Email Notifications:**

- Workflow failures
- Successful releases
- Deployment status

Troubleshooting
---------------

Failed Builds
^^^^^^^^^^^^^

**Common Issues:**

1. **Out of disk space**
   
   **Solution**: Run ``maximize_ubuntu_github_actions_build_space``
   
   .. code-block:: bash
   
      source util/ci_utils.sh
      maximize_ubuntu_github_actions_build_space

2. **ccache miss**
   
   **Solution**: Check cache key, ensure CMakeLists.txt hash is correct

3. **Timeout (6 hours limit)**
   
   **Solution**: Reduce parallel jobs or split workflow

4. **Docker build failures**
   
   **Solution**: Check Docker logs, verify base image availability

5. **Test failures**
   
   **Solution**: Review test logs, reproduce locally

**Debugging Steps:**

1. View workflow logs in GitHub Actions
2. Reproduce locally with Docker
3. Check artifact uploads for detailed logs
4. Enable debug logging:

   .. code-block:: yaml
   
      env:
        ACTIONS_STEP_DEBUG: true
        ACTIONS_RUNNER_DEBUG: true

Re-running Workflows
^^^^^^^^^^^^^^^^^^^^

**Via GitHub UI:**

1. Go to **Actions** tab
2. Select failed workflow run
3. Click **Re-run jobs** → **Re-run failed jobs**

**Via GitHub CLI:**

.. code-block:: bash

   # Install GitHub CLI
   brew install gh  # macOS
   # or
   sudo apt install gh  # Ubuntu
   
   # Authenticate
   gh auth login
   
   # Re-run workflow
   gh workflow run documentation.yml --ref main
   
   # Re-run specific run
   gh run rerun <run-id>
   
   # View workflow status
   gh run list --workflow=documentation.yml

Best Practices
--------------

1. **Fast Feedback Loop**
   
   - Run style checks first (fastest)
   - Build and test in parallel
   - Cache aggressively

2. **Fail Fast**
   
   - Stop on first failure
   - Don't waste resources on doomed builds

3. **Resource Optimization**
   
   - Use matrix builds for parallel execution
   - Limit concurrent jobs to avoid queue saturation
   - Clean up artifacts after successful merges

4. **Security**
   
   - Use ``secrets`` for sensitive data
   - Restrict permissions (``permissions: {}`` by default)
   - Review third-party actions before use

5. **Maintainability**
   
   - Keep workflows DRY (reuse ci_utils.sh)
   - Document complex logic
   - Version-pin action dependencies

Example: Complete CI Workflow
------------------------------

.. code-block:: yaml

   name: Complete CI Pipeline
   
   on: [push, pull_request]
   
   concurrency:
     group: ${{ github.workflow }}-${{ github.ref }}
     cancel-in-progress: true
   
   jobs:
     style-check:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - name: Check code style
           run: |
             find . -name "*.cpp" -o -name "*.h" | \
               xargs clang-format --dry-run --Werror
   
     build-and-test:
       needs: style-check
       strategy:
         fail-fast: false
         matrix:
           os: [ubuntu-22.04, macos-latest, windows-latest]
           python: ['3.11', '3.12']
       runs-on: ${{ matrix.os }}
       steps:
         - uses: actions/checkout@v4
         
         - uses: actions/setup-python@v5
           with:
             python-version: ${{ matrix.python }}
             cache: 'pip'
         
         - name: Restore ccache
           uses: actions/cache@v4
           with:
             path: ~/.cache/ccache
             key: ${{ runner.os }}-ccache-${{ hashFiles('**/CMakeLists.txt') }}
         
         - name: Build
           run: |
             mkdir build && cd build
             cmake .. -DCMAKE_BUILD_TYPE=Release \
                      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
             cmake --build . --parallel
         
         - name: Test
           run: |
             cd build
             ctest --output-on-failure --parallel
         
         - name: Upload test logs
           if: failure()
           uses: actions/upload-artifact@v4
           with:
             name: test-logs-${{ matrix.os }}-py${{ matrix.python }}
             path: build/Testing/Temporary/

Performance Metrics
-------------------

**Typical Build Times (without cache):**

.. list-table::
   :header-rows: 1

   * - Platform
     - Build Time
     - Test Time
     - Total
   * - Ubuntu (CPU)
     - 45 min
     - 5 min
     - 50 min
   * - Ubuntu (CUDA)
     - 60 min
     - 10 min
     - 70 min
   * - macOS
     - 40 min
     - 8 min
     - 48 min
   * - Windows
     - 55 min
     - 12 min
     - 67 min

**With ccache:**

- Build time reduced by **75-80%**
- Typical incremental build: **10-15 minutes**

Resources
---------

- `GitHub Actions Documentation <https://docs.github.com/en/actions>`_
- `Docker CI/CD Guide <https://docs.docker.com/ci-cd/>`_
- :doc:`docker` - Docker development guide
- :doc:`contributing` - Contributing guidelines
- `util/ci_utils.sh <https://github.com/Asher-1/ACloudViewer/blob/main/util/ci_utils.sh>`_ - CI utility functions
- `GitHub Actions Workflow Syntax <https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions>`_
