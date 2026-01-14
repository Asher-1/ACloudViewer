CI/CD Pipeline
==============

ACloudViewer uses automated CI/CD pipelines for continuous integration, testing, and deployment.

Overview
--------

The CI/CD system automates:

- **Building**: Compile across multiple platforms
- **Testing**: Run comprehensive test suites
- **Documentation**: Generate and deploy docs
- **Packaging**: Create release artifacts
- **Deployment**: Publish to GitHub Releases and PyPI

GitHub Actions Workflows
-------------------------

Main Workflows
^^^^^^^^^^^^^^

The project uses several GitHub Actions workflows located in ``.github/workflows/``:

.. list-table::
   :header-rows: 1

   * - Workflow
     - Trigger
     - Purpose
   * - ``ubuntu.yml``
     - Push, PR
     - Build and test on Ubuntu
   * - ``macos.yml``
     - Push, PR
     - Build and test on macOS
   * - ``windows.yml``
     - Push, PR
     - Build and test on Windows
   * - ``documentation.yml``
     - Push to main, Manual
     - Build and deploy documentation
   * - ``wheel.yml``
     - Release tag
     - Build Python wheels
   * - ``style-check.yml``
     - PR
     - Check code style

Documentation Workflow
^^^^^^^^^^^^^^^^^^^^^^

The documentation workflow (``documentation.yml``) is automatically triggered when:

- Changes are pushed to the ``main`` branch
- Manually triggered via "workflow_dispatch"

**Workflow Steps:**

1. **Generate downloads data**: Scan GitHub Releases dynamically
2. **Build Docker image**: Create documentation build environment
3. **Extract documentation**: Extract generated HTML from Docker
4. **Deploy to GitHub Pages**: Publish to ``https://asher-1.github.io/ACloudViewer/documentation/``

**Manual Trigger:**

.. code-block:: bash

   # Via GitHub UI:
   # 1. Go to Actions → Documentation
   # 2. Click "Run workflow"
   # 3. Select branch and DEVELOPER_BUILD option

Platform-Specific Builds
^^^^^^^^^^^^^^^^^^^^^^^^

Ubuntu Build
~~~~~~~~~~~~

.. code-block:: yaml

   name: Ubuntu
   
   on: [push, pull_request]
   
   jobs:
     build:
       runs-on: ubuntu-22.04
       steps:
         - uses: actions/checkout@v4
         - name: Build
           run: |
             mkdir build && cd build
             cmake .. -DCMAKE_BUILD_TYPE=Release
             make -j$(nproc)
         - name: Test
           run: |
             cd build
             ctest --output-on-failure

macOS Build
~~~~~~~~~~~

.. code-block:: yaml

   name: macOS
   
   on: [push, pull_request]
   
   jobs:
     build:
       runs-on: macos-latest
       steps:
         - uses: actions/checkout@v4
         - name: Install dependencies
           run: brew install cmake
         - name: Build
           run: |
             mkdir build && cd build
             cmake .. -DCMAKE_BUILD_TYPE=Release
             make -j$(sysctl -n hw.ncpu)

Windows Build
~~~~~~~~~~~~~

.. code-block:: yaml

   name: Windows
   
   on: [push, pull_request]
   
   jobs:
     build:
       runs-on: windows-latest
       steps:
         - uses: actions/checkout@v4
         - name: Build
           run: |
             mkdir build
             cd build
             cmake .. -G "Visual Studio 17 2022" -A x64
             cmake --build . --config Release

Wheel Building
^^^^^^^^^^^^^^

Python wheels are built for multiple platforms and Python versions:

.. code-block:: yaml

   strategy:
     matrix:
       os: [ubuntu-22.04, windows-latest, macos-latest]
       python-version: ['3.10', '3.11', '3.12']
   
   steps:
     - name: Build wheel
       run: python setup.py bdist_wheel

Local CI Testing
----------------

Test CI workflows locally using:

Act (GitHub Actions Local)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Install act
   brew install act  # macOS
   # or
   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
   
   # Run workflow locally
   act -j build  # Run 'build' job
   act pull_request  # Simulate PR event

Docker-based Testing
^^^^^^^^^^^^^^^^^^^^

Test in Docker environment:

.. code-block:: bash

   # Build test environment
   docker build -t acloudviewer-ci:test -f docker/Dockerfile.ci .
   
   # Run tests
   docker run --rm acloudviewer-ci:test \
       bash -c "cd build && ctest"

CI/CD Utilities
---------------

The ``util/ci_utils.sh`` script provides reusable CI functions:

Available Functions
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Install documentation dependencies
   install_docs_dependencies "${CLOUDVIEWER_ML_ROOT}"
   
   # Build documentation
   build_docs "$DEVELOPER_BUILD"
   
   # Build Python package
   build_pip_package
   
   # Test wheel
   test_wheel path/to/wheel.whl
   
   # Maximize GitHub Actions build space
   maximize_ubuntu_github_actions_build_space

Usage Example
^^^^^^^^^^^^^

.. code-block:: bash

   #!/bin/bash
   source util/ci_utils.sh
   
   # Setup environment
   export DEVELOPER_BUILD=ON
   export NPROC=$(nproc)
   
   # Build documentation
   build_docs "$DEVELOPER_BUILD"

Caching Strategies
------------------

ccache
^^^^^^

Speed up C++ compilation:

.. code-block:: yaml

   - name: Restore ccache
     uses: actions/cache@v3
     with:
       path: ~/.cache/ccache
       key: ${{ runner.os }}-ccache-${{ github.sha }}
       restore-keys: |
         ${{ runner.os }}-ccache-
   
   - name: Build with ccache
     run: |
       ccache -M 5G
       mkdir build && cd build
       cmake -DCMAKE_C_COMPILER_LAUNCHER=ccache \
             -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
       make -j$(nproc)
       ccache -s

pip Cache
^^^^^^^^^

Cache Python packages:

.. code-block:: yaml

   - name: Setup Python
     uses: actions/setup-python@v4
     with:
       python-version: '3.11'
       cache: 'pip'
   
   - name: Install dependencies
     run: pip install -r requirements.txt

Docker Layer Caching
^^^^^^^^^^^^^^^^^^^^

Cache Docker layers:

.. code-block:: yaml

   - name: Build Docker image
     uses: docker/build-push-action@v4
     with:
       context: .
       file: docker/Dockerfile.ci
       cache-from: type=gha
       cache-to: type=gha,mode=max

Artifact Management
-------------------

Upload Artifacts
^^^^^^^^^^^^^^^^

.. code-block:: yaml

   - name: Upload documentation
     uses: actions/upload-artifact@v4
     with:
       name: documentation
       path: docs/_out/html/
       retention-days: 30

Download Artifacts
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   - name: Download artifacts
     uses: actions/download-artifact@v4
     with:
       name: documentation
       path: ./docs

Release Management
------------------

Automated Releases
^^^^^^^^^^^^^^^^^^

Releases are triggered by pushing a tag:

.. code-block:: bash

   git tag -a v3.9.3 -m "Release version 3.9.3"
   git push origin v3.9.3

The release workflow:

1. Builds wheels for all platforms
2. Generates release notes from changelog
3. Uploads artifacts to GitHub Releases
4. Optionally publishes to PyPI

.. code-block:: yaml

   on:
     push:
       tags:
         - 'v*'
   
   jobs:
     release:
       runs-on: ubuntu-latest
       steps:
         - name: Create Release
           uses: softprops/action-gh-release@v1
           with:
             files: |
               dist/*.whl
               dist/*.tar.gz
             generate_release_notes: true

Pre-release Workflow
^^^^^^^^^^^^^^^^^^^^

For beta releases:

.. code-block:: yaml

   - name: Create Pre-release
     uses: softprops/action-gh-release@v1
     with:
       prerelease: true
       tag_name: v${{ github.ref_name }}-beta

Deployment Targets
------------------

GitHub Pages
^^^^^^^^^^^^

Documentation is deployed to:

- **Main Website**: ``https://asher-1.github.io/ACloudViewer/``
- **API Documentation**: ``https://asher-1.github.io/ACloudViewer/documentation/``

Deployment is automatic on push to ``main`` branch.

GitHub Releases
^^^^^^^^^^^^^^^

Release artifacts are published to:

``https://github.com/Asher-1/ACloudViewer/releases``

PyPI (Future)
^^^^^^^^^^^^^

Python wheels will be published to PyPI:

.. code-block:: bash

   pip install cloudviewer

Monitoring & Notifications
--------------------------

Build Status
^^^^^^^^^^^^

View build status in README:

.. image:: https://github.com/Asher-1/ACloudViewer/actions/workflows/ubuntu.yml/badge.svg
   :target: https://github.com/Asher-1/ACloudViewer/actions

Notifications
^^^^^^^^^^^^^

Configure GitHub notifications:

1. **Watch** the repository
2. **Custom** → Select "Workflows"
3. Receive email on:
   
   - Failed builds
   - Successful releases
   - Deployment status

Troubleshooting
---------------

Failed Builds
^^^^^^^^^^^^^

Check the workflow logs:

1. Go to **Actions** tab
2. Click on failed workflow run
3. Expand failed step
4. Review error messages

Common issues:

- **Out of disk space**: Clean build artifacts
- **Timeout**: Reduce parallel jobs (``-jN``)
- **Dependency issues**: Update ``requirements.txt``

Re-running Workflows
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Re-run failed jobs
   gh workflow run <workflow-name> --ref <branch>
   
   # Re-run all jobs
   gh run rerun <run-id>

Debugging CI
^^^^^^^^^^^^

Add debug output:

.. code-block:: yaml

   - name: Debug
     run: |
       echo "GitHub context:"
       echo "${{ toJSON(github) }}"
       
       echo "Environment:"
       env | sort

Best Practices
--------------

1. **Fast Feedback**: Run quick tests first
2. **Fail Fast**: Stop on first failure
3. **Parallel Execution**: Use matrix builds
4. **Caching**: Cache dependencies and build artifacts
5. **Minimal Rebuilds**: Use ccache and incremental builds
6. **Clear Logs**: Add informative echo statements
7. **Security**: Use secrets for sensitive data
8. **Resource Limits**: Set timeouts and resource constraints

Example Workflow
----------------

Complete example workflow:

.. code-block:: yaml

   name: Complete CI
   
   on: [push, pull_request]
   
   jobs:
     style-check:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - name: Check code style
           run: ./util/check_style.sh
   
     build-and-test:
       needs: style-check
       strategy:
         matrix:
           os: [ubuntu-22.04, macos-latest, windows-latest]
           python: ['3.10', '3.11', '3.12']
       runs-on: ${{ matrix.os }}
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v4
           with:
             python-version: ${{ matrix.python }}
         - name: Build
           run: |
             mkdir build && cd build
             cmake .. -DCMAKE_BUILD_TYPE=Release
             cmake --build . --parallel
         - name: Test
           run: |
             cd build
             ctest --output-on-failure
         - name: Upload artifacts
           if: failure()
           uses: actions/upload-artifact@v4
           with:
             name: test-logs-${{ matrix.os }}-py${{ matrix.python }}
             path: build/Testing/

Resources
---------

- `GitHub Actions Documentation <https://docs.github.com/en/actions>`_
- `Docker CI/CD Guide <https://docs.docker.com/ci-cd/>`_
- :doc:`docker` - Docker development
- :doc:`contributing` - Contributing guidelines
- `util/ci_utils.sh <https://github.com/Asher-1/ACloudViewer/blob/main/util/ci_utils.sh>`_ - CI utility functions
