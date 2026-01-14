Contributing to ACloudViewer
=============================

We welcome contributions from the community! This guide will help you get started with contributing to ACloudViewer.

Getting Started
---------------

1. **Fork the Repository**
   
   Fork the `ACloudViewer repository <https://github.com/Asher-1/ACloudViewer>`_ on GitHub.

2. **Clone Your Fork**
   
   .. code-block:: bash
   
      git clone https://github.com/YOUR_USERNAME/ACloudViewer.git
      cd ACloudViewer

3. **Set Up Development Environment**
   
   Follow the :ref:`compilation` guide to build ACloudViewer from source.

Development Workflow
--------------------

Creating a Branch
^^^^^^^^^^^^^^^^^

Create a new branch for your feature or bug fix:

.. code-block:: bash

   git checkout -b feature/your-feature-name

Making Changes
^^^^^^^^^^^^^^

1. Make your changes to the codebase
2. Follow the coding style guidelines (see below)
3. Add tests for new features
4. Update documentation as needed

Running Tests
^^^^^^^^^^^^^

Before submitting your changes, run the test suite:

.. code-block:: bash

   cd build
   make tests -j$(nproc)
   ctest

For Python tests:

.. code-block:: bash

   pytest python/test/

Code Style Guidelines
---------------------

C++ Code Style
^^^^^^^^^^^^^^

- Follow the `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html>`_
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 120 characters
- Use ``clang-format`` for automatic formatting

Python Code Style
^^^^^^^^^^^^^^^^^

- Follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
- Use 4 spaces for indentation
- Maximum line length: 88 characters (Black formatter)
- Use type hints where appropriate

Documentation
^^^^^^^^^^^^^

- Document all public APIs with clear docstrings
- Use Doxygen comments for C++ code
- Use Google-style docstrings for Python code
- Update relevant documentation in ``docs/`` directory

Commit Guidelines
-----------------

Commit Messages
^^^^^^^^^^^^^^^

Follow the `Conventional Commits <https://www.conventionalcommits.org/>`_ specification:

.. code-block:: text

   <type>(<scope>): <subject>
   
   <body>
   
   <footer>

Types:

- ``feat``: New feature
- ``fix``: Bug fix
- ``docs``: Documentation changes
- ``style``: Code style changes (formatting, etc.)
- ``refactor``: Code refactoring
- ``test``: Adding or updating tests
- ``chore``: Maintenance tasks

Examples:

.. code-block:: text

   feat(io): add support for LAZ format
   
   fix(visualization): correct point cloud rendering issue
   
   docs: update installation instructions

Submitting Changes
------------------

Creating a Pull Request
^^^^^^^^^^^^^^^^^^^^^^^

1. Push your changes to your fork:

   .. code-block:: bash
   
      git push origin feature/your-feature-name

2. Go to the `ACloudViewer repository <https://github.com/Asher-1/ACloudViewer>`_ and create a pull request

3. Fill out the pull request template with:
   
   - Clear description of changes
   - Related issue numbers
   - Testing performed
   - Screenshots (if applicable)

Pull Request Review
^^^^^^^^^^^^^^^^^^^

- Maintainers will review your PR
- Address feedback and update your branch as needed
- Once approved, your PR will be merged

Community Guidelines
--------------------

Code of Conduct
^^^^^^^^^^^^^^^

We follow the `Contributor Covenant Code of Conduct <https://www.contributor-covenant.org/>`_. 

Be respectful and constructive in all interactions.

Getting Help
^^^^^^^^^^^^

- **Issues**: Report bugs or request features on `GitHub Issues <https://github.com/Asher-1/ACloudViewer/issues>`_
- **Discussions**: Ask questions on `GitHub Discussions <https://github.com/Asher-1/ACloudViewer/discussions>`_
- **Documentation**: Check the `documentation </documentation/>`_ for guides and API references

Types of Contributions
----------------------

Bug Reports
^^^^^^^^^^^

When reporting bugs, please include:

- ACloudViewer version
- Operating system and version
- Steps to reproduce
- Expected vs actual behavior
- Error messages or logs

Feature Requests
^^^^^^^^^^^^^^^^

For feature requests:

- Clearly describe the proposed feature
- Explain use cases and benefits
- Provide examples if possible

Documentation Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

Documentation contributions are highly valued:

- Fix typos or unclear explanations
- Add examples and tutorials
- Improve API documentation
- Translate documentation

Code Contributions
^^^^^^^^^^^^^^^^^^

Areas where contributions are welcome:

- **Core Library**: Point cloud processing, mesh algorithms, I/O support
- **Visualization**: Rendering improvements, GUI enhancements
- **ML/DL**: Machine learning integrations, neural network support
- **Plugins**: New plugin development
- **Python Bindings**: Python API improvements
- **Performance**: Optimization and GPU acceleration

Recognition
-----------

Contributors will be acknowledged in:

- ``CHANGELOG.md``
- GitHub contributors page
- Release notes

Thank you for contributing to ACloudViewer! 🎉
