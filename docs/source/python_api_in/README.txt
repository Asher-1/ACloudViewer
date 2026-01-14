Custom Python API RST Files Override Directory
==============================================

This directory contains manually curated RST files for Python API documentation
that override the auto-generated versions. These files provide better organization
and structure for complex modules.

Files in this directory follow the pattern: cloudViewer.<module>.rst

When running make_docs.py, these files will be used instead of auto-generated ones
for the specified modules, allowing for:
  - Custom grouping of classes and functions
  - Better documentation structure
  - Manual control over what gets documented
  - Improved navigation and readability

Based on Open3D's documentation system pattern.

Current custom modules:
  - cloudViewer.visualization.rst  (Visualization classes and functions)
  - cloudViewer.io.rpc.rst         (Remote procedure call functionality)
