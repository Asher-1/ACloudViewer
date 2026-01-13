# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# Register 'ipython3' as an alias for 'python3' in Pygments
# This fixes "WARNING: Pygments lexer name 'ipython3' is not known"
from pygments.lexers import get_lexer_by_name
from pygments.lexers import Python3Lexer
from sphinx.highlighting import lexers
lexers['ipython3'] = Python3Lexer()

# -- Project information -----------------------------------------------------
project = 'ACloudViewer'
copyright = f'{datetime.now().year}, ACloudViewer Team'
author = 'ACloudViewer Team'
release = '3.9.3'  # Update this with actual version
version = '3.9'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',  # For API summary tables
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx_tabs.tabs',
    'sphinx_copybutton',
    'myst_parser',
    'nbsphinx',  # ✅ Re-enabled with pyenv environment
    'breathe',  # For C++ API documentation
]

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True  # Turn on autosummary
autosummary_imported_members = True

# -- Autodoc configuration (mock imports) ------------------------------------
# Mock dependencies to allow documentation build without actual modules
# 
# 1. Current: Mock cloudViewer (documentation builds, but API docs are basic)
# 2. Future: Build Python module, remove mock, use real autodoc (detailed API docs)
#
# To build Python module and remove mock:
#   1. Install all dependencies (Qt5WebSockets, PCL, etc.)
#   2. cd /path/to/ACloudViewer/build_app
#   3. cmake .. -DBUILD_PYTHON_MODULE=ON -DCMAKE_PREFIX_PATH=/opt/qt515/lib/cmake
#   4. make cloudViewer_pybind -j$(nproc)
#   5. Verify: cd lib/python_package && python3 -c "import cloudViewer; print(cloudViewer.__version__)"
#   6. Then in conf.py:
#      - Remove 'cloudViewer' from autodoc_mock_imports below
#      - Uncomment the sys.path.insert lines at the end of this section
#
# For a guided build process, use: docs/build_python_module.sh
#
autodoc_mock_imports = [
    # Only mock external dependencies (NOT cloudViewer - we use the real module now!)
    'numpy',
    'scipy',
    'matplotlib',
    'PIL',
    'open3d',
    'torch',
    'tensorflow',
    'cv2',
    'sklearn',
    'trimesh',
]

# ✅ USE REAL PYTHON MODULE
current_file_dir = os.path.dirname(os.path.realpath(__file__))
python_pkg_path = os.path.join(current_file_dir, "..", "..", "build_app", "lib", "Release", "Python", "cuda")

if os.path.exists(python_pkg_path):
    sys.path.insert(0, python_pkg_path)
    print(f"✓ Using REAL cloudViewer module from: {python_pkg_path}")
    
    # Import and create alias
    try:
        import pybind as cloudViewer
        sys.modules['cloudViewer'] = cloudViewer
        print(f"✓ cloudViewer module loaded: {', '.join([x for x in dir(cloudViewer) if not x.startswith('_')])}")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        autodoc_mock_imports.append('cloudViewer')
else:
    print(f"✗ Module not found at: {python_pkg_path}")
    autodoc_mock_imports.append('cloudViewer')

# Don't execute doctest code
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# This makes navigation show "Blob" instead of "cloudViewer.core.Blob"
add_module_names = False

# Don't warn about mocked members
autodoc_warningiserror = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'  # Modern, beautiful theme like Open3D

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2196F3",
        "color-brand-content": "#2196F3",
    },
    "dark_css_variables": {
        "color-brand-primary": "#42A5F5",
        "color-brand-content": "#42A5F5",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

html_title = f"ACloudViewer {release} Documentation"
html_logo = "../images/ACloudViewer_logo_horizontal.png"
html_favicon = "../images/ACloudViewer.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS
html_css_files = [
    'custom.css',
]

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
}

latex_documents = [
    (master_doc, 'ACloudViewer.tex', 'ACloudViewer Documentation',
     'ACloudViewer Team', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (master_doc, 'acloudviewer', 'ACloudViewer Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (master_doc, 'ACloudViewer', 'ACloudViewer Documentation',
     author, 'ACloudViewer', 'Point cloud and 3D data processing library.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for Breathe extension (C++ API) ---------------------------------
# Path is relative to docs/source/ when Sphinx runs
# Try multiple possible locations for Doxygen XML output
import os
_possible_doxygen_paths = [
    '../../build/docs/doxygen/xml',  # CMake build in build_app/
    '../../build_app/docs/doxygen/xml',  # CMake build in build_app/
    '../../build/docs/doxygen/xml',       # CMake build in build/
    '../doxygen/xml',                     # Standalone make_docs.py
]
_doxygen_xml_path = '../doxygen/xml'  # Default fallback
for path in _possible_doxygen_paths:
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if os.path.exists(abs_path):
        _doxygen_xml_path = path
        break

breathe_projects = {
    "ACloudViewer": _doxygen_xml_path
}
breathe_default_project = "ACloudViewer"
breathe_default_members = ('members', 'undoc-members')

# Suppress Breathe warnings and errors for malformed XML
breathe_debug_trace_directives = False
breathe_debug_trace_doxygen_ids = False
breathe_debug_trace_qualification = False

# Configure Breathe to be more lenient with XML parsing errors
import logging
logging.getLogger('breathe').setLevel(logging.ERROR)

# -- Options for nbsphinx (Jupyter notebooks) --------------------------------
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_timeout = 300

# Map 'ipython3' to 'python3' to suppress Pygments lexer warnings
# This fixes "WARNING: Pygments lexer name 'ipython3' is not known"
nbsphinx_codecell_lexer = 'python3'

# Prolog for all notebooks
nbsphinx_prolog = """
.. note::
   This tutorial is generated from a Jupyter notebook that can be downloaded and run interactively.
   
   📓 `Download notebook <https://github.com/Asher-1/ACloudViewer/tree/main/docs/jupyter>`_
"""

# -- Options for MyST parser -------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# -- Options for copybutton --------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Custom configuration ----------------------------------------------------

# GitHub repository
html_context = {
    "display_github": True,
    "github_user": "Asher-1",
    "github_repo": "ACloudViewer",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

# Version switching (for future multi-version support)
html_js_files = [
    'version_switch.js',
]

# Pygments style
pygments_style = 'sphinx'
pygments_dark_style = 'monokai'

# Suppress warnings (following Open3D's approach)
suppress_warnings = [
    'myst.header',
    'autodoc',
    'autodoc.import_object',
    'nbsphinx',
    'nbsphinx.localfile',
    'nbsphinx.gallery',
    'nbsphinx.thumbnail',
    'misc.highlighting_failure',  # Suppress Pygments lexer warnings
    'ref.duplicate',  # Suppress duplicate object description warnings
    'toc.not_readable',  # Suppress document not in toctree warnings
    'ref.python',  # Suppress Python reference warnings
    'ref.ref',  # Suppress general reference warnings
]

# Configure Sphinx to be less verbose about warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='nbsphinx')
warnings.filterwarnings('ignore', category=FutureWarning)

# Add Python path for autodoc
sys.path.insert(0, os.path.abspath('../../python'))

# Read the Docs integration
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    # Don't require C++ build on Read the Docs
    breathe_projects = {}

