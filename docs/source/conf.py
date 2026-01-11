# Configuration file for the Sphinx documentation builder.
# Based on Open3D's documentation system
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# -- Project information -----------------------------------------------------
project = 'ACloudViewer'
copyright = f'{datetime.now().year}, ACloudViewer Team'
author = 'ACloudViewer Team'
release = '3.9.3'  # Update this with actual version
version = '3.9'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
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
    'nbsphinx',
    'breathe',  # For C++ API documentation
]

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
breathe_projects = {
    "ACloudViewer": "../doxygen/xml"
}
breathe_default_project = "ACloudViewer"
breathe_default_members = ('members', 'undoc-members')

# -- Options for nbsphinx (Jupyter notebooks) --------------------------------
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_timeout = 300

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

# Suppress warnings
suppress_warnings = ['myst.header']

# Add Python path for autodoc
sys.path.insert(0, os.path.abspath('../../python'))

# Read the Docs integration
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    # Don't require C++ build on Read the Docs
    breathe_projects = {}

