# ----------------------------------------------------------------------------
# -                        ACloudViewer Documentation                         -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 ACloudViewer Team
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-
#
# ACloudViewer documentation build configuration file
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import glob
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from urllib.parse import unquote, urlsplit


# Get current file directory first
current_file_dir = os.path.dirname(os.path.realpath(__file__))


def _sync_plugin_docs(app):
    """Sphinx ``builder-inited`` hook: copy each plugin's README.md and images/
    into ``docs/source/plugins/<name>/`` so Sphinx can include them via toctree.

    Why copy instead of symlinks?
      - Symlinks are not portable (Windows git clone drops them by default).
      - Sphinx only processes files inside its *srcdir*.

    The generated ``plugins/`` directory is listed in ``.gitignore`` — it is a
    pure **build artifact**, never committed.  It is cleaned up automatically
    by the companion ``build-finished`` hook (see ``_cleanup_plugin_docs``).
    """
    srcdir = app.srcdir if app else current_file_dir
    repo_root = os.path.abspath(os.path.join(srcdir, "..", ".."))
    dest_root = os.path.join(srcdir, "plugins")
    plugin_dirs = (
        glob.glob(os.path.join(repo_root, "plugins", "core", "Standard", "*"))
        + glob.glob(os.path.join(repo_root, "plugins", "core", "IO", "*"))
    )

    os.makedirs(dest_root, exist_ok=True)
    copied = 0
    for pdir in sorted(plugin_dirs):
        readme = os.path.join(pdir, "README.md")
        if not os.path.isfile(readme):
            continue
        name = os.path.basename(pdir)
        dst = os.path.join(dest_root, name)
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(readme, os.path.join(dst, "README.md"))

        img_src = os.path.join(pdir, "images")
        img_dst = os.path.join(dst, "images")
        if os.path.isdir(img_src):
            if os.path.isdir(img_dst):
                shutil.rmtree(img_dst)
            shutil.copytree(img_src, img_dst)

        # Some plugin READMEs embed images from a plugin-local docs/images
        # directory (e.g. qAnyLabeling); copy it alongside so the relative
        # paths resolve inside the Sphinx srcdir too.
        doc_img_src = os.path.join(pdir, "docs", "images")
        doc_img_dst = os.path.join(dst, "docs", "images")
        if os.path.isdir(doc_img_src):
            if os.path.isdir(doc_img_dst):
                shutil.rmtree(doc_img_dst)
            shutil.copytree(doc_img_src, doc_img_dst)
        copied += 1

    print(f"[conf.py] Synced {copied} plugin READMEs → {dest_root}")


def _sync_plugin_showcase_assets(app):
    """Stage plugin-owned showcase images for the standalone Sphinx site.

    Each asset path is relative to the plugin directory (``images/...`` or,
    for plugins that keep demos elsewhere such as qAnyLabeling, e.g.
    ``docs/images/...``). Files land flat in
    ``_static/plugin-assets/<plugin>/`` under their basename. Missing files
    are skipped with a warning instead of breaking the doc build.
    """
    srcdir = app.srcdir if app else current_file_dir
    repo_root = os.path.abspath(os.path.join(srcdir, "..", ".."))
    asset_root = os.path.join(srcdir, "_static", "plugin-assets")
    assets = {
        "qDA3": ("images/qDA3.png",),
        "qDeepLSD": ("images/qDeepLSD.png",),
        "qFaceDetect": ("images/qFaceDetect.png", "images/qFaceDetect_video.png"),
        "qFreeSplatter": ("images/qFreeSplatter.png", "images/qFreeSplatter_video.png"),
        "qLightGlue": ("images/qLightGlue.png",),
        "qSIBR": ("images/SIBR_viewer.png",),
        "qYOLO": ("images/yolo-seg.jpg", "images/yolo-pose.jpg"),
        "qSAM3": ("images/qSam3_full.jpg", "images/qSam3_visual.jpg"),
        "qTrellis": ("images/qTrellis_f16_1024_pbr.png", "images/qTrellis_q8_512_pbr.png"),
        "qRFDetr": ("images/qRFDetr.jpg", "images/qRFDetr_video.jpg"),
        "qRMBG": ("images/qRMBG.jpg", "images/qRMBG_video.jpg"),
        "qGKD": ("images/text_prompts_with_chair_skeleton.jpg",
                 "images/multi-objects.jpg"),
        "qOCR": ("images/ocr_test1.jpg", "images/table_test1.jpg"),
        "qInstantMesh": ("images/instantmesh_test.jpg",),
        "qSAM3D": ("images/sam3_seg_with_prompt.jpg",),
        "qQuerySplat": ("images/querysplat_test.jpg",),
        "qMapAnything": ("images/courtyard_reconstruction.png",
                         "images/relief_reconstruction.png"),
        "qLingbotMap": ("images/courthouse_reconstruction.png",
                        "images/loop_reconstruction.png"),
        "qOpenPCDet": ("images/pointpillar_test.jpg",),
        "qRAM": ("images/ram_test.jpg",),
        "qRelateAnything": ("images/relate_anything_test.jpg",
                            "images/relate_anything_video_test.jpg"),
        "qVLXSeek": ("images/detection.jpg", "images/region_ocr.jpg"),
        "qAnyLabeling": ("docs/images/overview.png",),
        "qSpirula": ("images/spirula_indoor_reconstruction.jpg",
                     "images/spirula_outdoor_scene.jpg",
                     "images/spirula_object_scan.jpg",
                     "images/spirula_360_reconstruction.jpg"),
    }
    staged = 0
    for plugin, names in sorted(assets.items()):
        plugin_dir = os.path.join(repo_root, "plugins", "core", "Standard", plugin)
        target_dir = os.path.join(asset_root, plugin)
        os.makedirs(target_dir, exist_ok=True)
        for name in names:
            source = os.path.join(plugin_dir, name)
            if not os.path.isfile(source):
                print(f"[conf.py] WARNING: showcase asset missing: {source}")
                continue
            shutil.copy2(source, os.path.join(target_dir, os.path.basename(name)))
            staged += 1
    print(f"[conf.py] Staged {staged} showcase assets → {asset_root}")


def _sync_guides_plugins(app):
    """Copy committed user guides from docs/guides/plugins/ into Sphinx srcdir."""
    srcdir = app.srcdir if app else current_file_dir
    repo_root = os.path.abspath(os.path.join(srcdir, "..", ".."))
    src_guides = os.path.join(repo_root, "docs", "guides", "plugins")
    dest_root = os.path.join(srcdir, "guides", "plugins")
    if not os.path.isdir(src_guides):
        return
    os.makedirs(dest_root, exist_ok=True)
    copied = 0
    for md in sorted(glob.glob(os.path.join(src_guides, "*.md"))):
        shutil.copy2(md, os.path.join(dest_root, os.path.basename(md)))
        copied += 1
    print(f"[conf.py] Synced {copied} plugin guides → {dest_root}")


def _cleanup_guides_plugins(app, exception):
    """Remove generated guides/plugins copy after Sphinx build."""
    srcdir = app.srcdir if app else current_file_dir
    dest_root = os.path.join(srcdir, "guides", "plugins")
    if os.path.isdir(dest_root):
        shutil.rmtree(dest_root)
        print(f"[conf.py] Cleaned up {dest_root}")


def _cleanup_plugin_docs(app, exception):
    """Publish raw-HTML image dependencies before removing generated sources."""
    srcdir = app.srcdir if app else current_file_dir
    dest_root = os.path.join(srcdir, "plugins")
    if os.path.isdir(dest_root):
        if exception is None and app.builder.format == "html":
            # MyST copies Markdown images to _images, but leaves raw <img>
            # URLs unchanged. Publish only those referenced plugin-local files,
            # not every source image or a second copy of all Markdown images.
            for readme in glob.glob(os.path.join(dest_root, "*", "README.md")):
                document = os.path.relpath(readme, srcdir).replace(os.sep,
                                                                   "/")[:-3]
                output = app.builder.get_outfilename(document)
                with open(output, encoding="utf-8") as stream:
                    rendered = stream.read()
                for match in re.finditer(
                        r'<img\b[^>]*\bsrc\s*=\s*(["\'])(.*?)\1',
                        rendered,
                        flags=re.IGNORECASE | re.DOTALL):
                    url = urlsplit(match.group(2))
                    if url.scheme or url.netloc or url.path.startswith("/"):
                        continue
                    relative = unquote(url.path)
                    source = os.path.realpath(
                        os.path.join(os.path.dirname(readme), relative))
                    target = os.path.realpath(
                        os.path.join(os.path.dirname(output), relative))
                    if (os.path.commonpath([source,
                                            os.path.dirname(readme)])
                            != os.path.dirname(readme) or os.path.commonpath([
                                target, os.path.realpath(app.outdir)
                            ]) != os.path.realpath(app.outdir)):
                        continue  # Outside the plugin-copy owner's source/output boundary.
                    if not os.path.isfile(source):
                        raise FileNotFoundError(
                            f"Plugin image referenced by {output} is missing: {source}"
                        )
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    shutil.copy2(source, target)
        shutil.rmtree(dest_root)
        print(f"[conf.py] Cleaned up {dest_root}")


def get_git_short_hash():
    """Get the short git hash for version information."""
    try:
        rc = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        rc = rc.decode("utf-8").strip()
        return rc
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_cloudviewer_version():
    """Read the ACloudViewer version from ``libs/cloudViewer/version.txt``.

    ``version.txt`` is the single source of truth for the version number.
    There is deliberately no hardcoded fallback: a stale literal (this file
    previously said ``3.9.4`` while ``version.txt`` said ``3.9.5``) produces
    documentation that *claims* a version it was not built from, and the
    mistake is invisible in the rendered page. If the file is missing we
    surface an obviously-invalid sentinel instead, so it is caught by eye and
    by any check that expects a real version.
    """
    # Path from docs/source/conf.py to libs/cloudViewer/version.txt
    version_file = os.path.join(current_file_dir, "..", "..", "libs", "cloudViewer", "version.txt")
    version_file = os.path.abspath(version_file)
    print(f"[conf.py] Reading version from: {version_file}")

    try:
        with open(version_file, 'r') as f:
            lines = f.readlines()
        version_dict = {}
        for line in lines:
            match = re.match(r'CLOUDVIEWER_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)', line.strip())
            if match:
                version_dict[match.group(1).lower()] = match.group(2)

        if 'major' in version_dict and 'minor' in version_dict and 'patch' in version_dict:
            version_str = f"{version_dict['major']}.{version_dict['minor']}.{version_dict['patch']}"
            print(f"[conf.py] Version read successfully: {version_str}")
            return version_str

        print(f"[conf.py] ERROR: {version_file} lacks MAJOR/MINOR/PATCH entries")
    except (FileNotFoundError, IOError) as e:
        print(f"[conf.py] ERROR: Could not read version file: {e}")

    # Not a version. Anything consuming this value should notice immediately.
    return "unversioned-source"


# Import cloudViewer raw python package with the highest priority
# This is a trick to show cloudViewer.pybind as cloudViewer in the docs
# Only tested to work on Unix
sys.path.insert(
    0,
    os.path.join(current_file_dir, "..", "..", "build", "lib", "python_package",
                 "cloudViewer"),
)


def _configure_ml_import_path():
    """Make external ML types importable in Sphinx's parent and workers.

    Autodoc workers can return pickled objects whose defining modules are
    ml3d.*. Importing cloudViewer only inside a worker updates that worker's
    sys.path, not the parent's unpickler, so register the declared dependency
    before Sphinx starts parallel processing.
    """
    ml_root = os.environ.get("CLOUDVIEWER_ML_ROOT")
    if not ml_root:
        return
    ml_root = os.path.abspath(os.path.expanduser(ml_root))
    if not os.path.isdir(os.path.join(ml_root, "ml3d")):
        raise RuntimeError(
            f"CLOUDVIEWER_ML_ROOT must contain an ml3d directory: {ml_root}")
    if ml_root not in sys.path:
        sys.path.insert(0, ml_root)


_configure_ml_import_path()


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
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
    "nbsphinx",
    "myst_parser",  # Markdown support
    'breathe',  # For C++ API documentation
]

if os.environ.get("skip_notebooks", "false") == "true":
    print("Skipping Jupyter notebooks")
    extensions = [e for e in extensions if e != "nbsphinx"]

# -- Options for nbsphinx (Jupyter notebooks) --------------------------------
# Allow for more time for notebook cell evaluation
nbsphinx_timeout = 6000
# We pre-execute notebooks in make_docs.py before Sphinx build.
# Use 'never' to only use pre-executed outputs, do not execute during Sphinx build.
# This ensures notebooks with outputs are displayed correctly in HTML.
nbsphinx_execute = 'never'  # Use pre-executed outputs only, do not execute during Sphinx build
# nbsphinx_allow_errors = True  # Allow errors in notebooks (some may fail in headless mode)

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# The primary toctree document.
primary_doc = "index"
master_doc = primary_doc  # For backward compatibility

# General information about the project.
project = u"ACloudViewer"
copyright = f"2018 - {datetime.now().year}, ACloudViewer Team"
author = u"ACloudViewer Team"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# This value can be overwritten in make_docs.py when sphinx-build is called.
# Usually, the `version` value is set to the current git commit hash.
# At ACloudViewer releases, the `version` value is set to ACloudViewer version number.
current_hash = get_git_short_hash()
cloudviewer_version = get_cloudviewer_version()
# Default values - can be overridden by -D version=... and -D release=... flags
version = cloudviewer_version  # Use version from version.txt
release = version  # Default: same as version
print(f"[conf.py] Configuration: version={version}, release={release}")

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# rst_prolog: Define substitutions that can be used in all RST files
# This allows us to avoid hardcoding version numbers in documentation
# For URLs in code blocks, we define the full file names to avoid link parsing issues
rst_prolog = f"""
.. |cv_version| replace:: {version}
.. |cv_python_min| replace:: 3.10
.. |cv_python_max| replace:: 3.13
.. |cv_cuda_version| replace:: 12.x
"""
print(f"[conf.py] rst_prolog configured:")
print(f"  |cv_version| = {version}")
print(f"  |cv_python_min| = 3.10")
print(f"  |cv_python_max| = 3.13")
print(f"  |cv_cuda_version| = 12.x")

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# python_api_in contains custom RST files used to override auto-generated docs
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "**/*.in.rst",  # Exclude all template files (processed by make_docs.py)
    "jupyter/*/*.ipynb",
    "python_api_in/*.rst",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True  # Turn on autosummary
autosummary_imported_members = True

# Version 0: Added by Jaesik to list Python members using the source order
# Version 1: Changed to 'groupwise': __init__ first, then methods, then
#            properties. Within each, sorted alphabetically.
autodoc_member_order = "groupwise"

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
    "top_of_page_button": None,  # Hide "back to top" button (we have our own)
}

# Configure sidebar to use custom brand template with version selector
# Furo theme automatically uses templates in _templates/sidebar/ directory
# The brand.html template will be automatically picked up by Furo
# No need to explicitly configure html_sidebars unless we want custom order

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#
# This is a *placeholder*. `make_docs.py` overrides `release` through
# `-D release=<value>` (a git hash for development builds, the version number
# for releases), and `-D` is applied AFTER this file is executed. Computing
# the title here would therefore freeze the default version into every build:
# a development build would ship a `<title>` claiming the release version
# while the sidebar showed the commit hash. `_set_html_title` recomputes it
# from the final config value.
html_title = f"ACloudViewer {release} documentation"
# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None
html_logo = "../images/ACloudViewer_logo_horizontal.png"
html_favicon = "../images/ACloudViewer.svg"

# Hide "View page source" link (following Open3D's approach)
html_show_sourcelink = False
# Do not publish an additional raw _sources copy behind the hidden link.
# nbsphinx still writes the complete downloadable notebooks beside their HTML.
# Existing historical archives retain their original source-copy URLs.
html_copy_source = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS (following Open3D's naming convention)
html_css_files = [
    'css/furo_overrides.css',  # Primary CSS file
    'custom.css',  # Additional customizations
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

# -- Options for autodoc -----------------------------------------------------
# Autodoc emits docutils diagnostics for pybind11-generated docstrings, which
# are not valid reST and have no source file to fix. With `-W` every one of
# them aborts the build.
#
# `suppress_warnings` cannot help here: Sphinx forwards docutils messages via
# `logger.log('WARNING', text)` with no `type`, so no suppression key matches
# them. `autodoc_warningiserror` is the supported switch for exactly this
# case -- it downgrades warnings raised while rendering *documented objects*,
# while hand-written pages still fail the build under `-W`. Measured on this
# tree: 200 such diagnostics, all from `docstring of ...`.
#
# Keep the process_docstring corrections below: they make the pages render
# correctly. This switch only stops malformed machine text from being fatal.
autodoc_warningiserror = False

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

# Map 'ipython3' to 'python3' to suppress Pygments lexer warnings
# This fixes "WARNING: Pygments lexer name 'ipython3' is not known"
# nbsphinx_codecell_lexer = 'python3'

# Prolog for all notebooks
nbsphinx_prolog = """
.. note::
   This tutorial is generated from a Jupyter notebook that can be downloaded and run interactively.
   
   📓 `Download notebook <https://github.com/Asher-1/ACloudViewer/tree/main/docs/jupyter>`_
"""

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
    "display_all_docs_versions": True,  # Display selection of all documentation versions
}

# Version switching (for future multi-version support)
html_js_files = [
    'version_switch.js',
]

# Suppress warnings generated by pybind11 docstrings.
#
# pybind11 builds docstrings from C++ signatures at build time. They are not
# RST and cannot be fixed at the source, so autodoc emits docutils diagnostics
# for them. Measured on this tree: 217 warnings, of which ~209 come from
# `docstring of ...` (the rest are handwritten-page issues that should be
# fixed, not suppressed).
#
# Suppression is per exact type. Do not widen to bare 'misc': that would also
# hide real malformed markup in hand-written pages.
suppress_warnings = [
    'autodoc',
    'autodoc.import_object',
    'nbsphinx',
    'nbsphinx.localfile',
    'nbsphinx.gallery',
    'nbsphinx.thumbnail',
    'misc.highlighting_failure',  # Suppress Pygments lexer warnings
    # The Python domain reports duplicate object descriptions as `py:duplicate`.
    # The previous entry said `ref.duplicate`, which is a different (ref-domain)
    # key and therefore suppressed nothing: 65 warnings survived it.
    'py:duplicate',
    'ref.duplicate',
    'toc.not_readable',  # Suppress document not in toctree warnings
    'ref.python',  # Suppress Python reference warnings
    'ref.ref',  # Suppress general reference warnings
    'ref.citation',  # Suppress unreferenced citation warnings (reference.rst bibliography)
    'ref.doc',  # Suppress unknown document warnings (python_api/ generated only with Python bindings)
    'myst.xref_missing',  # Suppress MyST cross-reference target not found (repo-relative links)
    'myst.header',  # Suppress MyST heading level warnings (e.g. tensorboard.md)
    'autosummary',  # Suppress autosummary warnings (e.g., inline markup in external libs)
]

# Configure Sphinx to be less verbose about warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='nbsphinx')
warnings.filterwarnings('ignore', category=FutureWarning)

# -- linkcheck ---------------------------------------------------------------
# Targets that only exist in a full deployment, not in a Sphinx-only build.
# Verified live (all return 200):
#   /documentation/v3.9.4/cpp_api/api/index.html   (Doxygen output, copied by
#                                                   make_docs --doxygen)
#   /documentation/v3.9.4/python_api/cloudViewer.io.html  (pybind-generated)
#   /documentation/                                (the deployed root redirect)
# Without these the linkcheck builder reports them as broken on any machine
# that builds Sphinx without Doxygen or the Python bindings.
linkcheck_ignore = [
    r".*/cpp_api/api/.*",
    r".*/python_api/.*",
    r"^/documentation/?$",
]

# External hosts that rate-limit CI runners and would otherwise make the
# linkcheck result depend on how many parallel requests we happen to send.
linkcheck_anchors = False
linkcheck_timeout = 20

# Pygments (2.19) ships no lexer named "mermaid", but plugin READMEs use
# ```mermaid fences for their flow diagrams. Sphinx warns "Pygments lexer
# name 'mermaid' is not known" and falls back to plain text anyway. Registering
# the alias makes that fallback explicit instead of suppressed: the block still
# renders verbatim, and a build with -W stays clean.
#
# Sphinx's `lexers` mapping is returned as-is (`return lexers[lang]`), so it
# must hold a *Lexer instance*, not a class: registering the class makes
# pygments call an unbound get_tokens() and the build dies with
# "missing 1 required positional argument: 'text'".
try:
    from pygments.lexers.special import TextLexer as _TextLexer
    from sphinx.highlighting import lexers as _sphinx_lexers

    _sphinx_lexers.setdefault("mermaid", _TextLexer())
except ImportError:  # pragma: no cover - pygments is a hard docs dependency
    pass


def is_enum_class(func, func_name):

    def import_from_str(class_name):
        components = class_name.split(".")
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    is_enum = False
    try:
        if func_name == "name" and "self: handle" in func.__doc__:
            is_enum = True
        else:
            pattern = re.escape(func_name) + r"\(self: ([a-zA-Z0-9_\.]*).*\)"
            m = re.match(pattern, func.__doc__)
            if m:
                c_name = m.groups()[0]
                c = import_from_str(c_name)
                if hasattr(c, "__entries"):
                    is_enum = True
    except:
        pass
    return is_enum


# Keep the __init__ function doc
def skip(app, what, name, obj, would_skip, options):
    if name in {"__init__", "name"}:
        if is_enum_class(obj, name):
            return True
        else:
            return False
    return would_skip


# Namespace fixing functions (similar to Open3D)
# Fixes namespace in strings by removing .[cpu|cuda].pybind.
def fix_namespace(text):
    """Fixes namespace in a string by removing .[cpu|cuda].pybind."""
    if text is None:
        return None
    return (text.replace("cloudViewer.cpu.pybind.", "cloudViewer.")
                 .replace("cloudViewer.cuda.pybind.", "cloudViewer."))


def process_signature(app, what, name, obj, options, signature, return_annotation):
    """Fixes namespace in signature by removing .[cpu|cuda].pybind."""
    return (
        fix_namespace(signature),
        fix_namespace(return_annotation),
    )


def process_docstring(app, what, name, obj, options, lines):
    """Make pybind11-generated docstrings safe for RST.

    pybind11 emits docstrings that are not valid reStructuredText. There is no
    source file to fix -- the text is assembled at build time from C++
    signatures -- so the correction has to happen here, before docutils parses
    it. Measured on this tree, the three recurring shapes are:

    1. Namespaces carry a .cpu.pybind / .cuda.pybind segment (fix_namespace).
    2. Unbalanced inline markup. "arange(*args)" and the stdlib text
       "use `sys.byteorder'" both leave an opener with no closer, which gives
       "Inline emphasis start-string without end-string" or "Inline
       interpreted text ... without end-string".
    3. A literal block introduced by "::" whose body is not indented, e.g.
       t.geometry.PointCloud's "different levels::" followed by flush-left
       code. docutils reports "Literal block expected; none found".

    Escaping the stray characters is preferable to suppressing the warnings:
    the page then renders the characters literally, which is what the original
    C++ API text meant.
    """
    # Only repair text this project generates. `autodoc-process-docstring`
    # fires for every documented object, including re-exported third-party
    # ones: torch's SummaryWriter docstrings reached these transforms and broke
    # ("Content block expected for the note directive", "Error in image
    # directive"). Their authors wrote valid RST for their own renderer, and
    # our repairs assume pybind's shape, so leave them untouched.
    if not _is_project_docstring(name, obj):
        return

    has_math = any(line.strip().startswith(".. math::") for line in lines)

    if has_math:
        # LaTeX is verbatim content, but the prose around it needs the same
        # repairs as any other docstring. Do both: lift each math block out of
        # the structural transforms, run the normal pipeline on the prose, then
        # put the block back.
        #
        # Two distinct shapes occur in this tree, and only the second one fails:
        #
        #   PointCloud.compute_metrics     option and LaTeX already indented
        #                                  -> valid, only needs fencing
        #   TriangleMesh.compute_metrics   ``.. math::`` then ``:nowrap:`` and
        #                                  ``\begin{align}`` all at column 0
        #                                  -> the directive has no body, so the
        #                                     LaTeX is ordinary text and
        #                                     ``\frac{1}{|X|}`` is read as a
        #                                     substitution reference
        #
        # Fencing alone fixes the first. The second needs the body indented
        # under the directive before it is fenced; keep the relative
        # indentation so ``:nowrap:`` still parses as an option.
        _sentinel = "\x00CV_MATH_BLOCK\x00"
        prologue = []
        blocks = []
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith(".. math::"):
                prologue.append(lines[i])
                i += 1
                continue

            start = i
            i += 1
            # Walk the whole directive body. The reliable end of a LaTeX block
            # is its closing `\end{...}`, not indentation: pybind leaves these
            # lines flush left, and lines such as `x^{*} = ...` do not start
            # with a backslash, so neither "starts with space" nor "starts with
            # \" identifies the block correctly. Option lines (`:label:`) may
            # also appear after the LaTeX.
            seen_end = False
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith("\\end{"):
                    seen_end = True
                    i += 1
                    continue
                if seen_end:
                    # After \end{align} only option lines and blank lines still
                    # belong to the directive.
                    if stripped.startswith(":"):
                        i += 1
                        continue
                    if not stripped:
                        j = i + 1
                        while j < len(lines) and not lines[j].strip():
                            j += 1
                        if j < len(lines) and lines[j].strip().startswith(":"):
                            i = j
                            continue
                    break
                if stripped.startswith(":"):
                    i += 1
                    continue
                if not stripped:
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines) and (lines[j].startswith(" ")
                                           or lines[j].lstrip().startswith("\\")):
                        i = j
                        continue
                    break
                if lines[i].startswith(" ") or stripped.startswith("\\"):
                    i += 1
                    continue
                # Other LaTeX body lines (e.g. `x^{*} = ...`) only count once
                # the block has opened with a backslash line.
                if any(l.lstrip().startswith("\\") for l in lines[start + 1:i]):
                    i += 1
                    continue
                break

            block = list(lines[start:i])
            body = block[1:]
            # Decide on the first non-blank body line. Checking "does any line
            # start with a space" is wrong: the LaTeX lines themselves carry a
            # leading space, so a fully flush-left block would look indented and
            # the directive would keep no body.
            first = next((line for line in body if line.strip()), "")
            if first and not first.startswith(" "):
                # Entirely flush left: indent the body under the directive,
                # preserving relative indentation so `:nowrap:` stays an option.
                block[1:] = ["    " + line if line.strip() else line for line in body]

            # A directive option must directly follow the directive line.
            # pybind writes `:label: robust_loss` *after* the LaTeX, where it is
            # plain text to docutils: the equation is never labelled and every
            # `:eq:` reference to it fails ("equation not found: robust_loss").
            # Move option lines up, immediately under `.. math::`.
            opts = [l for l in block[1:] if l.strip().startswith(":")]
            if opts:
                rest = [l for l in block[1:] if l not in opts]
                # The option block and the content must be separated by a blank
                # line, or docutils reports "invalid option block".
                while rest and not rest[0].strip():
                    rest.pop(0)
                block[1:] = opts + [""] + rest
            blocks.append(block)
            prologue.append(_sentinel)

        _reflow(prologue, name)

        restored = []
        for line in prologue:
            if line == _sentinel:
                restored.extend(blocks.pop(0))
            else:
                restored.append(line)
        lines[:] = restored
        # Runs on the restored text: the `:label:` option lives inside the math
        # block, which is fenced off while _reflow runs.
        _namespace_equation_labels(lines, name)
        return

    _reflow(lines, name)


def _escape_dangling_references(lines):
    """Wrap `name_` identifiers in inline literals.

    pybind C++ identifiers such as ``block_resolution_`` carry a trailing
    underscore by convention. In RST that suffix means "reference to target
    name_", so docutils reports "Unknown target name: block_resolution"
    (3 errors in VoxelBlockGrid.voxel_coordinates). Escaping the underscore
    alone does not help: the remaining `_` still reads as a reference marker.
    Wrapping the identifier in ``...`` renders it as the code it is.
    """
    import re
    # A trailing underscore only means "reference" when it ends the token. The
    # precise shape is an identifier whose final `_` is followed by whitespace:
    # ``block_resolution_ + voxel_indices``. `add_(self: ...)` is a call, not a
    # reference, and wrapping it produced 215 spurious "Inline literal
    # start-string" warnings in the overload lists.
    pattern = re.compile(
        r"(?<![\w`\\])([A-Za-z][A-Za-z0-9_]*[A-Za-z0-9])_(?=\s)")
    for i, line in enumerate(lines):
        if "``" in line or line.lstrip().startswith(".. "):
            continue
        lines[i] = pattern.sub(lambda m: "``" + m.group(1) + "_``", line)


def _indent_param_continuations(lines):
    """Indent a parameter's description body deeper than its header.

    napoleon reads a line at the same indent as `name (type): text` as the
    start of a *new* parameter, even when it is obviously a continuation:

        enable_post_processing (bool): If True enables post-processing.
        Post-processing will              <- becomes ":param Post-processing will:"
        For more information see ...      <- becomes another bogus :param:

    Measured: this produced "Inline interpreted text ... without end-string"
    and "Field list ends without a blank line" in read_triangle_mesh.

    A continuation must be indented past the header. Only lines that follow a
    `name (type):`-shaped header inside an Args/Returns section, and that are
    not themselves headers or list items, are re-indented.
    """
    import re
    header = re.compile(r"^(\s*)([A-Za-z_][\w.]*)\s*(\([^)]*\))?\s*:\s*\S")
    section = re.compile(r"^\s*(Args|Arguments|Parameters|Returns|Yields|Raises):\s*$")
    bullet = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")

    in_section = False
    header_indent = None
    for i, line in enumerate(lines):
        if section.match(line):
            in_section = True
            header_indent = None
            continue
        if not in_section:
            continue
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if bullet.match(line):
            continue
        m = header.match(line)
        if m and (header_indent is None or len(m.group(1)) <= header_indent):
            header_indent = len(m.group(1))
            continue
        # A non-header line at (or below) the header indent is a continuation.
        if header_indent is not None and indent <= header_indent:
            lines[i] = "    " + line


def _second_pass_docstring(app, what, name, obj, options, lines):
    """Post-napoleon fixes, run as a second autodoc-process-docstring hook.

    `_blank_line_before_fields` operates on the lines list, but the event hands
    the handler six arguments. Connecting the internal helper directly made
    Sphinx raise "takes 1 positional argument but 6 were given" and abort the
    whole build -- so the adapter exists to keep the two signatures apart.
    """
    if _is_project_docstring(name, obj):
        _blank_line_before_fields(lines)


def _blank_line_before_fields(lines):
    """Separate a field list (`:param x:`) from the paragraph above it.

    napoleon emits the conversion in place, so a docstring written as

        Sweeps the triangle mesh rotationally about an axis.
        :param angle: The rotation angle in degree.

    becomes a paragraph immediately followed by a field list. docutils then
    reports "Unexpected indentation" at the first continuation line and
    "Block quote ends without a blank line" (measured in extrude_rotation).

    A field list is a block, so it needs a blank line above it. Only insert
    one when the preceding line is ordinary text, never between consecutive
    fields or above the first field of a list already separated.
    """
    i = 1
    while i < len(lines):
        cur = lines[i]
        prev = lines[i - 1]
        is_field = cur.lstrip().startswith(":") and ":" in cur.lstrip()[1:]
        prev_is_field = prev.lstrip().startswith(":") and ":" in prev.lstrip()[1:]
        if (is_field and prev.strip() and not prev_is_field
                and not prev.lstrip().startswith(".. ")):
            lines.insert(i, "")
            i += 1
        i += 1


def _namespace_equation_labels(lines, name):
    """Make a docstring's equation label unique to its own page.

    Several pybind classes embed the same literal `:label: robust_loss`, and the
    same equation is documented twice -- once under
    `cloudViewer.pipelines.registration.RobustKernel` and once under
    `cloudViewer.t.pipelines.registration.robust_kernel.RobustKernel`. The
    second page then fails `-W` with "duplicate label of equation robust_loss".

    The label is local to one docstring's own text, so qualifying it with the
    object path keeps each page self-consistent while removing the collision.
    The matching `:eq:` references are rewritten in the same pass, otherwise the
    renamed equation would be unreachable ("equation not found").
    """
    import re
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", (name or "")).strip("_")
    if not suffix:
        return

    labels = set()
    for line in lines:
        m = re.match(r"^\s*:label:\s*(\S+)\s*$", line)
        if m:
            labels.add(m.group(1))

    if not labels:
        return

    for i, line in enumerate(lines):
        m = re.match(r"^(\s*:label:\s*)(\S+)(\s*)$", line)
        if m and m.group(2) in labels:
            lines[i] = f"{m.group(1)}{m.group(2)}_{suffix}{m.group(3)}"
            continue
        for label in labels:
            line = re.sub(r":eq:`\s*" + re.escape(label) + r"\s*`",
                          f":eq:`{label}_{suffix}`", line)
        lines[i] = line


def _repair_broken_links(lines):
    """Restore the `>` that a malformed link target is missing.

    TriangleMesh.compute_convex_hull contains

        `QHull docs <http://www.qhull.org/html/qh-impre.htm#joggle`__ for more

    -- the URL's closing `>` was written as a backtick, so RST sees an anonymous
    reference with no target ("Anonymous hyperlink mismatch: 1 references but 0
    targets"). Restoring `>` makes it the working link the text intends.
    """
    import re
    pattern = re.compile(r"<([^<>\s]+)`__")
    for i, line in enumerate(lines):
        lines[i] = pattern.sub(lambda m: "<" + m.group(1) + ">`__", line)


def _is_project_docstring(name, obj):
    """True when this docstring is one of ours to repair.

    `autodoc-process-docstring` fires for every documented object, including
    re-exported third-party ones. Their authors wrote valid RST for their own
    renderer, and our normalisers assume pybind's layout, so touching them
    corrupts them (torch's SummaryWriter failed with "Content block expected
    for the note directive" and "Error in image directive").

    pybind members report `__module__` values under cloudViewer (the CUDA
    bindings use cloudViewer.cuda.pybind.*), so that is the primary test. When
    autodoc passes no object, fall back to the qualified name.
    """
    if obj is not None:
        mod = getattr(obj, "__module__", "") or ""
        if mod:
            return mod.startswith("cloudViewer")
    return (name or "").split(".")[0] == "cloudViewer"


def _reflow(lines, name=""):
    """Apply every structural and inline repair to a docstring body.
    Split out from process_docstring so a docstring containing math can run the
    same repairs on its prose while its math block stays fenced off.
    """
    _dedent_docstring(lines)
    _indent_google_sections(lines)
    _blank_line_before_lists(lines)
    _indent_directive_bodies(lines)
    _promote_pipe_art_blocks(lines)
    _separate_indent_changes(lines)
    _indent_param_continuations(lines)
    _repair_broken_links(lines)
    _namespace_equation_labels(lines, name)
    for i, line in enumerate(lines):
        fixed = fix_namespace(line)
        fixed = _escape_bare_stars(fixed)
        # Directives also end in "::" (".. note::"). Trimming those would turn
        # the directive into plain text and break the body indentation added
        # just above, so only rewrite a "::" that opens a literal block.
        if _opens_literal_block(fixed) and not _is_directive(fixed):
            # Turn "text::" into "text:" so docutils stops expecting an
            # indented literal block that will never arrive.
            fixed = fixed.rstrip()
            fixed = fixed[:-1]
        lines[i] = fixed

    # Runs last, on the whole block: backtick pairing is paragraph-scoped in
    # RST, so a line-by-line check cannot see an opener that closes on another
    # line.
    _escape_unbalanced_backticks(lines)
    _escape_dangling_references(lines)


def _dedent_docstring(lines):
    """Remove the uniform indentation Sphinx does not strip from pybind text.

    autodoc passes a docstring's lines through unchanged. Many pybind-generated
    docstrings look like:

        'Serves CloudViewer geometries via Dash.'
        ''
        '    This function creates a Plotly figure ...'

    The first line is flush left and the body is indented four spaces. docutils
    parses line 1 as a paragraph, then meets an indented block after a blank
    line and reports "Unexpected indentation" (ERROR/3). Measured on this tree
    that accounts for 106 of 258 remaining diagnostics.

    Dedenting the block is the fix, not suppression: the content is correct,
    only its incidental indentation is wrong. The first line is left alone
    because it is typically the summary.
    """
    if not lines:
        return
    body = [l for l in lines[1:] if l.strip()]
    if not body:
        return
    indent = min(len(l) - len(l.lstrip()) for l in body)
    if indent == 0:
        return
    for i in range(1, len(lines)):
        if lines[i].startswith(" " * indent):
            lines[i] = lines[i][indent:]


def _section_headers():
    import re
    return re.compile(
        r"^(\s*)(Args|Arguments|Parameters|Returns|Raises|Yields|"
        r"Examples?|Notes?|Warnings?|See Also):\s*$")


def _indent_google_sections(lines):
    """Indent Google-style section bodies that pybind leaves flush left.

    napoleon converts "Args:" into ":param x:" fields only when the entries
    are indented under the header. pybind11 writes them flush left:

        Args:
        filename (str): Path to the mesh file.

    napoleon sees no body, leaves the text alone, and docutils then reports
    "Unexpected indentation" (78 diagnostics on this tree) because the next
    line is indented relative to a paragraph that was never opened.

    Indenting the body here -- before napoleon runs, which a lower event
    priority ensures -- makes napoleon perform the conversion it was written
    to do, so the page shows real parameter fields instead of a mangled block.
    """
    header = _section_headers()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = header.match(line)
        if not m:
            i += 1
            continue

        base = len(m.group(1))
        i += 1
        body: list[str] = []
        while i < len(lines):
            nxt = lines[i]
            if header.match(nxt):
                break
            if not nxt.strip():
                # Keep the section open only if non-blank content follows at a
                # deeper indent; otherwise the section has ended.
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j >= len(lines):
                    break
                ind = len(lines[j]) - len(lines[j].lstrip())
                if ind > base:
                    body.append(nxt)
                    i += 1
                    continue
                break
            body.append(nxt)
            i += 1

        pad = " " * (base + 4)
        for b in body:
            if b.strip() and (len(b) - len(b.lstrip())) <= base:
                out.append(pad + b.lstrip())
            else:
                out.append(b)
    lines[:] = out


def _blank_line_before_lists(lines):
    """Separate a list from the text directly above it.

    pybind docstrings habitually write a field body and a list with no blank
    line between them:

        :param geometry: The 3D data can be provided as
                         - A list of geometry types
                         - A list of dictionaries

    docutils requires a blank line before a bullet list, and without one it
    reports "Unexpected indentation" or "Definition list ends without a blank
    line". Measured on this tree that accounts for most of the 81 + 13
    remaining diagnostics in those classes. Inserting the separator is correct:
    the list is meant to be a list.
    """
    import re

    bullet = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")
    i = 1
    while i < len(lines):
        prev, cur = lines[i - 1], lines[i]
        if (bullet.match(cur) and prev.strip()
                and not bullet.match(prev) and not prev.rstrip().endswith("::")):
            lines.insert(i, "")
            i += 1
        i += 1


def _indent_directive_bodies(lines):
    """Give RST directives in a pybind docstring the blank line they need.

    Tensor.__dlpack__ documents itself as:

        .. note::
        ``to_dlpack`` is a legacy DLPack interface. ...

    A directive's body must be separated by a blank line and indented. Without
    that, docutils reports "Explicit markup ends without a blank line" (and,
    for an empty body, "Content block expected"). Insert the blank line and
    indent the following block so the note renders as a note instead of
    leaking raw markup onto the page.
    """
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip() if lines[i] else ""
        if not re.match(r"^\.\.\s+[\w-]+::(?:\s+.*)?$", stripped):
            i += 1
            continue

        # A directive may already carry content on its opening line (for
        # example WebRTC's '.. note:: Ordering ...'). Its flush-left
        # continuation still belongs to that directive, just as with an empty
        # opening line. Do not mistake the inline text for a directive end.
        # Body starts at the next line; stop at a blank line that is already
        # followed by indented text (the directive is already well-formed).
        j = i + 1
        if j < len(lines) and (not lines[j].strip() or lines[j].startswith(" ")):
            i += 1
            continue

        body = []
        while j < len(lines) and lines[j].strip():
            body.append(lines[j])
            j += 1
        if not body:
            i += 1
            continue

        # A directive's option block (`:nowrap:`, `:name:`) and its content
        # must both be indented under the directive, and the content may follow
        # the option after a blank line. `.. math::` is the case this tree
        # hits: pybind writes
        #
        #     .. math::
        #         :nowrap:
        #
        #     \begin{align}
        #
        # so the option parses but the LaTeX does not, and `\frac{1}{|X|}`
        # becomes a substitution reference ("Undefined substitution
        # referenced: X"). Take the whole run -- across blank lines, up to the
        # next flush-left construct -- and indent it, preserving relative
        # indentation so the option still parses as an option.
        if body and body[0].lstrip().startswith(":"):
            j = len(body) and i + 1 + len(body) or j
            while j < len(lines) and (
                    not lines[j].strip()
                    or (lines[j].startswith(" ")
                        and not lines[j].lstrip().startswith(".. "))):
                j += 1
            body = lines[i + 1:j]
            while body and not body[-1].strip():
                body.pop()
                j -= 1

        for k, line in enumerate(body):
            if line.strip():
                body[k] = "   " + line
        # Inline content and its wrapped continuation form one paragraph.
        # Empty directive headers instead need the separating blank line.
        separator = [""] if stripped.endswith("::") else []
        lines[i + 1:i + 1 + len(body)] = separator + body
        i = i + 1 + len(body)
        continue


def _promote_pipe_art_blocks(lines):
    """Turn ASCII tree art into a literal block.

    Several pybind docstrings embed a directory tree:

        Data class for `RedwoodIndoorOffice2`, containing ... :

            RedwoodIndoorOffice2
            |-- colors
            |   |-- 00000.jpg
            '-- office.ply

    docutils reads ``|-- colors`` as a *substitution reference* opener (the
    ``|name|`` syntax) and every line then reports "Inline substitution
    reference start-string without end-string" -- 60 diagnostics on this tree.
    It also reads the trailing ". :" as the start of a field list.

    A literal block is what the author meant: the text must render verbatim.
    Emitting ``::`` above the run and indenting it preserves the art exactly.

    Detection keys on the tree markers (``|``, ``'--``, ``-- ``), not on
    indentation: this runs after ``_dedent_docstring``, which has already
    flattened the block to column 0.
    """
    art_markers = ("|", "'--", "`--")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.lstrip().startswith(art_markers):
            i += 1
            continue

        # Grow the run: consecutive non-blank lines that are art or plain
        # content belonging to the same tree (indented continuation).
        start = i
        while start > 0 and lines[start - 1].strip() and (
                lines[start - 1].startswith(" ") or lines[start - 1].lstrip().startswith(art_markers)):
            start -= 1

        end = i
        while end < len(lines) and lines[end].strip():
            end += 1

        # Already a literal block (previous line is "::") -- leave it alone.
        if start > 0 and lines[start - 1].rstrip().endswith("::"):
            i = end
            continue

        # The art root ("Redwood") is the first line of the run when it carries
        # no marker of its own but introduces the tree.
        block = lines[start:end]
        if start > 0:
            prev = lines[start - 1].rstrip()
            if prev.endswith(". :"):
                lines[start - 1] = prev[:-2]

        lines[start:end] = ["::", ""] + ["    " + b for b in block]
        i = end + 3


import re as _re


def _separate_indent_changes(lines):
    """Insert a blank line wherever consecutive text lines change indentation.

    pybind docstrings interleave a paragraph with continuation-aligned code::

        The attributes of the triangle mesh have different levels:

        import cloudViewer as cv3d
        mesh.vertex.positions = cv3d.core.Tensor([[0, 0, 0],
                                                 [0, 0, 1]], dtype_f, device)

    The second line of that call is indented only to align under the opening
    bracket. docutils has no paragraph open at that point, so it reports
    "Unexpected indentation" (65) and then "Block quote ends without a blank
    line" (36) when the indent drops back.

    A blank line between two different indentation levels makes each run its
    own block, which is what the text means. Measured on this tree this one
    transform clears 101 of the 197 remaining diagnostics.

    **Never inside a directive body.** A directive's option block
    (``:nowrap:``) and its content both live under the ``.. math::`` line, and
    the blank line this would insert between them ends the directive. The
    content then parses as ordinary text, and math such as ``\\frac{1}{|X|}``
    becomes a substitution reference -- measured as an "Undefined substitution
    referenced: X" failure in ``PointCloud.compute_metrics``.
    """
    directive_indent = None      # indent of the governing ".. name::" line
    field_indent = None          # indent of the governing ":param x:" line
    i = 1
    while i < len(lines):
        prev, cur = lines[i - 1], lines[i]
        cur_indent = len(cur) - len(cur.lstrip()) if cur.strip() else 0
        prev_indent = len(prev) - len(prev.lstrip()) if prev.strip() else 0

        # Track whether we are inside a directive: a line starting with ".. "
        # opens one at its own indent; leaving that indent closes it.
        if cur.strip().startswith(".. "):
            directive_indent = cur_indent
        elif cur.strip() and directive_indent is not None and cur_indent <= directive_indent:
            directive_indent = None

        # Track field bodies too. A `:param x:` continuation is deliberately
        # indented under its field, and inserting a blank line there splits the
        # description ("Block quote ends without a blank line; unexpected
        # unindent" in extrude_rotation).
        if cur.strip().startswith(":") and ":" in cur.strip()[1:]:
            field_indent = cur_indent
        elif cur.strip() and field_indent is not None and cur_indent <= field_indent:
            field_indent = None

        inside_directive = (directive_indent is not None
                            and (cur_indent > directive_indent
                                 or prev_indent > directive_indent))
        # A parameter header ("name (type): text") owns the lines that follow
        # it, even when they are less indented than the header. pybind writes
        # the continuation flush at one space:
        #
        #     resolution (int): The resolution defines the number of
        #      intermediate sweeps
        #
        # Without this the blank line this pass inserts between the header and
        # its continuation splits the field ("Block quote ends without a blank
        # line; unexpected unindent" in extrude_rotation).
        prev_is_param = re.match(
            r"^\s*[A-Za-z_][\w.]*\s*(\([^)]*\))?\s*:\s*\S", prev) is not None
        inside_field = (field_indent is not None
                        and (cur_indent > field_indent
                             or prev_indent > field_indent)) or prev_is_param

        if (prev.strip() and cur.strip()
                and prev_indent != cur_indent
                and not inside_directive
                and not inside_field):
            lines.insert(i, "")
            i += 1
        i += 1


def _escape_unbalanced_backticks(lines):
    """Escape backticks in any paragraph whose pairing is not balanced.

    Backtick pairing in RST is **paragraph-scoped**, not line-scoped, so this
    must see the whole block. Two genuine shapes on this tree:

    1. `FileGeometry.from_bytes` carries the CPython text
       "use `sys.byteorder' ..." -- an interpreted-text opener closed with a
       quote instead of a backtick. The opener is on one line; docutils then
       scans forward and reports "Inline interpreted text or phrase reference
       start-string without end-string" at the paragraph start. A line-local
       check cannot see this, which is why the earlier per-line version let it
       through.
    2. `SequentialMatchingOptions.vocab_tree_path` contains "````" -- four
       backticks, which is an odd *marker* count even though the character
       count is even. A parity test passes it; docutils opens and never closes.

    Both render the characters literally when escaped, which is what the
    source text meant.
    """
    if not lines:
        return

    # Split into paragraphs on blank lines; RST cannot pair across one.
    start = 0
    for idx in range(len(lines) + 1):
        if idx == len(lines) or not lines[idx].strip():
            _escape_paragraph_backticks(lines, start, idx)
            start = idx + 1


def _escape_paragraph_backticks(lines, start, end):
    """Fix pairing for lines[start:end] as one paragraph."""
    if start >= end:
        return

    # CPython builtins still use the Python 2 ``literal'' quoting convention.
    # docutils opens an inline literal at `` and looks for ``, never finding it
    # because the text closes with ''. Rewrite to the modern delimiter.
    for i in range(start, end):
        lines[i] = re.sub(r"``([^`']+)''", lambda m: "``" + m.group(1) + "``", lines[i])

    text = "\n".join(lines[start:end])

    # Measured rules for what docutils treats as an unterminated opener, over
    # runs of 1..9 backticks with no closer:
    #
    #   run length 1, 2, 5..9  -> no diagnostic
    #   run length 3, 4        -> "without end-string"
    #
    # So a bare 3 or 4 is the malformed shape (`SequentialMatchingOptions.
    # vocab_tree_path` contains a bare 4); a run of 2 opens an inline literal
    # that a later run of 2 closes, and a longer run is taken as literal text.
    bad_run = re.search(r"(?<!`)`{3,4}(?!`)", text)

    # An unpaired single backtick opens interpreted text with no closer. This
    # is the FileGeometry.from_bytes case, where the opener on one line is
    # "closed" by a quote on the next.
    unpaired_single = len(re.findall(r"(?<!`)`(?!`)", text)) % 2 == 1

    if not bad_run and not unpaired_single:
        return

    for i in range(start, end):
        lines[i] = re.sub(r"`+", lambda m: "\\" + m.group(0), lines[i])


def _opens_literal_block(text):
    """True when the line ends a sentence with '::' and opens a literal block."""
    if text is None:
        return False
    stripped = text.rstrip()
    return stripped.endswith("::") and len(stripped) > 2


def _is_directive(text):
    """True for an RST directive such as '.. note::'.

    Directives end in '::' too, but their trailing colons are syntax, not a
    literal-block marker. Stripping them silently converted the note into
    plain text.
    """
    return text is not None and text.lstrip().startswith(".. ")


def _escape_bare_stars(text):
    """Escape '*' runs that docutils would read as inline markup.

    Two shapes occur in pybind docstrings, both of them code rather than
    emphasis::

        dict(**kwargs) -> new dictionary ...
        np.meshgrid(*3*[np.linspace(-1, 1, num=64)], indexing='ij')

    docutils opens a strong reference at ``**kwargs`` and an emphasis at
    ``*3*`` (the pair closes early and leaves a second opener), reporting
    "Inline strong/emphasis start-string without end-string". Escaping every
    run of one or more asterisks renders them literally, which is what the
    code means. Measured: 8 strong + 2 emphasis diagnostics, all cleared.
    """
    if text is None:
        return text
    import re
    return re.sub(r"\*+", lambda m: "\\" + m.group(0), text)


def _set_html_title(app, config):
    """Recompute html_title from the final `release` value.

    Runs on ``config-inited``, which fires after ``-D release=...`` has been
    applied. Computing the title at import time instead would bake in the
    value from version.txt even for development builds, where make_docs.py
    substitutes a ``main-<hash>`` release string.
    """
    config.html_title = f"ACloudViewer {config.release} documentation"


def setup(app):
    """Sphinx setup function to connect autodoc events and plugin doc sync."""
    # pybind11-generated docstrings are normalised, not suppressed: see
    # process_docstring below. `-W` therefore stays authoritative for every
    # page, generated or hand-written.

    # Runs after -D overrides, so the title tracks the effective release.
    app.connect("config-inited", _set_html_title)
    app.connect("builder-inited", _sync_plugin_docs)
    app.connect("builder-inited", _sync_guides_plugins)
    app.connect("builder-inited", _sync_plugin_showcase_assets)
    app.connect("build-finished", _cleanup_plugin_docs)
    app.connect("build-finished", _cleanup_guides_plugins)
    app.connect("autodoc-skip-member", skip)
    app.connect("autodoc-process-signature", process_signature)
    # Priority matters: napoleon registers at the default 500 and converts
    # "Args:" sections only when their entries are indented. pybind docstrings
    # write "Args:" with flush-left entries, so napoleon silently does nothing
    # and docutils later reports "Unexpected indentation". Running our
    # normalisation FIRST (lower priority runs earlier) lets napoleon see a
    # well-formed docstring.
    app.connect("autodoc-process-docstring", process_docstring, priority=400)
    # Second pass, after napoleon (which registers at the default 500) has
    # converted "Args:" into ":param x:" fields. The blank line a field list
    # needs above it can only be seen once the fields exist, so this cannot run
    # in the first hook -- at priority 400 there are no fields to separate yet.
    app.connect("autodoc-process-docstring", _second_pass_docstring,
                priority=600)
    # Add Google analytics
    app.add_js_file("https://www.googletagmanager.com/gtag/js?id=G-3TQPKGV6Z3",
                    **{'async': 'async'})
    app.add_js_file(None,
                    body="""
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-3TQPKGV6Z3');""")
