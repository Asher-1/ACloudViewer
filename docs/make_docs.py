#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# -                       ACloudViewer Documentation Builder                 -
# ----------------------------------------------------------------------------
# Based on Open3D's documentation build system
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""
Sphinx documentation builder for ACloudViewer

Usage examples:
    # Build documentation (Sphinx + Doxygen)
    python make_docs.py --sphinx --doxygen

    # Build for release (use version number instead of git hash)
    python make_docs.py --is_release --sphinx --doxygen

    # Build only Sphinx documentation
    python make_docs.py --sphinx

    # Build only C++ API documentation
    python make_docs.py --doxygen

    # Clean and rebuild
    python make_docs.py --clean --sphinx --doxygen
"""

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _create_or_clear_dir(dir_path):
    """Create directory or clear it if it exists."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Removed directory {dir_path}")
    os.makedirs(dir_path)
    print(f"Created directory {dir_path}")


class DoxygenDocsBuilder:
    """Build C++ API documentation using Doxygen."""

    def __init__(self, html_output_dir):
        self.html_output_dir = html_output_dir

    def run(self):
        """Run Doxygen to generate C++ API documentation."""
        if not os.path.exists("Doxyfile"):
            print("⚠️  Doxyfile not found, skipping C++ API documentation")
            return

        print("🔨 Building C++ API documentation with Doxygen...")
        doxygen_temp_dir = "doxygen"
        _create_or_clear_dir(doxygen_temp_dir)

        cmd = ["doxygen", "Doxyfile"]
        print(f'Running: "{" ".join(cmd)}"')
        try:
            subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
            
            # Copy Doxygen HTML output to final location
            output_path = os.path.join(self.html_output_dir, "html", "cpp_api")
            if os.path.exists(os.path.join("doxygen", "html")):
                shutil.copytree(
                    os.path.join("doxygen", "html"),
                    output_path,
                )
                print(f"✅ Doxygen docs generated at {output_path}/index.html")
            else:
                print("⚠️  Doxygen HTML output not found")
        except subprocess.CalledProcessError as e:
            print(f"❌ Doxygen build failed: {e}")
            raise
        finally:
            # Clean up temporary directory
            if os.path.exists(doxygen_temp_dir):
                shutil.rmtree(doxygen_temp_dir)


class SphinxDocsBuilder:
    """Build main documentation and Python API using Sphinx."""

    def __init__(self, current_file_dir, html_output_dir, is_release, parallel):
        self.current_file_dir = current_file_dir
        self.html_output_dir = html_output_dir
        self.is_release = is_release
        self.parallel = parallel

    def run(self):
        """Run Sphinx to build HTML documentation."""
        print("🔨 Building documentation with Sphinx...")
        
        build_dir = os.path.join(self.html_output_dir, "html")
        nproc = multiprocessing.cpu_count() if self.parallel else 1
        print(f"Building docs with {nproc} processes")

        # Get version from project if available
        version = "3.9"
        release = "3.9.3"
        
        if self.is_release:
            print(f"Building docs for release: {release}")
            cmd = [
                "sphinx-build",
                "-j", str(nproc),
                "-b", "html",
                "-D", f"version={version}",
                "-D", f"release={release}",
                "source",  # Source directory
                build_dir,  # Output directory
            ]
        else:
            print("Building development documentation")
            cmd = [
                "sphinx-build",
                "-j", str(nproc),
                "-b", "html",
                "source",
                build_dir,
            ]

        print(f'Running: "{" ".join(cmd)}"')
        try:
            subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
            
            # Create symlink for convenience
            html_link = os.path.join(self.current_file_dir, "html")
            if not os.path.exists(html_link):
                os.symlink(build_dir, html_link)
                
            index_html = Path(build_dir) / "index.html"
            if index_html.exists():
                print(f"✅ Sphinx docs generated at {index_html.as_uri()}")
            else:
                print(f"✅ Sphinx docs generated at {build_dir}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Sphinx build failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Build ACloudViewer documentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--sphinx",
        action="store_true",
        default=False,
        help="Build Sphinx documentation (HTML)",
    )
    parser.add_argument(
        "--doxygen",
        action="store_true",
        default=False,
        help="Build Doxygen documentation (C++ API)",
    )
    parser.add_argument(
        "--is_release",
        action="store_true",
        default=False,
        help="Build for release (use version number instead of git hash)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Enable parallel Sphinx build",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean output directory before building",
    )

    args = parser.parse_args()

    # Get current directory
    pwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(pwd)

    print("=" * 70)
    print("  ACloudViewer Documentation Build System")
    print("=" * 70)
    print()

    # Output directory
    html_output_dir = os.path.join(pwd, "_out")
    
    # Clean or create output directory
    if args.clean or (args.sphinx or args.doxygen):
        _create_or_clear_dir(html_output_dir)

    # Build Sphinx documentation
    if args.sphinx:
        sdb = SphinxDocsBuilder(pwd, html_output_dir, args.is_release, args.parallel)
        sdb.run()
    else:
        print("ℹ️  Sphinx build disabled, use --sphinx to enable")

    # Build Doxygen documentation  
    if args.doxygen:
        ddb = DoxygenDocsBuilder(html_output_dir)
        ddb.run()
    else:
        print("ℹ️  Doxygen build disabled, use --doxygen to enable")

    # If neither Sphinx nor Doxygen specified, show help
    if not args.sphinx and not args.doxygen:
        print()
        print("⚠️  No build targets specified!")
        print("Use --sphinx and/or --doxygen to build documentation")
        parser.print_help()
        return 1

    print()
    print("=" * 70)
    print("✅ Documentation build complete!")
    print("=" * 70)
    
    # Show output location
    if args.sphinx:
        html_dir = os.path.join(html_output_dir, "html")
        print(f"\n📂 Output: {html_dir}")
        print(f"\nTo view the documentation:")
        print(f"  cd {pwd}")
        print(f"  python3 -m http.server 8080 --directory {html_dir}")
        print(f"  Then open: http://localhost:8080")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

