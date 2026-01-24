# CMake configuration for ACloudViewer documentation
#
# Build Strategy:
#   CMake delegates to make_docs.py for all documentation generation.
#   This ensures consistency between CMake builds and direct make_docs.py calls.
#
# Architecture:
#   CMakeLists.txt → make_docs.py → {Doxygen, Sphinx, Python API generation}
#
# Usage:
#   cd build_app && make documentation
#
# The 'documentation' target calls make_docs.py which handles:
#   1. Doxygen: Generate C++ API HTML → _out/html/cpp_api/api/
#   2. Python API: Generate .rst files from cloudViewer module
#   3. Jupyter: Copy notebook files to tutorial directories
#   4. Sphinx: Build HTML docs → _out/html/
#
# Reference: util/ci_utils.sh::build_docs() for the canonical implementation
#
cmake_minimum_required(VERSION 3.18)

# Documentation options (for compatibility, but make_docs.py decides what to build)
option(BUILD_DOCUMENTATION "Build documentation" ON)

if(NOT BUILD_DOCUMENTATION)
    message(STATUS "Documentation build is disabled. Enable with -DBUILD_DOCUMENTATION=ON")
    return()
endif()

# Find Python (required for make_docs.py)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Check if make_docs.py exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/make_docs.py")
    message(FATAL_ERROR "make_docs.py not found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Documentation paths
set(DOCS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DOCS_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(DOCS_OUTPUT_DIR ${DOCS_BUILD_DIR}/_out)
set(DOCS_HTML_DIR ${DOCS_OUTPUT_DIR}/html)

message(STATUS "Documentation paths:")
message(STATUS "  - Source:   ${DOCS_SOURCE_DIR}")
message(STATUS "  - Build:    ${DOCS_BUILD_DIR}")
message(STATUS "  - Output:   ${DOCS_HTML_DIR}")
message(STATUS "  - Doxygen:  ${DOCS_DOXYGEN_DIR}")

# Set up PYTHONPATH for documentation build
# Include potential locations where cloudViewer module might be
set(DOC_PYTHONPATH 
    "${CMAKE_BINARY_DIR}/lib/python_package"
    "${CMAKE_BINARY_DIR}/lib/Release/Python/cuda"
    "${CMAKE_SOURCE_DIR}/build/lib/python_package"
    "${CMAKE_SOURCE_DIR}/build_app/lib/python_package"
    "${CMAKE_SOURCE_DIR}/build_app/lib/Release/Python/cuda"
)
string(REPLACE ";" ":" DOC_PYTHONPATH_STR "${DOC_PYTHONPATH}")

# Doxygen output directory (in build directory, not source!)
set(DOCS_DOXYGEN_DIR ${DOCS_BUILD_DIR}/doxygen)

# Main documentation target
# Delegates all work to make_docs.py (matches ci_utils.sh::build_docs)
add_custom_target(documentation
    COMMAND ${CMAKE_COMMAND} -E echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    COMMAND ${CMAKE_COMMAND} -E echo "📚 Building ACloudViewer Documentation"
    COMMAND ${CMAKE_COMMAND} -E echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Strategy: Delegate to make_docs.py for consistency"
    COMMAND ${CMAKE_COMMAND} -E echo "  1. Doxygen   → C++ API HTML"
    COMMAND ${CMAKE_COMMAND} -E echo "  2. Python API → Generate .rst from cloudViewer module"
    COMMAND ${CMAKE_COMMAND} -E echo "  3. Jupyter   → Copy notebooks"
    COMMAND ${CMAKE_COMMAND} -E echo "  4. Sphinx    → Build HTML docs"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E env 
        "PYTHONPATH=${DOC_PYTHONPATH_STR}:$ENV{PYTHONPATH}"
        ${Python3_EXECUTABLE} ${DOCS_SOURCE_DIR}/make_docs.py --is_release --parallel --sphinx --doxygen --output-dir ${DOCS_OUTPUT_DIR} --doxygen-dir ${DOCS_DOXYGEN_DIR} --execute-notebooks=auto --py-api-rst=always --py-example-rst=always
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    COMMAND ${CMAKE_COMMAND} -E echo "✅ Documentation Build Complete"
    COMMAND ${CMAKE_COMMAND} -E echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "📦 Output: ${DOCS_HTML_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "🌐 Preview: python3 -m http.server --directory ${DOCS_HTML_DIR} 8080"
    WORKING_DIRECTORY ${DOCS_SOURCE_DIR}
    COMMENT "Building documentation via make_docs.py"
    VERBATIM
)

# Make documentation depend on python-package if available
if(TARGET python-package)
    add_dependencies(documentation python-package)
    message(STATUS "Documentation will use cloudViewer from python-package target")
else()
    message(WARNING "python-package target not found.")
    message(WARNING "Documentation may not include Python API if cloudViewer is not installed.")
    message(WARNING "To fix: Build with -DBUILD_PYTHON_MODULE=ON")
endif()

# Legacy convenience targets (for backward compatibility)
# These all delegate to the main 'documentation' target

# Doxygen-only build (actually builds everything via make_docs.py)
add_custom_target(doxygen
    COMMAND ${CMAKE_COMMAND} --build . --target documentation
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building documentation (doxygen target is an alias)"
)

# Sphinx-only build (actually builds everything via make_docs.py)
add_custom_target(sphinx-html
    COMMAND ${CMAKE_COMMAND} --build . --target documentation
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building documentation (sphinx-html target is an alias)"
)

# Clean target
add_custom_target(doc-clean
    COMMAND ${CMAKE_COMMAND} -E echo "🗑️  Cleaning documentation build artifacts..."
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${DOCS_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${DOCS_BUILD_DIR}/doxygen
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${DOCS_SOURCE_DIR}/doxygen
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${DOCS_SOURCE_DIR}/source/python_api
    COMMAND ${CMAKE_COMMAND} -E echo "✅ Cleaned:"
    COMMAND ${CMAKE_COMMAND} -E echo "   - ${DOCS_OUTPUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "   - ${DOCS_BUILD_DIR}/doxygen"
    COMMAND ${CMAKE_COMMAND} -E echo "   - ${DOCS_SOURCE_DIR}/doxygen (legacy)"
    COMMAND ${CMAKE_COMMAND} -E echo "   - ${DOCS_SOURCE_DIR}/source/python_api"
    WORKING_DIRECTORY ${DOCS_SOURCE_DIR}
    COMMENT "Cleaning documentation build artifacts"
)

# Install documentation (optional)
if(EXISTS ${DOCS_HTML_DIR})
    install(DIRECTORY ${DOCS_HTML_DIR}/
        DESTINATION share/doc/acloudviewer
        COMPONENT documentation
        OPTIONAL
    )
endif()

# Help message
add_custom_target(doc-help
    COMMAND ${CMAKE_COMMAND} -E echo "═══════════════════════════════════════════════════════════════"
    COMMAND ${CMAKE_COMMAND} -E echo "ACloudViewer Documentation Build Targets"
    COMMAND ${CMAKE_COMMAND} -E echo "═══════════════════════════════════════════════════════════════"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Main target:"
    COMMAND ${CMAKE_COMMAND} -E echo "  documentation  - Build all documentation (calls make_docs.py)"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Legacy aliases (all call 'documentation'):"
    COMMAND ${CMAKE_COMMAND} -E echo "  doxygen        - Alias for documentation"
    COMMAND ${CMAKE_COMMAND} -E echo "  sphinx-html    - Alias for documentation"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Utility targets:"
    COMMAND ${CMAKE_COMMAND} -E echo "  doc-clean      - Clean documentation artifacts"
    COMMAND ${CMAKE_COMMAND} -E echo "  doc-help       - Show this message"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Output directory:"
    COMMAND ${CMAKE_COMMAND} -E echo "  ${DOCS_HTML_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Implementation:"
    COMMAND ${CMAKE_COMMAND} -E echo "  All targets delegate to make_docs.py for consistency"
    COMMAND ${CMAKE_COMMAND} -E echo "  Reference: util/ci_utils.sh::build_docs()"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "═══════════════════════════════════════════════════════════════"
)

message(STATUS "Documentation build configured:")
message(STATUS "  - Main target: 'documentation' (calls make_docs.py)")
message(STATUS "  - HTML output:  ${DOCS_HTML_DIR}")
message(STATUS "  - Doxygen temp: ${DOCS_DOXYGEN_DIR}")
message(STATUS "  - Python paths: ${DOC_PYTHONPATH}")
