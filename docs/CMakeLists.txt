# CMake configuration for ACloudViewer documentation
#
# Build Strategy (following Open3D):
#   1. Doxygen generates independent C++ API HTML
#   2. Sphinx generates Python API + Tutorials
#   3. Outputs are combined via file system
#
# Two build methods available:
#
#   Method 1: Direct CMake targets (faster, for development)
#     cd build_app && make doxygen sphinx-html
#
#   Method 2: Through make_docs.py (complete, for production)
#     cd docs && python make_docs.py --sphinx --doxygen
#     (includes Python API auto-generation, Jupyter notebooks copy, etc.)
#
# IMPORTANT: Before running CMake, set up the Python environment:
#   export PYENV_ROOT=~/.pyenv
#   export PYTHON_VERSION=3.11
#   export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH"
#
# Or use the CI/CD script:
#   source util/ci_utils.sh && build_docs ON
#
# Reference: https://github.com/isl-org/Open3D/tree/main/docs
#
cmake_minimum_required(VERSION 3.18)

# Documentation options
option(BUILD_DOXYGEN "Build C++ API documentation with Doxygen" ON)
option(BUILD_SPHINX "Build Sphinx documentation (HTML)" ON)
option(BUILD_SPHINX_LATEX "Build Sphinx documentation (LaTeX/PDF)" OFF)

if(NOT BUILD_DOCUMENTATION)
    message(STATUS "Documentation build is disabled. Enable with -DBUILD_DOCUMENTATION=ON")
    return()
endif()

# Find required tools
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(Doxygen)

# Check dependencies
if(BUILD_DOXYGEN AND NOT DOXYGEN_FOUND)
    message(WARNING "Doxygen not found. C++ API documentation will not be built.")
    set(BUILD_DOXYGEN OFF)
endif()

# Check for Sphinx
if(BUILD_SPHINX)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sphinx"
        RESULT_VARIABLE SPHINX_CHECK
        OUTPUT_QUIET ERROR_QUIET
    )
    if(SPHINX_CHECK)
        message(WARNING "Sphinx not found. Install with: pip install sphinx")
        set(BUILD_SPHINX OFF)
    endif()
endif()

# Documentation paths
set(DOCS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DOCS_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(DOCS_OUTPUT_DIR ${DOCS_BUILD_DIR}/_out)
set(DOXYGEN_OUTPUT_DIR ${DOCS_BUILD_DIR}/doxygen)
set(DOXYGEN_XML_DIR ${DOXYGEN_OUTPUT_DIR}/xml)
set(SPHINX_SOURCE_DIR ${DOCS_SOURCE_DIR}/source)
set(SPHINX_OUTPUT_DIR ${DOCS_OUTPUT_DIR}/html)

# Configure Doxyfile
if(BUILD_DOXYGEN)
    set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR})
    set(DOXYGEN_OUTPUT_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
    
    configure_file(
        ${DOCS_SOURCE_DIR}/Doxyfile.in
        ${DOCS_BUILD_DIR}/Doxyfile
        @ONLY
    )
    
    # Doxygen target
    add_custom_target(doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOCS_BUILD_DIR}/Doxyfile
        WORKING_DIRECTORY ${DOCS_SOURCE_DIR}
        COMMENT "Generating C++ API documentation with Doxygen"
        VERBATIM
    )
    
    # Copy mainpage images to Doxygen HTML output
    # These images are referenced in mainpage.dox via \htmlonly blocks
    # and won't be automatically copied by Doxygen
    add_custom_command(TARGET doxygen POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "📸 Copying mainpage images to Doxygen output..."
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DOCS_SOURCE_DIR}/images/AbstractionLayers.png
            ${DOCS_SOURCE_DIR}/images/MainUI.png
            ${DOCS_SOURCE_DIR}/images/ICP-registration.png
            ${DOCS_SOURCE_DIR}/images/Reconstruction.png
            ${DOCS_SOURCE_DIR}/images/SemanticAnnotation.png
            ${DOXYGEN_OUTPUT_DIR}/html/
        COMMENT "Copying mainpage images (AbstractionLayers.png, MainUI.png, etc.)"
        VERBATIM
    )
    
    message(STATUS "Doxygen documentation will be built to: ${DOXYGEN_OUTPUT_DIR}")
endif()

# Sphinx documentation
if(BUILD_SPHINX)
    # Find sphinx-build
    find_program(SPHINX_BUILD
        NAMES sphinx-build sphinx-build-3
        DOC "Path to sphinx-build executable"
    )
    
    if(NOT SPHINX_BUILD)
        message(WARNING "sphinx-build not found. Please install Sphinx.")
        set(BUILD_SPHINX OFF)
    else()
        message(STATUS "Found sphinx-build: ${SPHINX_BUILD}")
        
        # Sphinx options
        set(SPHINX_OPTS "" CACHE STRING "Extra options for sphinx-build")
        set(SPHINX_VERBOSE "" CACHE STRING "Verbosity: -v, -vv, -vvv")
        
        # Ensure Pandoc is in PATH (for nbsphinx)
        find_program(PANDOC_EXECUTABLE pandoc
            PATHS $ENV{HOME}/bin /usr/local/bin /usr/bin
            DOC "Path to pandoc executable"
        )
        if(PANDOC_EXECUTABLE)
            get_filename_component(PANDOC_DIR ${PANDOC_EXECUTABLE} DIRECTORY)
            message(STATUS "Found pandoc: ${PANDOC_EXECUTABLE}")
        else()
            message(WARNING "Pandoc not found. Jupyter notebooks may not render properly.")
        endif()
        
        # HTML documentation
        add_custom_target(sphinx-html
            COMMAND ${CMAKE_COMMAND} -E env "PATH=${PANDOC_DIR}:$ENV{PATH}"
                ${SPHINX_BUILD}
                -b html
                ${SPHINX_VERBOSE}
                ${SPHINX_OPTS}
                -d ${DOCS_BUILD_DIR}/.doctrees
                ${SPHINX_SOURCE_DIR}
                ${SPHINX_OUTPUT_DIR}
            WORKING_DIRECTORY ${DOCS_SOURCE_DIR}
            COMMENT "Building HTML documentation with Sphinx"
            VERBATIM
        )
        
        # Make sphinx-html depend on doxygen if enabled
        if(BUILD_DOXYGEN)
            add_dependencies(sphinx-html doxygen)
            
            # Copy Doxygen HTML to Sphinx output (for unified documentation)
            add_custom_command(TARGET sphinx-html POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "📁 Copying Doxygen HTML to unified output..."
                COMMAND ${CMAKE_COMMAND} -E remove_directory ${SPHINX_OUTPUT_DIR}/cpp_api
                COMMAND ${CMAKE_COMMAND} -E copy_directory 
                    ${DOXYGEN_OUTPUT_DIR}/html
                    ${SPHINX_OUTPUT_DIR}/cpp_api
                COMMAND ${CMAKE_COMMAND} -E echo "✅ Doxygen HTML copied to ${SPHINX_OUTPUT_DIR}/cpp_api"
                COMMENT "Copying Doxygen HTML to unified documentation output"
                VERBATIM
            )
        endif()
        
        # LaTeX/PDF documentation (optional)
        if(BUILD_SPHINX_LATEX)
            add_custom_target(sphinx-latex
                COMMAND ${SPHINX_BUILD}
                    -b latex
                    ${SPHINX_VERBOSE}
                    ${SPHINX_OPTS}
                    -d ${DOCS_BUILD_DIR}/.doctrees
                    ${SPHINX_SOURCE_DIR}
                    ${DOCS_BUILD_DIR}/latex
                WORKING_DIRECTORY ${DOCS_SOURCE_DIR}
                COMMENT "Building LaTeX documentation with Sphinx"
                VERBATIM
            )
            
            if(BUILD_DOXYGEN)
                add_dependencies(sphinx-latex doxygen)
            endif()
        endif()
        
        message(STATUS "Sphinx HTML documentation will be built to: ${SPHINX_OUTPUT_DIR}")
    endif()
endif()

# Main documentation target (builds everything)
if(BUILD_DOXYGEN OR BUILD_SPHINX)
    add_custom_target(documentation ALL)
    
    if(BUILD_DOXYGEN)
        add_dependencies(documentation doxygen)
    endif()
    
    if(BUILD_SPHINX)
        add_dependencies(documentation sphinx-html)
        
        if(BUILD_SPHINX_LATEX)
            add_dependencies(documentation sphinx-latex)
        endif()
    endif()
    
    message(STATUS "Documentation will be built. Main target: documentation")
    message(STATUS "  - Doxygen: ${BUILD_DOXYGEN}")
    message(STATUS "  - Sphinx HTML: ${BUILD_SPHINX}")
    message(STATUS "  - Sphinx LaTeX: ${BUILD_SPHINX_LATEX}")
else()
    message(STATUS "No documentation targets enabled")
endif()

# Install documentation (optional)
if(BUILD_SPHINX AND EXISTS ${SPHINX_OUTPUT_DIR})
    install(DIRECTORY ${SPHINX_OUTPUT_DIR}/
        DESTINATION share/doc/acloudviewer
        COMPONENT documentation
        OPTIONAL
    )
endif()

# Clean target
add_custom_target(doc-clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${DOCS_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${DOXYGEN_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${DOCS_BUILD_DIR}/.doctrees
    COMMENT "Cleaning documentation build artifacts"
)

# Help message
add_custom_target(doc-help
    COMMAND ${CMAKE_COMMAND} -E echo "ACloudViewer Documentation Build Targets:"
    COMMAND ${CMAKE_COMMAND} -E echo "  documentation  - Build all documentation (default)"
    COMMAND ${CMAKE_COMMAND} -E echo "  doxygen        - Build C++ API documentation only"
    COMMAND ${CMAKE_COMMAND} -E echo "  sphinx-html    - Build Sphinx HTML documentation only"
    COMMAND ${CMAKE_COMMAND} -E echo "  sphinx-latex   - Build Sphinx LaTeX documentation (if enabled)"
    COMMAND ${CMAKE_COMMAND} -E echo "  doc-clean      - Clean documentation build artifacts"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Output directories:"
    COMMAND ${CMAKE_COMMAND} -E echo "  HTML:    ${SPHINX_OUTPUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "  Doxygen: ${DOXYGEN_OUTPUT_DIR}"
)

