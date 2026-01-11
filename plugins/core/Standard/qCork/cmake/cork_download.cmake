include(ExternalProject)

# ==============================================================================
# Cork library setup - Cross-platform compatible
# ==============================================================================
# Windows: Uses pre-built binaries with MPIR (no compilation needed)
# Linux/macOS: Builds from source using GMP from conda or system
# ==============================================================================

if(WIN32)
    # -------------------------------------------------------------------------
    # Windows Platform: Download pre-built binaries with MPIR
    # -------------------------------------------------------------------------
    # No compilation or GMP required on Windows
    # Uses MPIR instead of GMP (Windows-compatible alternative)
    ExternalProject_Add(ext_cork
        PREFIX cork
        URL https://github.com/Asher-1/cloudViewer_downloads/releases/download/1.9.0/cork.7z
        URL_HASH MD5=8014B0317BE35DE273A78780A688F49A
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/cork"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
    
    ExternalProject_Get_Property(ext_cork SOURCE_DIR)
    set(CORK_DIR ${SOURCE_DIR})
    message(STATUS "Cork pre-built binaries will be downloaded to: ${CORK_DIR}")
else()
    # -------------------------------------------------------------------------
    # Linux/macOS Platform: Build from source using GMP
    # -------------------------------------------------------------------------
    # When building with conda, search in conda environment first
    if(BUILD_WITH_CONDA AND DEFINED ENV{CONDA_PREFIX})
        set(CONDA_PREFIX $ENV{CONDA_PREFIX})
        message(STATUS "Searching for GMP in conda environment: ${CONDA_PREFIX}")
        
        # Set pkg-config path to search in conda environment first
        set(ENV{PKG_CONFIG_PATH} "${CONDA_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
        
        # Try to find GMP in conda environment
        find_library(GMP_LIBRARY
            NAMES gmp libgmp
            PATHS ${CONDA_PREFIX}/lib
            NO_DEFAULT_PATH
        )
        
        find_path(GMP_INCLUDE_DIR
            NAMES gmp.h
            PATHS ${CONDA_PREFIX}/include
            NO_DEFAULT_PATH
        )
        
        if(GMP_LIBRARY AND GMP_INCLUDE_DIR)
            set(GMP_FOUND TRUE)
            set(GMP_LIBRARIES ${GMP_LIBRARY})
            set(GMP_LIBRARY_DIRS ${CONDA_PREFIX}/lib)
            set(GMP_INCLUDE_DIRS ${GMP_INCLUDE_DIR})
            message(STATUS "Found GMP in conda environment")
            message(STATUS "  GMP library: ${GMP_LIBRARY}")
            message(STATUS "  GMP include: ${GMP_INCLUDE_DIR}")
        else()
            message(WARNING "GMP not found in conda environment, falling back to system search")
            set(GMP_FOUND FALSE)
        endif()
    endif()
    
    # If not found in conda or not using conda, use pkg-config to search system-wide
    if(NOT GMP_FOUND)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(GMP REQUIRED gmp)
        
        if(NOT GMP_FOUND)
            message(FATAL_ERROR "GMP library not found. Please install it using your package manager:\n"
                                "  Conda: conda install gmp\n"
                                "  Ubuntu/Debian: sudo apt-get install libgmp-dev\n"
                                "  Fedora/RHEL: sudo dnf install gmp-devel\n"
                                "  macOS: brew install gmp")
        endif()
    endif()
    
    message(STATUS "GMP version: ${GMP_VERSION}")
    message(STATUS "GMP libraries: ${GMP_LIBRARIES}")
    message(STATUS "GMP library dirs: ${GMP_LIBRARY_DIRS}")
    message(STATUS "GMP include dirs: ${GMP_INCLUDE_DIRS}")
    
    # Create a build script to ensure proper compilation
    set(CORK_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_cork.sh")
    
    # Determine appropriate compiler (use system default or CMAKE_CXX_COMPILER)
    get_filename_component(COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME)
    
    file(WRITE ${CORK_BUILD_SCRIPT}
"#!/bin/bash
set -e
cd \"$1\"
echo \"Building Cork library...\"
echo \"Using compiler: ${CMAKE_CXX_COMPILER}\"
echo \"GMP include: ${GMP_INCLUDE_DIRS}\"
echo \"GMP library dirs: ${GMP_LIBRARY_DIRS}\"

# Export GMP paths for Cork's Makefile (for safety)
export GMP_INC_DIR=\"${GMP_INCLUDE_DIRS}\"
export GMP_LIB_DIR=\"${GMP_LIBRARY_DIRS}\"

# Rewrite makeConstants file with correct GMP paths
# This file is included by Makefile and would override environment variables
echo \"Creating makeConstants with correct GMP paths...\"
cat > makeConstants << EOF
GMP_INC_DIR = ${GMP_INCLUDE_DIRS}
GMP_LIB_DIR = ${GMP_LIBRARY_DIRS}
EOF

# Fix GCC compatibility issue in mesh.topoCache.tpp
# Comment out problematic debug output that requires operator<< for TopoTri
if [ -f \"src/mesh/mesh.topoCache.tpp\" ]; then
    echo \"Patching mesh.topoCache.tpp for GCC compatibility...\"
    sed -i.bak '507s/^/\\/\\//' src/mesh/mesh.topoCache.tpp || true
    # Line 507: cout << \" \" << t << \": \" << *t << endl;
fi

# Patch Makefile for cross-platform compatibility
if [ -f \"Makefile\" ]; then
    echo \"Patching Cork Makefile...\"
    
    # Remove gmpxx dependency - Cork only needs gmp (not gmpxx)
    # This works on both Linux and macOS
    sed -i.bak 's/-lgmpxx -lgmp/-lgmp/g' Makefile
    
    # Ensure -fPIC is present in CCFLAGS (critical for linking with shared libraries)
    if ! grep -q "CCFLAGS.*-fPIC" Makefile; then
        echo \"Adding -fPIC to CCFLAGS...\"
        sed -i.bak_pic 's/^CCFLAGS[[:space:]]*:=/CCFLAGS := -fPIC /g' Makefile
    else
        echo \"-fPIC already present in CCFLAGS\"
    fi
    
    # Platform-specific compiler handling
    if [[ \"\$OSTYPE\" == \"linux-gnu\"* ]] || [[ \"\$OSTYPE\" == \"linux\" ]]; then
        # Linux: replace clang with g++ (more commonly available)
        echo \"Linux detected: replacing clang++ with g++ in Makefile...\"
        sed -i.bak2 's/clang++/g++/g' Makefile
        sed -i.bak3 's/clang/gcc/g' Makefile
    else
        # macOS: keep clang++ as is (system default compiler)
        echo \"macOS detected: keeping clang++ in Makefile...\"
    fi
fi

echo \"GMP_INC_DIR=\${GMP_INC_DIR}\"
echo \"GMP_LIB_DIR=\${GMP_LIB_DIR}\"

# Cork's Makefile uses GMP_INC_DIR and GMP_LIB_DIR from environment
# The Makefile already defines proper CXXFLAGS including GMP include paths
# We just need to run make - it will use the compilers defined in Makefile
make clean || true
make 2>&1

# Ensure lib directory exists and copy the library
mkdir -p lib
if [ -f \"lib/libcork.a\" ]; then
    echo \"Cork library found at lib/libcork.a\"
elif [ -f \"bin/libcork.a\" ]; then
    cp bin/libcork.a lib/
    echo \"Cork library copied from bin/ to lib/\"
else
    echo \"ERROR: Cork library not found after build!\"
    find . -name \"libcork.a\" -o -name \"*.a\" || true
    exit 1
fi
echo \"Cork build completed successfully\"
")
    
    ExternalProject_Add(ext_cork
        PREFIX cork
        GIT_REPOSITORY https://github.com/Asher-1/cork.git
        GIT_TAG master
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/cork"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND chmod +x ${CORK_BUILD_SCRIPT}
        BUILD_IN_SOURCE 1
        BUILD_COMMAND bash ${CORK_BUILD_SCRIPT} <SOURCE_DIR>
        INSTALL_COMMAND ""
    )
    
    ExternalProject_Get_Property(ext_cork SOURCE_DIR)
    
    # Set variables (no PARENT_SCOPE needed when using include())
    # SOURCE_DIR points to the extracted directory, we need to point to the actual source
    set(CORK_DIR ${SOURCE_DIR})
    set(CORK_LIB_DIR ${SOURCE_DIR}/lib)
    set(CORK_LIBRARY ${SOURCE_DIR}/lib/libcork.a)
    
    message(STATUS "Cork source directory: ${CORK_DIR}")
    message(STATUS "Cork library will be: ${CORK_LIBRARY}")
endif()