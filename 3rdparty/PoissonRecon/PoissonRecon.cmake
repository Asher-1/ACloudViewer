include(ExternalProject)

# Byproducts the Ninja graph can trace back to the ExternalProject's install
# step. Declared before ExternalProject_Add so it can reference the same
# expressions; per-config MSVC names follow the solver's DEBUG_POSTFIX "d".
if(NOT MSVC)
    set(POISSON_BUILD_BYPRODUCTS
        "${CMAKE_BINARY_DIR}/poisson/install/lib/libpoisson_recon.a")
else()
    set(POISSON_BUILD_BYPRODUCTS
        "${CMAKE_BINARY_DIR}/poisson/install/lib/poisson_recon.lib"
        "${CMAKE_BINARY_DIR}/poisson/install/lib/poisson_recond.lib")
endif()

# Unified PoissonRecon dependency: Adaptive Multigrid Solvers v18.75, vendored
# verbatim from upstream COLMAP's in-tree copy (colmap@88dc019465be,
# src/thirdparty/PoissonRecon). One source tree serves every downstream module:
#   - libs/Reconstruction/src/mvs/poisson_meshing.cc   -> RunPoissonRecon /
#     RunSurfaceTrimmer CLI-style entries (byte-parity path with upstream COLMAP)
#   - libs/CV_db/src/SurfaceReconstructionPoisson.cpp  -> PoissonReconLib wrapper
#   - plugins/core/Standard/qPoissonRecon              -> PoissonReconLib wrapper
# Replaces the previous split (Reconstruction's in-tree v8.0 flat tree +
# Open3D-PoissonRecon v12.00 ExternalProject).
ExternalProject_Add(ext_poisson
    PREFIX poisson
    URL https://raw.githubusercontent.com/Asher-1/cloudViewer_downloads/master/poisson/PoissonRecon-18.75-colmap-88dc0194.tar.gz
    URL_HASH SHA256=14b8d564becef8c4e3be384f4064adb02749024f335010f57cddb24af9da126d
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/poisson"
    INSTALL_DIR "${CMAKE_BINARY_DIR}/poisson/install"
    SOURCE_DIR "poisson/src/ext_poisson/PoissonRecon" # Add extra directory level for POISSON_INCLUDE_DIRS.
    # The vendored tree carries COLMAP-macro CMakeLists.txt; inject the
    # standalone solver recipe and build/install it once for all consumers.
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different
                  ${CMAKE_CURRENT_LIST_DIR}/solver.CMakeLists.txt
                  <SOURCE_DIR>/CMakeLists.txt
    # The solver recipe is an isolated CMake instance (not a subdirectory of
    # the main project), so the standard must be forwarded explicitly: without
    # it AppleClang falls back to a pre-C++11 default and MyExceptions.h's
    # `noexcept` overrides fail to compile (macOS runs 39021395360/39021395394).
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_CXX_STANDARD=17
               -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config $<CONFIG>
    INSTALL_COMMAND ${CMAKE_COMMAND} --install . --config $<CONFIG>
    # Ninja-family generators (the CI Ninja-Multi-Config graph check, the
    # Windows NMC build) require every consumed file to have a producing rule;
    # Make is lax, which is why local Unix-Makefiles trees never exposed the
    # missing declaration (runs 39021395507/39021395388: "missing and no known
    # rule to make it: .../poisson/install/lib/libpoisson_recon.a").
    BUILD_BYPRODUCTS ${POISSON_BUILD_BYPRODUCTS}
)
ExternalProject_Get_Property(ext_poisson SOURCE_DIR)
ExternalProject_Get_Property(ext_poisson INSTALL_DIR)
set(POISSON_INCLUDE_DIRS ${SOURCE_DIR}) # Not using "/" is critical.

# Solver objects (RunPoissonRecon/RunSurfaceTrimmer CLI entries). IMPORTED
# because the sources only exist after the ExternalProject download/build.
# The MSVC Debug name follows the solver recipe's DEBUG_POSTFIX "d". On MSVC
# the location is expressed through per-config IMPORTED_LOCATION_<CONFIG>
# properties: a $<CONFIG:Debug> genex inside the generic IMPORTED_LOCATION is
# NOT evaluated by the Ninja Multi-Config graph and leaks literally into the
# link inputs ("missing and no known rule to make it: poisson/install/lib/
# poisson_recon$<$<CONFIG:Debug>:d>.lib", run 39025562222).
if(NOT MSVC)
    set(POISSON_STATIC_LIBRARY "${INSTALL_DIR}/lib/libpoisson_recon.a")
    set(POISSON_IMPORTED_LOCATION_PROPS
        IMPORTED_LOCATION "${POISSON_STATIC_LIBRARY}")
else()
    set(POISSON_STATIC_LIBRARY "${INSTALL_DIR}/lib/poisson_recon.lib")
    set(POISSON_IMPORTED_LOCATION_PROPS
        IMPORTED_LOCATION "${POISSON_STATIC_LIBRARY}"
        IMPORTED_LOCATION_RELEASE "${INSTALL_DIR}/lib/poisson_recon.lib"
        IMPORTED_LOCATION_DEBUG "${INSTALL_DIR}/lib/poisson_recond.lib")
endif()
add_library(poisson_recon STATIC IMPORTED GLOBAL)
add_dependencies(poisson_recon ext_poisson)
set_target_properties(poisson_recon PROPERTIES
    ${POISSON_IMPORTED_LOCATION_PROPS}
    INTERFACE_INCLUDE_DIRECTORIES "${POISSON_INCLUDE_DIRS}"
)

# The in-memory wrapper shared by CV_db and the qPoissonRecon plugin. It only
# instantiates the header-defined Reconstructor::Implicit facade, so it needs
# the include dirs but not the solver objects. Compiled without OpenMP flags:
# the wrapper TU drives the solver through ThreadPool's ASYNC backend, which is
# also the backend COLMAP falls back to when OpenMP is absent.
add_library(poisson_recon_lib STATIC
    ${CMAKE_CURRENT_LIST_DIR}/PoissonReconLib/PoissonReconLib.cpp
)
add_dependencies(poisson_recon_lib ext_poisson)
target_include_directories(poisson_recon_lib
    PUBLIC
        $<BUILD_INTERFACE:${POISSON_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/PoissonReconLib>
)
target_compile_definitions(poisson_recon_lib PRIVATE RELEASE)
if(MSVC)
    target_compile_options(poisson_recon_lib PRIVATE /bigobj)
endif()
