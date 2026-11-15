cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

project( QPOISSON_RECON_LIB )

# The PoissonRecon engine and the in-memory PoissonReconLib wrapper are now
# provided by the unified 3rdparty/PoissonRecon integration (single vendored
# Adaptive Multigrid Solvers v18.75 tree shared with libs/CV_db and the
# reconstruction poisson_mesher). This target remains for the plugin's
# historical CMake contract and simply forwards to the shared wrapper.
add_library( ${PROJECT_NAME} INTERFACE )

target_link_libraries( ${PROJECT_NAME} INTERFACE poisson_recon_lib )
