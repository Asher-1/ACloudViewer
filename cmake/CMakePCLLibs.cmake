# ------------------------------------------------------------------------------
# Qt
# -----------------------------------------------------------------------------
## we will use cmake automoc feature
# init qt
set(CMAKE_AUTOMOC ON) # for meta object compiler
# set(CMAKE_AUTORCC ON) # resource files
# set(CMAKE_AUTOUIC ON) # UI files

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(VTK_USE_PTHREADS 1)
find_package (VTK REQUIRED)
find_package (PCL REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} )
include_directories( ${VTK_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )
