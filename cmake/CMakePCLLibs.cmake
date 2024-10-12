# ------------------------------------------------------------------------------
# Qt
# -----------------------------------------------------------------------------

# see here: https://github.com/PointCloudLibrary/pcl/issues/3680
# when this is fixed, we can remove the following 3 lines.
if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

## we will use cmake automoc feature
# init qt
set(CMAKE_AUTOMOC ON) # for meta object compiler
# set(CMAKE_AUTORCC ON) # resource files
# set(CMAKE_AUTOUIC ON) # UI files

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(VTK_USE_PTHREADS 1)
find_package (PCL REQUIRED) # must before find_package (VTK REQUIRED), otherwise link errors.
if (NOT PCL_FOUND)
     message(FATAL_ERROR
     "External PCL not found but ACloudViewer requires PCL to function")
else ()
     message(STATUS "Found System installed PCL: " ${PCL_VERSION})
endif ()
find_package (VTK QUIET) # must quiet to avoid boost version issues
if (NOT VTK_FOUND)
     message(FATAL_ERROR
     "External VTK not found but ParaView requires VTK to function")
else ()
     message(STATUS "Found System installed VTK: " ${VTK_VERSION})
endif ()
include_directories( ${PCL_INCLUDE_DIRS} )
include_directories( ${VTK_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

set(PCL_FOUND ${PCL_FOUND} PARENT_SCOPE)
set(PCL_VERSION ${PCL_VERSION} PARENT_SCOPE)
set(VTK_FOUND ${VTK_FOUND} PARENT_SCOPE)
set(VTK_VERSION ${VTK_VERSION} PARENT_SCOPE)