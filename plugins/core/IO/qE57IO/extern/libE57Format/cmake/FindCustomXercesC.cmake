include(FindPackageHandleStandardArgs)
# TODO XercesC_api && XercesC version from header

find_library(XercesC_LIBRARIES
        NAMES XercesC xerces-c
        HINTS /usr/lib
        /usr/local/lib
        $ENV{CONDA_PREFIX}/lib
        )

find_path(XercesC_INCLUDE_DIRS
        NAMES xercesc
        HINTS /usr/include
        /usr/local/include
        $ENV{CONDA_PREFIX}/include
        )
message(DEBUG "XercesC_INCLUDE_DIRS: ${XercesC_INCLUDE_DIRS}")
message(DEBUG "XercesC_LIBRARIES: ${XercesC_LIBRARIES}")

find_package_handle_standard_args(XercesC
        REQUIRED_VARS XercesC_LIBRARIES XercesC_INCLUDE_DIRS
        HANDLE_COMPONENTS
        )

if (XercesC_FOUND)
    mark_as_advanced(XercesC_LIBRARIES XercesC_INCLUDE_DIRS)
endif ()

if (XercesC_FOUND AND NOT TARGET XercesC::XercesC)
    add_library(XercesC::XercesC SHARED IMPORTED)
    set_target_properties(XercesC::XercesC PROPERTIES
                        IMPORTED_LOCATION ${XercesC_LIBRARIES})
    target_include_directories(XercesC::XercesC INTERFACE ${XercesC_INCLUDE_DIRS})
endif ()
