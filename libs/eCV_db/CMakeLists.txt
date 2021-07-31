project( ECV_DB_LIB )

add_library(${PROJECT_NAME} SHARED )

add_subdirectory( include )
add_subdirectory( src )

if (MSVC)
    if (CVCORELIB_USE_TBB)
        include_directories( ${TBB_INCLUDE_DIRS})
    endif()
endif()

# INTERNAL LIB
target_link_libraries( ${PROJECT_NAME} PUBLIC CVCoreLib )
# Qt
target_link_libraries(${PROJECT_NAME} PUBLIC Qt5::OpenGL)

# Enforce 3rd party dependencies
if(USE_SIMD)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${CLOUDVIEWERCONFIG_SSE_DEFINITIONS})
    target_compile_options(${PROJECT_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CLOUDVIEWERCONFIG_SSE_COMPILE_OPTIONS}>")
endif()
cloudViewer_link_3rdparty_libraries(${PROJECT_NAME})

option( OPTION_USE_RANSAC_LIB "Build with Ransac Lib" ON )
if ( ${OPTION_USE_RANSAC_LIB} )
	#target_link_libraries( ${PROJECT_NAME} PRIVATE 3rdparty_ransacSD )
	target_compile_definitions( ${PROJECT_NAME} PUBLIC CV_RANSAC_SUPPORT POINTSWITHINDEX)
endif()

# Add custom preprocessor definitions
target_compile_definitions( ${PROJECT_NAME} PRIVATE ECV_DB_LIBRARY_BUILD )

CopyDLLToDirectory(${PROJECT_NAME})

set_target_properties( ${PROJECT_NAME} PROPERTIES
	CXX_VISIBILITY_PRESET hidden
)

# install (shared) lib to specified destinations
InstallSharedLibrary( TARGET ${PROJECT_NAME} )

include( cmake/InstallCGALDependencies.cmake )
