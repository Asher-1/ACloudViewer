add_subdirectory( eCV_db )
add_subdirectory( eCV_io )

# Convenience function to link against all third-party libraries
# We need this because we create a lot of object libraries to assemble
# the main ErowCloudViewer library
function(cloudViewer_link_3rdparty_libraries target)
    target_link_libraries(${target} PRIVATE ${CloudViewer_3RDPARTY_PRIVATE_TARGETS} ECV_DB_LIB ECV_IO_LIB)
    target_link_libraries(${target} PUBLIC ${CloudViewer_3RDPARTY_PUBLIC_TARGETS} ECV_DB_LIB ECV_IO_LIB)
    foreach(dep IN LISTS CloudViewer_3RDPARTY_HEADER_TARGETS)
        if(TARGET ${dep})
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            if(inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${inc})
            endif()
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
            if(inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${inc})
            endif()
            get_property(def TARGET ${dep} PROPERTY INTERFACE_COMPILE_DEFINITIONS)
            if(def)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ${def})
            endif()
        endif()
    endforeach()
endfunction()

add_subdirectory( OpenGLEngine )

add_subdirectory( CVViewer )

if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
	if( MSVC )
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo)
	else()
		SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
	endif()
endif()

# ECV PYTHON LIB
if (BUILD_PYTHON_MODULE)
    add_subdirectory( Python )
endif()

# PCL ENGINE LIB
if (USE_PCL_BACKEND) 
	include( CMakePCLLibs )
	add_subdirectory( PCLEngine ) # must always be included after eCV_db and eCV_io (dependency)
endif()

add_subdirectory( CVAppCommon)
add_subdirectory( CVPluginStub )    # Must come before CCPluginAPI
add_subdirectory( CVPluginAPI )
