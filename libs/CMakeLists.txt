add_subdirectory( eCV_db )
add_subdirectory( eCV_io )

# Links <target> against all 3rdparty libraries.
# We need this because we create a lot of object libraries to assemble the main CloudViewer library.
function(cloudViewer_link_3rdparty_libraries target)
    # Directly pass public and private dependencies to the target.
    target_link_libraries(${target} PRIVATE ${CloudViewer_3RDPARTY_PRIVATE_TARGETS} ECV_DB_LIB ECV_IO_LIB)
    target_link_libraries(${target} PUBLIC ${CloudViewer_3RDPARTY_PUBLIC_TARGETS} ECV_DB_LIB ECV_IO_LIB)

    if(USE_SIMD)
        target_compile_definitions(${target} PRIVATE ${CLOUDVIEWERCONFIG_SSE_DEFINITIONS})
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CLOUDVIEWERCONFIG_SSE_COMPILE_OPTIONS}>")
    endif()

    # Propagate interface properties of header dependencies to target.
    foreach(dep IN LISTS CloudViewer_3RDPARTY_HEADER_TARGETS)
        if(TARGET ${dep})
            foreach(prop IN ITEMS
                INTERFACE_INCLUDE_DIRECTORIES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                INTERFACE_COMPILE_DEFINITIONS
            )
                get_property(prop_value TARGET ${dep} PROPERTY ${prop})
                if(prop_value)
                    set_property(TARGET ${target} APPEND PROPERTY ${prop} ${prop_value})
                endif()
            endforeach()
        else()
            message(WARNING "Skipping non-existent header dependency ${dep}")
        endif()
    endforeach()

    # Link header dependencies privately.
    target_link_libraries(${target} PRIVATE ${CloudViewer_3RDPARTY_HEADER_TARGETS})
endfunction()

if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
	if( MSVC )
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
		SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo)
	else()
		SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
	endif()
endif()

add_subdirectory( cloudViewer )

add_subdirectory( CVViewer )

if(BUILD_RECONSTRUCTION)
    add_subdirectory( Reconstruction )
endif()

# ECV PYTHON LIB
if (BUILD_PYTHON_MODULE)
    add_subdirectory( Python )
endif()

if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
    if (MSVC)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo)
    else ()
        SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif ()
endif ()

# PCL ENGINE LIB
if (USE_PCL_BACKEND)
    include( CMakePCLLibs )
	add_subdirectory( PCLEngine ) # must always be included after eCV_db and eCV_io (dependency)
endif()

add_subdirectory( CVAppCommon )
add_subdirectory( CVPluginStub )    # Must come before CCPluginAPI
add_subdirectory( CVPluginAPI )
