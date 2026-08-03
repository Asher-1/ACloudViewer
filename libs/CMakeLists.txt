add_subdirectory( CV_db )
add_subdirectory( CV_io )

# Links <target> against all 3rdparty libraries.
# We need this because we create a lot of object libraries to assemble the main CloudViewer library.
function(cloudViewer_link_3rdparty_libraries target)
    cmake_parse_arguments(ARG "EXCLUDE_CUDA_STATIC" "" "" ${ARGN})

    set(_3rdparty_private_targets ${CloudViewer_3RDPARTY_PRIVATE_TARGETS})
    if(ARG_EXCLUDE_CUDA_STATIC)
        list(FILTER _3rdparty_private_targets EXCLUDE REGEX "^3rdparty_cublas$|^3rdparty_cuda_npp$")
    endif()

    # Directly pass public and private dependencies to the target.
    target_link_libraries(${target} PRIVATE ${_3rdparty_private_targets} CV_DB_LIB CV_IO_LIB)
    target_link_libraries(${target} PUBLIC ${CloudViewer_3RDPARTY_PUBLIC_TARGETS} CV_DB_LIB CV_IO_LIB)

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

cloudviewer_apply_dev_internal_lib_layout()
add_subdirectory( cloudViewer )

cloudviewer_apply_dev_cvviewer_lib_layout()
add_subdirectory( CVViewer )

if(BUILD_RECONSTRUCTION)
    add_subdirectory( Reconstruction )
endif()

# ECV PYTHON LIB — pybind output is lib/<Config>/Python/, never bin/.
if (BUILD_PYTHON_MODULE)
    add_subdirectory( Python )
endif()

cloudviewer_apply_dev_runtime_bin_layout()

# VTK ENGINE LIB
if (USE_VTK_BACKEND)
	add_subdirectory( VtkEngine ) # must always be included after CV_db and CV_io (dependency)
	set(QVTK_ENGINE_LIB_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/VtkEngine PARENT_SCOPE)
endif()

add_subdirectory( CVAppCommon )
add_subdirectory( CVPluginStub )    # Must come before CCPluginAPI
add_subdirectory( CVPluginAPI )
