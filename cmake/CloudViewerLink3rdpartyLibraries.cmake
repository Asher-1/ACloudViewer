# cloudViewer_link_3rdparty_libraries(target)
#
# Links <target> against all 3rdparty libraries.
# We need this because we create a lot of object libraries to assemble the main CloudViewer library.
function(cloudViewer_link_3rdparty_libraries target)
    # Directly pass public and private dependencies to the target.
    target_link_libraries(${target} PRIVATE ${CloudViewer_3RDPARTY_PRIVATE_TARGETS})
    target_link_libraries(${target} PUBLIC ${CloudViewer_3RDPARTY_PUBLIC_TARGETS})

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

function(cloudViewer_link_static_lib target dependency)
    target_include_directories(${target} PUBLIC ${OpenCV_INCLUDE_DIRS})
    if (APPLE)
        set_target_properties(${target} PROPERTIES
            LINK_FLAGS "-Wl,-force_load,${OpenCV_LIB_DIR}/lib${OpenCV_LIBS}.a"
        )
        # set_target_properties(${target} PROPERTIES
        #     LINK_FLAGS "-Wl,-ObjC"
        # )
        # target_link_libraries(${target} ${dependency})
    elseif (UNIX)
        # Directly pass public and private dependencies to the target.
        set_target_properties(${target} PROPERTIES
            LINK_FLAGS "-Wl,--whole-archive -Wl,--start-group"
        )
        target_link_libraries(${target} ${dependency})
        set_property(TARGET ${target} APPEND_STRING PROPERTY
            LINK_FLAGS " -Wl,--end-group"
        )
    elseif (WIN32)
        set_target_properties(${target} PROPERTIES
            LINK_FLAGS "/WHOLEARCHIVE"
        )
        target_link_libraries(${target} ${dependency})
    endif()

    set_target_properties(${target} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN 1
    )
endfunction()