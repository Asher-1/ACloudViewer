if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
    add_definitions(-DLZ4_DLL_EXPORT)
endif()

if (UNIX AND NOT APPLE)
    # fix issues about imported target "FLANN::FLANN" with non-exists include path
    COLMAP_ADD_STATIC_LIBRARY(flann
        ${CMAKE_CURRENT_LIST_DIR}/flann.hpp ${CMAKE_CURRENT_LIST_DIR}/flann.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ext/lz4.h ${CMAKE_CURRENT_LIST_DIR}/ext/lz4.c
        ${CMAKE_CURRENT_LIST_DIR}/ext/lz4hc.h ${CMAKE_CURRENT_LIST_DIR}/ext/lz4hc.c
    )
else ()
    COLMAP_ADD_STATIC_LIBRARY(lz4
        ${CMAKE_CURRENT_LIST_DIR}/ext/lz4.h ${CMAKE_CURRENT_LIST_DIR}/ext/lz4.c
        ${CMAKE_CURRENT_LIST_DIR}/ext/lz4hc.h ${CMAKE_CURRENT_LIST_DIR}/ext/lz4hc.c
    )
    COLMAP_ADD_STATIC_LIBRARY(flann
        flann.hpp flann.cpp
    )
    target_link_libraries(flann PUBLIC lz4)
endif()
