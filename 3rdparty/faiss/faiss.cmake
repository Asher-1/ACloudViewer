include(FetchContent)

# Match COLMAP's CPU-only FAISS configuration.  It is intentionally kept out
# of the global dependency lists: only reconstruction's descriptor index and
# visual vocabulary consumers need its headers and static library.
FetchContent_Declare(
    faiss
    URL https://github.com/facebookresearch/faiss/archive/refs/tags/v1.14.1.zip
    URL_HASH SHA256=4b1ae7e7a0a46385b4084f0e3945623a15fcf99d793bf44d82aae8e24f11e5f5
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/faiss"
    PATCH_COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=<SOURCE_DIR>
        -P ${CMAKE_CURRENT_LIST_DIR}/patch_cmake_minimum.cmake
)

if (NOT MSVC)
    set(FAISS_OPT_LEVEL dd CACHE STRING "" FORCE)
endif ()
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_MKL OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_EXTRAS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(faiss)

if (WIN32)
    target_compile_definitions(faiss PUBLIC FAISS_MAIN_LIB)
endif ()

add_library(3rdparty_faiss INTERFACE)
target_link_libraries(3rdparty_faiss INTERFACE faiss)
