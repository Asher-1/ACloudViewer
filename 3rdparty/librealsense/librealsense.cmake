include(ExternalProject)

find_package(Git QUIET REQUIRED)

set(PATCH_CUDACRT_COMMAND "")
if(WIN32)
  set(PATCH_CUDACRT_COMMAND 
    ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace ${CMAKE_CURRENT_LIST_DIR}/fix-cudacrt.patch
  )
endif()

set(PATCH_MACOS_ARM64_COMMAND "")
if(APPLE)
  set(PATCH_MACOS_ARM64_COMMAND 
    ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace --quiet ${CMAKE_CURRENT_LIST_DIR}/fix-macos-arm64.patch
  )
endif()

ExternalProject_Add(
    ext_librealsense
    PREFIX librealsense
    URL https://github.com/IntelRealSense/librealsense/archive/refs/tags/v2.44.0.tar.gz #  2020 Apr 1
    URL_HASH SHA256=5b0158592646984f0f7348da3783e2fb49e99308a97f2348fe3cc82c770c6dde
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/librealsense"
    BUILD_ALWAYS 0
    # patch for macOS ARM64 unsupported options: -mfpu=neon -mfloat-abi=hard
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        ${CloudViewer_3RDPARTY_DIR}/librealsense/unix_config.cmake 
        <SOURCE_DIR>/CMake/unix_config.cmake
    # Patch for libusb static build failure on Linux
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CloudViewer_3RDPARTY_DIR}/librealsense/libusb-CMakeLists.txt
        <SOURCE_DIR>/third-party/libusb/CMakeLists.txt
    COMMAND ${GIT_EXECUTABLE} init
    # Patch for CRT mismatch in CUDA code (Windows)
    COMMAND ${PATCH_CUDACRT_COMMAND}
    # Patch for macOS ARM64 support for versions < 2.50.0
    COMMAND ${PATCH_MACOS_ARM64_COMMAND}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_UNIT_TESTS=OFF
        -DBUILD_GLSL_EXTENSIONS=OFF
        -DBUILD_GRAPHICAL_EXAMPLES=OFF
        -DBUILD_PYTHON_BINDINGS=OFF
        -DBUILD_WITH_CUDA=${BUILD_CUDA_MODULE}
        -DUSE_EXTERNAL_USB=ON
        # Syncing GLIBCXX_USE_CXX11_ABI for MSVC causes problems, but directly
        # checking CXX_COMPILER_ID is not supported.
        $<IF:$<PLATFORM_ID:Windows>,"",-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=$<BOOL:${GLIBCXX_USE_CXX11_ABI}>>
        $<$<PLATFORM_ID:Darwin>:-DBUILD_WITH_OPENMP=OFF>
        $<$<PLATFORM_ID:Darwin>:-DHWM_OVER_XU=OFF>
        $<$<PLATFORM_ID:Windows>:-DBUILD_WITH_STATIC_CRT=${STATIC_WINDOWS_RUNTIME}>
        ${ExternalProject_CMAKE_ARGS_hidden}
    CMAKE_CACHE_ARGS    # Lists must be passed via CMAKE_CACHE_ARGS
        -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}realsense2${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}realsense-file${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}fw${CMAKE_STATIC_LIBRARY_SUFFIX}
)


ExternalProject_Get_Property(ext_librealsense INSTALL_DIR)
set(LIBREALSENSE_INCLUDE_DIR "${INSTALL_DIR}/include/") # "/" is critical.
set(LIBREALSENSE_LIB_DIR "${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR}")

set(LIBREALSENSE_LIBRARIES realsense2 fw realsense-file usb) # The order is critical.
if(MSVC)    # Rename debug libs to ${LIBREALSENSE_LIBRARIES}. rem (comment) is no-op
    ExternalProject_Add_Step(ext_librealsense rename_debug_libs
        COMMAND $<IF:$<CONFIG:Debug>,move,rem> /Y realsense2d.lib realsense2.lib
        COMMAND $<IF:$<CONFIG:Debug>,move,rem> /Y fwd.lib fw.lib
        COMMAND $<IF:$<CONFIG:Debug>,move,rem> /Y realsense-filed.lib realsense-file.lib
        WORKING_DIRECTORY "${LIBREALSENSE_LIB_DIR}"
        DEPENDEES install
    )
endif()

ExternalProject_Add_Step(ext_librealsense copy_libusb_to_lib_folder
    COMMAND ${CMAKE_COMMAND} -E copy
    "<BINARY_DIR>/libusb_install/lib/${CMAKE_STATIC_LIBRARY_PREFIX}usb${CMAKE_STATIC_LIBRARY_SUFFIX}"
    "${LIBREALSENSE_LIB_DIR}"
    DEPENDEES install
    BYPRODUCTS "${LIBREALSENSE_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}usb${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )
