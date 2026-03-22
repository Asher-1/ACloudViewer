cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option(PLUGIN_STANDARD_QSIBR "Install qSIBR (SIBR Image-Based Rendering) plugin" OFF)

if(PLUGIN_STANDARD_QSIBR)
    project(QSIBR_PLUGIN)

    # =============================================
    # Plugin shared library (full SIBR integration)
    # =============================================
    AddPlugin(NAME ${PROJECT_NAME})
    add_subdirectory(include)
    add_subdirectory(src)

    # =============================================
    # SIBR build configuration
    # =============================================
    set(SIBR_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SIBR")
    set(SIBR_3RDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")

    # ----- ACloudViewer 3rdparty dependencies -----

    # OpenGL (use project-level target; supports headless/OSMesa)
    if(TARGET 3rdparty_opengl)
        set(SIBR_OPENGL_LIBS 3rdparty_opengl)
    else()
        message(FATAL_ERROR "qSIBR requires 3rdparty_opengl target from the project")
    endif()

    # OpenMP
    if(TARGET 3rdparty_openmp)
        set(SIBR_OPENMP_LIBS 3rdparty_openmp)
    elseif(TARGET OpenMP::OpenMP_CXX)
        set(SIBR_OPENMP_LIBS OpenMP::OpenMP_CXX)
    else()
        set(SIBR_OPENMP_LIBS "")
    endif()

    # Boost
    if(TARGET 3rdparty_boost)
        set(SIBR_BOOST_LIBS 3rdparty_boost)
    else()
        find_package(Boost REQUIRED COMPONENTS system chrono filesystem date_time)
        set(SIBR_BOOST_LIBS ${Boost_LIBRARIES})
    endif()

    # OpenCV
    if(TARGET 3rdparty_opencv)
        set(SIBR_OPENCV_LIBS 3rdparty_opencv)
    else()
        find_package(OpenCV REQUIRED)
        set(SIBR_OPENCV_LIBS ${OpenCV_LIBRARIES})
    endif()

    # GLEW
    if(TARGET 3rdparty_glew)
        set(SIBR_GLEW_LIBS 3rdparty_glew)
    else()
        find_package(GLEW REQUIRED)
        set(SIBR_GLEW_LIBS GLEW::GLEW)
    endif()

    # GLFW
    if(TARGET 3rdparty_glfw)
        set(SIBR_GLFW_LIBS 3rdparty_glfw)
    elseif(TARGET glfw)
        set(SIBR_GLFW_LIBS glfw)
    else()
        find_package(glfw3 REQUIRED)
        set(SIBR_GLFW_LIBS glfw)
    endif()

    # Assimp
    if(TARGET 3rdparty_assimp)
        set(SIBR_ASSIMP_LIBS 3rdparty_assimp)
    else()
        find_package(assimp REQUIRED)
        set(SIBR_ASSIMP_LIBS assimp::assimp)
    endif()

    # zlib: 3rdparty_zlib uses HIDDEN (--exclude-libs) which hides minizip
    # symbols needed by assimp. Extract the raw library path for direct linking.
    if(TARGET 3rdparty_zlib)
        set(SIBR_ZLIB_LIBS 3rdparty_zlib)
        get_target_property(_zlib_libs 3rdparty_zlib INTERFACE_LINK_LIBRARIES)
        set(SIBR_ZLIB_RAW "")
        if(_zlib_libs)
            foreach(_zl ${_zlib_libs})
                if(_zl MATCHES "libz\\." OR _zl MATCHES "zlib")
                    list(APPEND SIBR_ZLIB_RAW "${_zl}")
                endif()
            endforeach()
        endif()
        if(NOT SIBR_ZLIB_RAW)
            find_package(ZLIB QUIET)
            if(ZLIB_FOUND)
                set(SIBR_ZLIB_RAW ZLIB::ZLIB)
            else()
                message(WARNING "qSIBR: Could not extract raw zlib path from 3rdparty_zlib; linking may fail")
                set(SIBR_ZLIB_RAW "")
            endif()
        endif()
    elseif(TARGET ZLIB::ZLIB)
        set(SIBR_ZLIB_LIBS ZLIB::ZLIB)
        set(SIBR_ZLIB_RAW ZLIB::ZLIB)
    else()
        find_package(ZLIB QUIET)
        if(ZLIB_FOUND)
            set(SIBR_ZLIB_LIBS ZLIB::ZLIB)
            set(SIBR_ZLIB_RAW ZLIB::ZLIB)
        else()
            set(SIBR_ZLIB_LIBS "")
            set(SIBR_ZLIB_RAW "")
        endif()
    endif()

    # Embree
    if(TARGET 3rdparty_embree)
        set(SIBR_EMBREE_LIBS 3rdparty_embree)
    else()
        find_package(embree 3 QUIET)
        if(embree_FOUND)
            set(SIBR_EMBREE_LIBS embree)
        else()
            message(WARNING "Embree3 not found - SIBR raycaster will have limited functionality")
            set(SIBR_EMBREE_LIBS "")
        endif()
    endif()

    # Eigen3
    if(TARGET 3rdparty_eigen3)
        set(SIBR_EIGEN_INCLUDE "")
        set(SIBR_EIGEN_TARGET 3rdparty_eigen3)
    else()
        find_package(Eigen3 REQUIRED)
        set(SIBR_EIGEN_INCLUDE ${EIGEN3_INCLUDE_DIR})
        set(SIBR_EIGEN_TARGET Eigen3::Eigen)
    endif()

    # Nanoflann
    if(TARGET 3rdparty_nanoflann)
        set(SIBR_NANOFLANN_TARGET 3rdparty_nanoflann)
    else()
        set(SIBR_NANOFLANN_TARGET "")
    endif()

    # tinyfiledialogs (replaces nativefiledialog for cross-platform file picker)
    if(TARGET 3rdparty_tinyfiledialogs)
        set(SIBR_FILEDIALOG_LIBS 3rdparty_tinyfiledialogs)
    else()
        message(FATAL_ERROR "qSIBR requires 3rdparty_tinyfiledialogs target from the project")
    endif()

    # SIBR uses its own imgui version with custom GLFW3/GL3 backend.
    set(SIBR_IMGUI_DIR "${SIBR_3RDPARTY_DIR}/imgui")
    add_library(sibr_imgui STATIC
        "${SIBR_IMGUI_DIR}/imgui.cpp"
        "${SIBR_IMGUI_DIR}/imgui_draw.cpp"
        "${SIBR_IMGUI_DIR}/imgui_demo.cpp"
        "${SIBR_IMGUI_DIR}/examples/opengl3_example/imgui_impl_glfw_gl3.cpp"
    )
    target_include_directories(sibr_imgui PUBLIC
        "${SIBR_3RDPARTY_DIR}"
        "${SIBR_IMGUI_DIR}"
        "${SIBR_IMGUI_DIR}/examples/opengl3_example"
    )
    target_link_libraries(sibr_imgui PUBLIC
        ${SIBR_GLEW_LIBS} ${SIBR_GLFW_LIBS} ${SIBR_OPENGL_LIBS}
    )
    if(NOT MSVC)
        target_compile_options(sibr_imgui PRIVATE -fpermissive -Wno-error)
    endif()
    set_target_properties(sibr_imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set(SIBR_IMGUI_LIBS sibr_imgui)
    set(SIBR_IMGUI_INCLUDE "${SIBR_3RDPARTY_DIR}" "${SIBR_IMGUI_DIR}" "${SIBR_IMGUI_DIR}/examples/opengl3_example")

    # FFmpeg (optional, for video module)
    find_package(FFmpeg QUIET COMPONENTS avcodec avformat avutil swscale)
    if(NOT FFmpeg_FOUND)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(FFMPEG QUIET libavcodec libavformat libavutil libswscale)
        endif()
    endif()

    # CUDA for Gaussian Splatting (use global BUILD_CUDA_MODULE)
    set(SIBR_HAS_CUDA OFF)
    if(BUILD_CUDA_MODULE AND TARGET CUDA::cudart)
        set(SIBR_HAS_CUDA ON)
        message(STATUS "qSIBR: CUDA available via BUILD_CUDA_MODULE")
    endif()

    # =============================================
    # Embedded 3rd party libraries
    # =============================================

    # rapidxml (header-only)
    add_library(sibr_rapidxml INTERFACE)
    target_include_directories(sibr_rapidxml INTERFACE "${SIBR_3RDPARTY_DIR}")

    # picojson (header-only)
    add_library(sibr_picojson INTERFACE)
    target_include_directories(sibr_picojson INTERFACE "${SIBR_3RDPARTY_DIR}")

    # xatlas
    add_library(sibr_xatlas STATIC
        "${SIBR_3RDPARTY_DIR}/xatlas/source/xatlas/xatlas.cpp"
    )
    target_include_directories(sibr_xatlas PUBLIC
        "${SIBR_3RDPARTY_DIR}/xatlas/source/xatlas"
    )
    set_target_properties(sibr_xatlas PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # mrf (graph cut / maxflow)
    add_library(sibr_mrf STATIC
        "${SIBR_3RDPARTY_DIR}/mrf/graph.cpp"
        "${SIBR_3RDPARTY_DIR}/mrf/maxflow.cpp"
    )
    target_include_directories(sibr_mrf PUBLIC "${SIBR_3RDPARTY_DIR}/mrf")
    set_target_properties(sibr_mrf PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # =============================================
    # Common include directories for SIBR
    # =============================================
    set(SIBR_COMMON_INCLUDES
        "${SIBR_SRC_DIR}"
        "${SIBR_EIGEN_INCLUDE}"
        ${SIBR_IMGUI_INCLUDE}
    )

    # Helper macro: configure a SIBR static library
    macro(sibr_add_library TARGET_NAME SRC_DIR)
        file(GLOB _sibr_src
            "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.h" "${SRC_DIR}/*.hpp"
            "${SRC_DIR}/shaders/*.frag" "${SRC_DIR}/shaders/*.vert" "${SRC_DIR}/shaders/*.geom"
            "${SRC_DIR}/shaders/*.fp" "${SRC_DIR}/shaders/*.vp" "${SRC_DIR}/shaders/*.gp"
            "${SRC_DIR}/interface/*.cpp" "${SRC_DIR}/interface/*.h"
        )
        add_library(${TARGET_NAME} STATIC ${_sibr_src})
        target_include_directories(${TARGET_NAME} PUBLIC ${SIBR_COMMON_INCLUDES})
        target_compile_definitions(${TARGET_NAME} PUBLIC
            BOOST_ALL_DYN_LINK
            EIGEN_INITIALIZE_MATRICES_BY_ZERO
        )
        if(NOT MSVC)
            target_compile_options(${TARGET_NAME} PRIVATE -fpermissive -Wno-error)
        endif()
        set_target_properties(${TARGET_NAME} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            FOLDER "plugins/qSIBR/core"
        )
        if(SIBR_EIGEN_TARGET)
            target_link_libraries(${TARGET_NAME} PUBLIC ${SIBR_EIGEN_TARGET})
        endif()
    endmacro()

    # =============================================
    # SIBR Core Libraries (STATIC)
    # =============================================

    # sibr_system (base library)
    sibr_add_library(sibr_system "${SIBR_SRC_DIR}/core/system")
    target_link_libraries(sibr_system PUBLIC
        ${SIBR_BOOST_LIBS}
        ${SIBR_FILEDIALOG_LIBS}
        sibr_picojson
        sibr_rapidxml
    )

    # sibr_graphics
    sibr_add_library(sibr_graphics "${SIBR_SRC_DIR}/core/graphics")
    target_link_libraries(sibr_graphics PUBLIC
        sibr_system
        ${SIBR_BOOST_LIBS}
        ${SIBR_ASSIMP_LIBS}
        ${SIBR_GLEW_LIBS}
        ${SIBR_OPENGL_LIBS}
        ${SIBR_OPENCV_LIBS}
        ${SIBR_IMGUI_LIBS}
        ${SIBR_GLFW_LIBS}
        ${SIBR_OPENMP_LIBS}
    )
    if(UNIX AND NOT APPLE)
        target_link_libraries(sibr_graphics PUBLIC
            rt m dl X11 pthread Xrandr Xinerama Xxf86vm Xcursor
        )
    elseif(WIN32)
        target_link_libraries(sibr_graphics PUBLIC ws2_32 shlwapi)
    endif()

    # sibr_assets
    sibr_add_library(sibr_assets "${SIBR_SRC_DIR}/core/assets")
    target_link_libraries(sibr_assets PUBLIC
        sibr_graphics
        sibr_xatlas
        ${SIBR_BOOST_LIBS}
        ${SIBR_ASSIMP_LIBS}
        ${SIBR_GLEW_LIBS}
        ${SIBR_OPENGL_LIBS}
        ${SIBR_OPENCV_LIBS}
    )

    # sibr_raycaster
    sibr_add_library(sibr_raycaster "${SIBR_SRC_DIR}/core/raycaster")
    target_link_libraries(sibr_raycaster PUBLIC
        sibr_graphics
        sibr_assets
        ${SIBR_EMBREE_LIBS}
    )
    if(SIBR_NANOFLANN_TARGET)
        target_link_libraries(sibr_raycaster PUBLIC ${SIBR_NANOFLANN_TARGET})
    endif()

    # sibr_scene
    sibr_add_library(sibr_scene "${SIBR_SRC_DIR}/core/scene")
    target_link_libraries(sibr_scene PUBLIC
        sibr_assets
        sibr_graphics
        sibr_raycaster
        ${SIBR_BOOST_LIBS}
        ${SIBR_ASSIMP_LIBS}
        ${SIBR_GLEW_LIBS}
        ${SIBR_OPENGL_LIBS}
        ${SIBR_OPENCV_LIBS}
    )

    # sibr_video (exclude VideoUtils.cpp which needs opencv2/ximgproc contrib module)
    sibr_add_library(sibr_video "${SIBR_SRC_DIR}/core/video")
    get_target_property(_video_srcs sibr_video SOURCES)
    list(FILTER _video_srcs EXCLUDE REGEX "VideoUtils\\.cpp$")
    set_target_properties(sibr_video PROPERTIES SOURCES "${_video_srcs}")
    target_link_libraries(sibr_video PUBLIC sibr_graphics)
    if(FFMPEG_FOUND OR FFmpeg_FOUND)
        target_include_directories(sibr_video PRIVATE ${FFMPEG_INCLUDE_DIRS})
        # Add FFmpeg library directories to linker search path
        if(DEFINED CONDA_PREFIX)
            target_link_directories(sibr_video PRIVATE ${CONDA_PREFIX}/lib)
        endif()
        target_link_libraries(sibr_video PUBLIC ${FFMPEG_LIBRARIES})
        target_compile_definitions(sibr_video PRIVATE SIBR_HAS_FFMPEG)
    endif()

    # sibr_view
    sibr_add_library(sibr_view "${SIBR_SRC_DIR}/core/view")
    target_link_libraries(sibr_view PUBLIC
        sibr_graphics
        sibr_assets
        sibr_raycaster
        sibr_scene
        sibr_video
        ${SIBR_BOOST_LIBS}
        ${SIBR_OPENCV_LIBS}
        ${SIBR_OPENMP_LIBS}
        ${SIBR_IMGUI_LIBS}
    )

    # sibr_renderer
    sibr_add_library(sibr_renderer "${SIBR_SRC_DIR}/core/renderer")
    target_link_libraries(sibr_renderer PUBLIC
        sibr_system
        sibr_view
        sibr_assets
        sibr_scene
        ${SIBR_BOOST_LIBS}
        ${SIBR_ASSIMP_LIBS}
        ${SIBR_GLEW_LIBS}
        ${SIBR_OPENGL_LIBS}
        ${SIBR_OPENCV_LIBS}
        ${SIBR_GLFW_LIBS}
    )

    # sibr_imgproc
    sibr_add_library(sibr_imgproc "${SIBR_SRC_DIR}/core/imgproc")
    target_link_libraries(sibr_imgproc PUBLIC
        sibr_system
        sibr_graphics
        sibr_assets
        sibr_raycaster
        sibr_mrf
        ${SIBR_BOOST_LIBS}
        ${SIBR_OPENCV_LIBS}
    )

    # =============================================
    # SIBR Project Libraries (STATIC)
    # =============================================

    # sibr_ulr (ULR renderer)
    file(GLOB SIBR_ULR_SOURCES
        "${SIBR_SRC_DIR}/projects/ulr/renderer/*.cpp"
        "${SIBR_SRC_DIR}/projects/ulr/renderer/*.h"
        "${SIBR_SRC_DIR}/projects/ulr/renderer/*.hpp"
        "${SIBR_SRC_DIR}/projects/ulr/renderer/shaders/*.frag"
        "${SIBR_SRC_DIR}/projects/ulr/renderer/shaders/*.vert"
        "${SIBR_SRC_DIR}/projects/ulr/renderer/shaders/*.geom"
    )
    add_library(sibr_ulr STATIC ${SIBR_ULR_SOURCES})
    target_include_directories(sibr_ulr PUBLIC ${SIBR_COMMON_INCLUDES})
    target_link_libraries(sibr_ulr PUBLIC
        sibr_system sibr_view sibr_assets sibr_renderer
        ${SIBR_BOOST_LIBS} ${SIBR_ASSIMP_LIBS} ${SIBR_GLEW_LIBS}
        ${SIBR_OPENGL_LIBS} ${SIBR_OPENCV_LIBS} ${SIBR_GLFW_LIBS}
    )
    target_compile_definitions(sibr_ulr PUBLIC BOOST_ALL_DYN_LINK EIGEN_INITIALIZE_MATRICES_BY_ZERO)
    set_target_properties(sibr_ulr PROPERTIES POSITION_INDEPENDENT_CODE ON FOLDER "plugins/qSIBR/projects")
    if(SIBR_EIGEN_TARGET)
        target_link_libraries(sibr_ulr PUBLIC ${SIBR_EIGEN_TARGET})
    endif()

    # sibr_basic (Basic renderers: PointBased + TexturedMesh)
    file(GLOB SIBR_BASIC_SOURCES
        "${SIBR_SRC_DIR}/projects/basic/renderer/*.cpp"
        "${SIBR_SRC_DIR}/projects/basic/renderer/*.h"
        "${SIBR_SRC_DIR}/projects/basic/renderer/*.hpp"
    )
    if(SIBR_BASIC_SOURCES)
        add_library(sibr_basic STATIC ${SIBR_BASIC_SOURCES})
        target_include_directories(sibr_basic PUBLIC ${SIBR_COMMON_INCLUDES})
        target_link_libraries(sibr_basic PUBLIC
            sibr_system sibr_view sibr_assets sibr_renderer
            ${SIBR_BOOST_LIBS} ${SIBR_GLEW_LIBS} ${SIBR_OPENGL_LIBS}
            ${SIBR_OPENCV_LIBS} ${SIBR_GLFW_LIBS}
        )
        target_compile_definitions(sibr_basic PUBLIC BOOST_ALL_DYN_LINK EIGEN_INITIALIZE_MATRICES_BY_ZERO)
        set_target_properties(sibr_basic PROPERTIES POSITION_INDEPENDENT_CODE ON FOLDER "plugins/qSIBR/projects")
        if(SIBR_EIGEN_TARGET)
            target_link_libraries(sibr_basic PUBLIC ${SIBR_EIGEN_TARGET})
        endif()
    endif()

    # sibr_remote (Remote Gaussian viewer - network based, no CUDA required)
    file(GLOB SIBR_REMOTE_SOURCES
        "${SIBR_SRC_DIR}/projects/remote/renderer/*.cpp"
        "${SIBR_SRC_DIR}/projects/remote/renderer/*.h"
        "${SIBR_SRC_DIR}/projects/remote/renderer/*.hpp"
    )
    if(SIBR_REMOTE_SOURCES)
        add_library(sibr_remote STATIC ${SIBR_REMOTE_SOURCES})
        target_include_directories(sibr_remote PUBLIC
            ${SIBR_COMMON_INCLUDES}
            "${SIBR_SRC_DIR}/projects/remote"
        )
        target_link_libraries(sibr_remote PUBLIC
            sibr_system sibr_view sibr_assets sibr_renderer
            ${SIBR_BOOST_LIBS} ${SIBR_GLEW_LIBS} ${SIBR_OPENGL_LIBS}
            ${SIBR_OPENCV_LIBS} ${SIBR_GLFW_LIBS}
        )
        target_compile_definitions(sibr_remote PUBLIC BOOST_ALL_DYN_LINK EIGEN_INITIALIZE_MATRICES_BY_ZERO)
        set_target_properties(sibr_remote PROPERTIES POSITION_INDEPENDENT_CODE ON FOLDER "plugins/qSIBR/projects")
        if(SIBR_EIGEN_TARGET)
            target_link_libraries(sibr_remote PUBLIC ${SIBR_EIGEN_TARGET})
        endif()
        set(SIBR_HAS_REMOTE ON)
    else()
        set(SIBR_HAS_REMOTE OFF)
    endif()

    # CudaRasterizer + sibr_gaussian (3D Gaussian Splatting viewer)
    set(SIBR_CUDA_RASTERIZER_DIR "${SIBR_3RDPARTY_DIR}/CudaRasterizer")
    if(SIBR_HAS_CUDA AND EXISTS "${SIBR_CUDA_RASTERIZER_DIR}/cuda_rasterizer/rasterizer.h")
        add_library(CudaRasterizer STATIC
            "${SIBR_CUDA_RASTERIZER_DIR}/cuda_rasterizer/backward.cu"
            "${SIBR_CUDA_RASTERIZER_DIR}/cuda_rasterizer/forward.cu"
            "${SIBR_CUDA_RASTERIZER_DIR}/cuda_rasterizer/rasterizer_impl.cu"
        )
        target_include_directories(CudaRasterizer PUBLIC
            "${SIBR_CUDA_RASTERIZER_DIR}/cuda_rasterizer"
            "${SIBR_CUDA_RASTERIZER_DIR}"
        )
        target_include_directories(CudaRasterizer PRIVATE
            "${SIBR_CUDA_RASTERIZER_DIR}/third_party/glm"
        )
        target_link_libraries(CudaRasterizer PUBLIC CUDA::cudart)
        set_target_properties(CudaRasterizer PROPERTIES
            CUDA_STANDARD 14
            POSITION_INDEPENDENT_CODE ON
            FOLDER "plugins/qSIBR/projects"
        )
        message(STATUS "qSIBR: CudaRasterizer found - 3D Gaussian Splatting viewer enabled")

        file(GLOB SIBR_GAUSSIAN_SOURCES
            "${SIBR_SRC_DIR}/projects/gaussianviewer/renderer/*.cpp"
            "${SIBR_SRC_DIR}/projects/gaussianviewer/renderer/*.h"
            "${SIBR_SRC_DIR}/projects/gaussianviewer/renderer/*.hpp"
        )
        if(SIBR_GAUSSIAN_SOURCES)
            add_library(sibr_gaussian STATIC ${SIBR_GAUSSIAN_SOURCES})
            target_include_directories(sibr_gaussian PUBLIC
                ${SIBR_COMMON_INCLUDES}
                "${SIBR_SRC_DIR}/projects/gaussianviewer/renderer"
            )
            target_link_libraries(sibr_gaussian PUBLIC
                sibr_system sibr_view sibr_assets sibr_renderer
                ${SIBR_BOOST_LIBS} ${SIBR_GLEW_LIBS} ${SIBR_OPENGL_LIBS}
                ${SIBR_OPENCV_LIBS} ${SIBR_GLFW_LIBS}
                CudaRasterizer CUDA::cudart
            )
            target_compile_definitions(sibr_gaussian PUBLIC
                BOOST_ALL_DYN_LINK EIGEN_INITIALIZE_MATRICES_BY_ZERO SIBR_HAS_CUDA
            )
            if(NOT MSVC)
                target_compile_options(sibr_gaussian PRIVATE -fpermissive -Wno-error)
            endif()
            set_target_properties(sibr_gaussian PROPERTIES
                POSITION_INDEPENDENT_CODE ON FOLDER "plugins/qSIBR/projects"
            )
            if(SIBR_EIGEN_TARGET)
                target_link_libraries(sibr_gaussian PUBLIC ${SIBR_EIGEN_TARGET})
            endif()
        endif()
    else()
        if(SIBR_HAS_CUDA)
            message(STATUS "qSIBR: CudaRasterizer not found at ${SIBR_CUDA_RASTERIZER_DIR} - Gaussian viewer disabled")
        endif()
    endif()

    # Convenience: all SIBR libraries
    set(SIBR_ALL_LIBS
        sibr_ulr
        sibr_imgproc
        sibr_renderer
        sibr_view
        sibr_video
        sibr_scene
        sibr_raycaster
        sibr_assets
        sibr_graphics
        sibr_system
        sibr_imgui
        sibr_xatlas
        sibr_mrf
        sibr_picojson
        sibr_rapidxml
    )
    if(TARGET sibr_basic)
        list(APPEND SIBR_ALL_LIBS sibr_basic)
    endif()
    if(TARGET sibr_remote)
        list(APPEND SIBR_ALL_LIBS sibr_remote)
    endif()
    if(SIBR_HAS_CUDA AND TARGET sibr_gaussian)
        list(APPEND SIBR_ALL_LIBS sibr_gaussian)
    endif()

    # =============================================
    # SIBR Standalone Executables (optional)
    # =============================================
    # All viewer functionality is already integrated into the plugin .so.
    # These standalone apps are only needed for independent CLI usage
    # or debugging; disable them to reduce build time and package size.
    option(SIBR_BUILD_STANDALONE_APPS
        "Build standalone SIBR viewer executables (redundant when using the plugin)" OFF)
    option(SIBR_BUILD_DATASET_TOOLS
        "Build SIBR dataset preprocessing tools" OFF)

    macro(sibr_add_app APP_TARGET APP_SRC_DIR)
        file(GLOB _app_src "${APP_SRC_DIR}/*.cpp" "${APP_SRC_DIR}/*.h" "${APP_SRC_DIR}/*.hpp")
        file(GLOB _app_rsc "${APP_SRC_DIR}/resources/*.ini")
        add_executable(${APP_TARGET} ${_app_src})
        target_include_directories(${APP_TARGET} PRIVATE ${SIBR_COMMON_INCLUDES})
        target_compile_definitions(${APP_TARGET} PRIVATE
            BOOST_ALL_DYN_LINK EIGEN_INITIALIZE_MATRICES_BY_ZERO
        )
        if(NOT MSVC)
            target_compile_options(${APP_TARGET} PRIVATE -fpermissive -Wno-error)
        endif()
        set_target_properties(${APP_TARGET} PROPERTIES
            FOLDER "plugins/qSIBR/apps"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
        if(SIBR_EIGEN_TARGET)
            target_link_libraries(${APP_TARGET} PRIVATE ${SIBR_EIGEN_TARGET})
        endif()
        if(UNIX AND NOT APPLE)
            target_link_libraries(${APP_TARGET} PRIVATE
                rt m dl X11 pthread Xrandr Xinerama Xxf86vm Xcursor
            )
            target_link_options(${APP_TARGET} PRIVATE
                "LINKER:--copy-dt-needed-entries"
                "LINKER:--start-group"
                "LINKER:--gc-sections"
            )
        elseif(WIN32)
            target_link_libraries(${APP_TARGET} PRIVATE ws2_32 shlwapi)
        endif()
    endmacro()

    set(SIBR_APP_TARGETS "")

    set(SIBR_COMMON_APP_LIBS
        ${SIBR_ALL_LIBS}
        ${SIBR_BOOST_LIBS}
        ${SIBR_GLEW_LIBS}
        ${SIBR_OPENGL_LIBS}
        ${SIBR_OPENCV_LIBS}
        ${SIBR_GLFW_LIBS}
        ${SIBR_OPENMP_LIBS}
        ${SIBR_EMBREE_LIBS}
        ${SIBR_ASSIMP_LIBS}
        ${SIBR_ZLIB_RAW}
    )

    if(SIBR_BUILD_STANDALONE_APPS)
        # ULR Viewer
        sibr_add_app(SIBR_ulr_app "${SIBR_SRC_DIR}/projects/ulr/apps/ulr")
        target_link_libraries(SIBR_ulr_app PRIVATE ${SIBR_COMMON_APP_LIBS})
        list(APPEND SIBR_APP_TARGETS SIBR_ulr_app)

        # ULR v2 Viewer
        sibr_add_app(SIBR_ulrv2_app "${SIBR_SRC_DIR}/projects/ulr/apps/ulrv2")
        target_link_libraries(SIBR_ulrv2_app PRIVATE ${SIBR_COMMON_APP_LIBS})
        list(APPEND SIBR_APP_TARGETS SIBR_ulrv2_app)

        # Textured Mesh Viewer
        sibr_add_app(SIBR_texturedMesh_app "${SIBR_SRC_DIR}/projects/ulr/apps/texturedMesh")
        target_link_libraries(SIBR_texturedMesh_app PRIVATE ${SIBR_COMMON_APP_LIBS})
        list(APPEND SIBR_APP_TARGETS SIBR_texturedMesh_app)

        # Remote Gaussian Viewer (network, no CUDA)
        if(SIBR_HAS_REMOTE)
            sibr_add_app(SIBR_remoteGaussian_app
                "${SIBR_SRC_DIR}/projects/remote/apps/remoteGaussianUI")
            target_link_libraries(SIBR_remoteGaussian_app PRIVATE ${SIBR_COMMON_APP_LIBS})
            list(APPEND SIBR_APP_TARGETS SIBR_remoteGaussian_app)
        endif()

        # 3D Gaussian Splatting Viewer (CUDA required)
        if(SIBR_HAS_CUDA AND TARGET sibr_gaussian)
            sibr_add_app(SIBR_gaussianViewer_app
                "${SIBR_SRC_DIR}/projects/gaussianviewer/apps/gaussianViewer")
            target_link_libraries(SIBR_gaussianViewer_app PRIVATE
                ${SIBR_COMMON_APP_LIBS} CUDA::cudart CUDA::cuda_driver)
            list(APPEND SIBR_APP_TARGETS SIBR_gaussianViewer_app)
        endif()

        # Basic Point-Based Viewer
        if(TARGET sibr_basic AND EXISTS "${SIBR_SRC_DIR}/projects/basic/apps/pointBased/main.cpp")
            sibr_add_app(SIBR_pointBased_app
                "${SIBR_SRC_DIR}/projects/basic/apps/pointBased")
            target_link_libraries(SIBR_pointBased_app PRIVATE ${SIBR_COMMON_APP_LIBS})
            list(APPEND SIBR_APP_TARGETS SIBR_pointBased_app)
        endif()
    endif()

    # =============================================
    # Dataset Tool Executables
    # =============================================

    set(SIBR_PREPROCESS_DIR "${SIBR_SRC_DIR}/projects/dataset_tools/preprocess")

    set(DATASET_TOOL_APPS
        alignMeshes
        cameraConverter
        clippingPlanes
        cropFromCenter
        distordCrop
        nvmToSIBR
        prepareColmap4Sibr
        textureMesh
        tonemapper
        unwrapMesh
    )

    # Always embed tool entry points into the plugin .so.
    # Each tool main.cpp uses #ifdef SIBR_TOOL_EMBEDDED to expose
    # sibr_tool_<name>() instead of main().
    foreach(tool_name ${DATASET_TOOL_APPS})
        set(tool_main "${SIBR_PREPROCESS_DIR}/${tool_name}/main.cpp")
        if(EXISTS "${tool_main}")
            target_sources(${PROJECT_NAME} PRIVATE "${tool_main}")
            set_source_files_properties("${tool_main}" PROPERTIES
                COMPILE_DEFINITIONS "SIBR_TOOL_EMBEDDED"
            )
            if(NOT MSVC)
                set_source_files_properties("${tool_main}" PROPERTIES
                    COMPILE_OPTIONS "-fpermissive;-Wno-error"
                )
            endif()
        endif()
    endforeach()

    # Optionally also build standalone tool executables.
    if(SIBR_BUILD_DATASET_TOOLS)
        foreach(tool_name ${DATASET_TOOL_APPS})
            set(tool_dir "${SIBR_PREPROCESS_DIR}/${tool_name}")
            if(EXISTS "${tool_dir}/main.cpp")
                sibr_add_app(${tool_name} "${tool_dir}")
                target_link_libraries(${tool_name} PRIVATE ${SIBR_COMMON_APP_LIBS})
            endif()
        endforeach()
    endif()

    # =============================================
    # Link SIBR libraries into the plugin.so
    # =============================================
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${SIBR_COMMON_INCLUDES}
        "${SIBR_SRC_DIR}/projects/ulr/renderer"
        "${SIBR_SRC_DIR}/projects/basic/renderer"
        "${SIBR_SRC_DIR}/projects/remote"
        "${SIBR_SRC_DIR}/projects/remote/renderer"
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        BOOST_ALL_DYN_LINK EIGEN_INITIALIZE_MATRICES_BY_ZERO
    )
    if(SIBR_HAS_REMOTE)
        target_compile_definitions(${PROJECT_NAME} PRIVATE SIBR_HAS_REMOTE)
    endif()
    if(SIBR_HAS_CUDA AND TARGET sibr_gaussian)
        target_compile_definitions(${PROJECT_NAME} PRIVATE SIBR_HAS_CUDA)
        target_include_directories(${PROJECT_NAME} PRIVATE
            "${SIBR_SRC_DIR}/projects/gaussianviewer/renderer"
        )
    endif()
    target_link_libraries(${PROJECT_NAME}
        ${SIBR_ALL_LIBS}
        ${SIBR_BOOST_LIBS}
        ${SIBR_ASSIMP_LIBS}
        ${SIBR_ZLIB_RAW}
        ${SIBR_GLEW_LIBS}
        ${SIBR_OPENGL_LIBS}
        ${SIBR_OPENCV_LIBS}
        ${SIBR_GLFW_LIBS}
        ${SIBR_OPENMP_LIBS}
        ${SIBR_EMBREE_LIBS}
    )
    # Add FFmpeg libraries for sibr_video support (if found)
    if(FFMPEG_FOUND OR FFmpeg_FOUND)
        if(DEFINED CONDA_PREFIX)
            target_link_directories(${PROJECT_NAME} PRIVATE ${CONDA_PREFIX}/lib)
        endif()
        target_link_libraries(${PROJECT_NAME} ${FFMPEG_LIBRARIES})
    endif()
    if(SIBR_NANOFLANN_TARGET)
        target_link_libraries(${PROJECT_NAME} ${SIBR_NANOFLANN_TARGET})
    endif()
    if(SIBR_IMGUI_LIBS)
        target_link_libraries(${PROJECT_NAME} ${SIBR_IMGUI_LIBS})
    endif()
    if(SIBR_EIGEN_TARGET)
        target_link_libraries(${PROJECT_NAME} ${SIBR_EIGEN_TARGET})
    endif()
    if(UNIX AND NOT APPLE)
        target_link_libraries(${PROJECT_NAME}
            rt m dl X11 pthread Xrandr Xinerama Xxf86vm Xcursor
        )
    elseif(WIN32)
        target_link_libraries(${PROJECT_NAME} ws2_32 shlwapi)
    endif()

    # =============================================
    # Install rules
    # =============================================
    if(SIBR_APP_TARGETS)
        install(TARGETS ${SIBR_APP_TARGETS} RUNTIME DESTINATION bin)
    endif()
    if(SIBR_BUILD_DATASET_TOOLS)
        foreach(tool_name ${DATASET_TOOL_APPS})
            if(TARGET ${tool_name})
                install(TARGETS ${tool_name} RUNTIME DESTINATION bin)
            endif()
        endforeach()
    endif()

    # All SIBR runtime assets are installed alongside the executable (bin/).
    # getInstallDirectory() now returns the exe directory directly.
    set(SIBR_BIN_DIR "${CMAKE_BINARY_DIR}/bin")

    # Shader resources: preserve subdirectory structure for runtime lookup
    # SIBR resolves shaders via getShadersDirectory("core"), getShadersDirectory("ulr"), etc.
    function(sibr_install_shaders SRC_DIR SUBFOLDER)
        file(GLOB _shader_files "${SRC_DIR}/*.frag" "${SRC_DIR}/*.vert" "${SRC_DIR}/*.geom"
                                "${SRC_DIR}/*.fp" "${SRC_DIR}/*.vp" "${SRC_DIR}/*.gp")
        foreach(_sf ${_shader_files})
            get_filename_component(_sf_name ${_sf} NAME)
            configure_file(${_sf} "${SIBR_BIN_DIR}/shaders/${SUBFOLDER}/${_sf_name}" COPYONLY)
        endforeach()
        install(FILES ${_shader_files} DESTINATION bin/shaders/${SUBFOLDER})
    endfunction()

    sibr_install_shaders("${SIBR_SRC_DIR}/core/renderer/shaders" "core")
    sibr_install_shaders("${SIBR_SRC_DIR}/core/view/shaders" "core")
    sibr_install_shaders("${SIBR_SRC_DIR}/projects/ulr/renderer/shaders" "ulr")
    if(SIBR_HAS_CUDA AND TARGET sibr_gaussian)
        sibr_install_shaders("${SIBR_SRC_DIR}/projects/gaussianviewer/renderer/shaders" "gaussian")
    endif()

    # INI / resource files: SIBR resolves via getResourcesDirectory() -> "sibr_resources"
    # Uses sibr_resources/ to avoid conflict with ACloudViewer's own resources/ directory.
    function(sibr_install_resources SRC_DIR SUBFOLDER)
        file(GLOB _rsc_files "${SRC_DIR}/*.ini" "${SRC_DIR}/*.json")
        foreach(_rf ${_rsc_files})
            get_filename_component(_rf_name ${_rf} NAME)
            configure_file(${_rf} "${SIBR_BIN_DIR}/sibr_resources/${SUBFOLDER}/${_rf_name}" COPYONLY)
        endforeach()
        install(FILES ${_rsc_files} DESTINATION bin/sibr_resources/${SUBFOLDER})
    endfunction()

    sibr_install_resources("${SIBR_SRC_DIR}/core/graphics/resources" "core")
    sibr_install_resources("${SIBR_SRC_DIR}/projects/ulr/apps/ulr/resources" "ulr")
    sibr_install_resources("${SIBR_SRC_DIR}/projects/ulr/apps/ulrv2/resources" "ulr")
    sibr_install_resources("${SIBR_SRC_DIR}/projects/ulr/apps/texturedMesh/resources" "ulr")
    sibr_install_resources("${SIBR_SRC_DIR}/projects/basic/apps/texturedMesh/resources" "basic")

    # ibr_resources.ini: SIBR Resources singleton uses this to locate shader files.
    # Paths are relative to getInstallDirectory() (= the exe directory, i.e. bin/).
    set(_ibr_rsc_paths "shaders/core\nshaders/ulr\nsibr_resources/core\nsibr_resources/ulr\nsibr_resources/basic")
    if(SIBR_HAS_CUDA AND TARGET sibr_gaussian)
        string(APPEND _ibr_rsc_paths "\nshaders/gaussian\nsibr_resources/gaussian")
    endif()
    file(WRITE "${SIBR_BIN_DIR}/ibr_resources.ini" "${_ibr_rsc_paths}\n")
    install(FILES "${SIBR_BIN_DIR}/ibr_resources.ini" DESTINATION bin)

endif()
