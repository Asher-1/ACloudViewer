# ------------------------------------------------------------------------------
# FFmpeg support for qManualCalib (based on qAnimation/QTFFmpegWrapper)
# ------------------------------------------------------------------------------

if(MCALIB_FFMPEG_CONFIGURED)
    return()
endif()
set(MCALIB_FFMPEG_CONFIGURED TRUE)

set(MCALIB_FFMPEG_FOUND FALSE)

set(_MCALIB_QANIM_FFMPEG_CMAKE
    "${CMAKE_CURRENT_LIST_DIR}/../../qAnimation/extern/QTFFmpegWrapper/cmake/FFmpegSupport.cmake")

if(NOT EXISTS "${_MCALIB_QANIM_FFMPEG_CMAKE}")
    message(STATUS "qManualCalib: qAnimation FFmpegSupport.cmake not found; H.264/HEVC decode disabled")
    return()
endif()

# Probe pkg-config first (Linux/macOS); FFmpegSupport.cmake reuses these variables.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG QUIET
        libavcodec
        libavformat
        libavutil
        libswscale
    )
endif()

if(NOT FFMPEG_FOUND AND NOT DEFINED ENV{CONDA_PREFIX})
    find_path(_MCALIB_FF_AVCODEC_INCLUDE_DIR
        libavcodec/avcodec.h
        HINTS
            /usr/include
            /usr/local/include
            /opt/homebrew/include
    )
    find_library(_MCALIB_FF_AVCODEC_LIBRARY
        NAMES avcodec
        HINTS
            /usr/lib
            /usr/local/lib
            /opt/homebrew/lib
    )
    if(_MCALIB_FF_AVCODEC_INCLUDE_DIR AND _MCALIB_FF_AVCODEC_LIBRARY)
        set(FFMPEG_INCLUDE_DIRS "${_MCALIB_FF_AVCODEC_INCLUDE_DIR}")
        get_filename_component(FFMPEG_LIBRARY_DIR "${_MCALIB_FF_AVCODEC_LIBRARY}" DIRECTORY)
        set(FFMPEG_FOUND TRUE)
    endif()
endif()

if(NOT FFMPEG_FOUND AND NOT DEFINED ENV{CONDA_PREFIX})
    message(STATUS "qManualCalib: FFmpeg not found; H.264/HEVC decode disabled")
    return()
endif()

include("${_MCALIB_QANIM_FFMPEG_CMAKE}")

if(NOT EXISTS "${FFMPEG_INCLUDE_DIRS}")
    message(STATUS "qManualCalib: FFmpeg include dir missing; H.264/HEVC decode disabled")
    return()
endif()

if(NOT EXISTS "${FFMPEG_LIBRARY_DIR}")
    message(STATUS "qManualCalib: FFmpeg library dir missing; H.264/HEVC decode disabled")
    return()
endif()

set(MCALIB_FFMPEG_FOUND TRUE)
message(STATUS "qManualCalib: FFmpeg enabled (${FFMPEG_INCLUDE_DIRS})")

function(mcalib_target_link_ffmpeg target_name)
    if(NOT MCALIB_FFMPEG_FOUND)
        return()
    endif()
    target_include_directories(${target_name} PRIVATE ${FFMPEG_INCLUDE_DIRS})
    set(_MCALIB_FF_LIBS "")
    foreach(libfile avutil avcodec avformat swscale)
        if(WIN32)
            list(APPEND _MCALIB_FF_LIBS ${FFMPEG_LIBRARY_DIR}/${libfile}.lib)
        elseif(APPLE)
            list(APPEND _MCALIB_FF_LIBS ${FFMPEG_LIBRARY_DIR}/lib${libfile}.dylib)
        else()
            list(APPEND _MCALIB_FF_LIBS ${FFMPEG_LIBRARY_DIR}/lib${libfile}.so)
        endif()
    endforeach()
    target_link_libraries(${target_name} PUBLIC ${_MCALIB_FF_LIBS})
    if(APPLE)
        target_link_libraries(${target_name} PRIVATE
            "-liconv"
            "-L${FFMPEG_X264_LIBRARY_DIR} -lx264"
            "-lz"
            "-framework CoreVideo"
        )
    endif()
    target_compile_definitions(${target_name} PRIVATE
        MCALIB_HAS_FFMPEG
        __STDC_CONSTANT_MACROS
    )
    if(MSVC)
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
    endif()
endfunction()

function(mcalib_export_ffmpeg_runtime dest_dir)
    if(NOT MCALIB_FFMPEG_FOUND OR NOT WIN32)
        return()
    endif()
    export_ffmpeg_dlls(${dest_dir})
endfunction()
