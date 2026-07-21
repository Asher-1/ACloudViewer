cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

add_library(MCALIB_CALIB_IO STATIC)

set(MCALIB_CALIB_IO_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CalibConfigParser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CameraModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RosBagReader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RosBagWriter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ProtoDecoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/VideoCodecDecoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BagAlignment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BagDiscovery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CloudDecombine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ExtrinsicCompare.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/PcdExport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mcalib_portability.cpp
)

target_sources(MCALIB_CALIB_IO PRIVATE ${MCALIB_CALIB_IO_SRCS})

target_compile_features(MCALIB_CALIB_IO PUBLIC cxx_std_17)

target_include_directories(MCALIB_CALIB_IO
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_definitions(MCALIB_CALIB_IO PRIVATE _USE_MATH_DEFINES NOMINMAX)
elseif(APPLE)
    target_compile_options(MCALIB_CALIB_IO PRIVATE -Wall -Wextra)
else()
    target_compile_options(MCALIB_CALIB_IO PRIVATE -Wall -Wextra)
endif()

target_link_libraries(MCALIB_CALIB_IO
    PUBLIC
        3rdparty_opencv
        3rdparty_eigen3
        3rdparty_threads
    PRIVATE
        CVCoreLib
)

if(MCALIB_WITH_FFMPEG_SUPPORT)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/MCalibFFmpeg.cmake)
    if(MCALIB_FFMPEG_FOUND)
        mcalib_target_link_ffmpeg(MCALIB_CALIB_IO)
    else()
        message(STATUS "qManualCalib calib_io: FFmpeg unavailable; H.264/HEVC decode disabled")
    endif()
else()
    message(STATUS "qManualCalib calib_io: MCALIB_WITH_FFMPEG_SUPPORT=OFF")
endif()

# Embedded LZ4 / BZ2 for ROS bag decompression (private to this library)
set(MCALIB_CALIB_IO_EMBEDDED_SRCS "")
set(MCALIB_LZ4_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/lz4)
if(EXISTS ${MCALIB_LZ4_DIR}/lz4.c)
    list(APPEND MCALIB_CALIB_IO_EMBEDDED_SRCS ${MCALIB_LZ4_DIR}/lz4.c)
    target_sources(MCALIB_CALIB_IO PRIVATE ${MCALIB_LZ4_DIR}/lz4.c)
    target_include_directories(MCALIB_CALIB_IO PRIVATE ${MCALIB_LZ4_DIR})
    target_compile_definitions(MCALIB_CALIB_IO PRIVATE MCALIB_HAS_LZ4)
    message(STATUS "qManualCalib calib_io: LZ4 enabled (embedded)")
endif()

set(MCALIB_BZ2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/bz2)
set(MCALIB_BZ2_SRCS
    ${MCALIB_BZ2_DIR}/blocksort.c
    ${MCALIB_BZ2_DIR}/bzlib.c
    ${MCALIB_BZ2_DIR}/compress.c
    ${MCALIB_BZ2_DIR}/crctable.c
    ${MCALIB_BZ2_DIR}/decompress.c
    ${MCALIB_BZ2_DIR}/huffman.c
    ${MCALIB_BZ2_DIR}/randtable.c
)
if(EXISTS ${MCALIB_BZ2_DIR}/bzlib.c)
    list(APPEND MCALIB_CALIB_IO_EMBEDDED_SRCS ${MCALIB_BZ2_SRCS})
    target_sources(MCALIB_CALIB_IO PRIVATE ${MCALIB_BZ2_SRCS})
    target_include_directories(MCALIB_CALIB_IO PRIVATE ${MCALIB_BZ2_DIR})
    target_compile_definitions(MCALIB_CALIB_IO PRIVATE MCALIB_HAS_BZ2)
    message(STATUS "qManualCalib calib_io: BZ2 enabled (embedded)")
endif()

if(MCALIB_CALIB_IO_EMBEDDED_SRCS)
    if(MSVC)
        set_source_files_properties(${MCALIB_CALIB_IO_EMBEDDED_SRCS}
            PROPERTIES COMPILE_FLAGS "/wd4996 /wd4244 /wd4267"
        )
    else()
        set_source_files_properties(${MCALIB_CALIB_IO_EMBEDDED_SRCS}
            PROPERTIES COMPILE_FLAGS "-fvisibility=hidden"
        )
    endif()
endif()
