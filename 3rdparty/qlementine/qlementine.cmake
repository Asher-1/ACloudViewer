include(ExternalProject)
find_package(Git REQUIRED)

set(_qlementine_version v1.5.2)
set(_qlementine_archive_hash e59107e4ea2d70393dddec0c246d59fe30bd4b20360255ae56cf793ac45187d8)
set(_qlementine_patch "${CMAKE_CURRENT_LIST_DIR}/patches/0001-qt5-qt6-compatibility.patch")
set(_qlementine_manifest "${CMAKE_CURRENT_LIST_DIR}/port-manifest.json")
set(_qlementine_runner "${CMAKE_CURRENT_LIST_DIR}/patches/apply_qlementine_patch.cmake")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_qlementine_patch}" "${_qlementine_manifest}" "${_qlementine_runner}")
file(SHA256 "${_qlementine_patch}" _qlementine_patch_hash)
file(SHA256 "${_qlementine_manifest}" _qlementine_manifest_hash)
file(SHA256 "${_qlementine_runner}" _qlementine_runner_hash)
string(SHA256 _qlementine_source_id
    "${_qlementine_version}:${_qlementine_archive_hash}:${_qlementine_patch_hash}:${_qlementine_manifest_hash}:${_qlementine_runner_hash}:Qt6=${USE_QT6}")
string(SUBSTRING "${_qlementine_source_id}" 0 16 _qlementine_source_id)

# cmake/CMakeExternalLibs.cmake is the single owner of the Qt major and imported
# aliases; fail rather than silently rediscover/select another Qt for the app.
if(NOT DEFINED QT_PREFIX OR NOT TARGET Qt::Core OR NOT TARGET Qt::Widgets OR
   NOT TARGET Qt::Svg)
    message(FATAL_ERROR
        "Qlementine requires the central Qt owner to provide QT_PREFIX and Qt::Core/Widgets/Svg")
endif()
set(_qlementine_qt "${QT_PREFIX}")
set(_qlementine_args ${ExternalProject_CMAKE_ARGS_hidden})
# Re-encode forwarded list values before ExternalProject expands its arguments.
list(FILTER _qlementine_args EXCLUDE REGEX
    "^-DCMAKE_(CONFIGURATION_TYPES|MSVC_RUNTIME_LIBRARY|CXX_COMPILER_LAUNCHER|C_COMPILER_LAUNCHER)(:|=)")
# The shared arguments are a CMake list. Keep each assignment and discard any
# dangling configuration/launcher entries before appending the encoded forms.
list(FILTER _qlementine_args INCLUDE REGEX "^-D")
foreach(_var CXX_COMPILER CXX_COMPILER_LAUNCHER TOOLCHAIN_FILE BUILD_TYPE
             CONFIGURATION_TYPES MSVC_RUNTIME_LIBRARY OSX_ARCHITECTURES
             OSX_DEPLOYMENT_TARGET OSX_SYSROOT SYSROOT SYSROOT_COMPILE SYSROOT_LINK
             CXX_COMPILER_TARGET CXX_COMPILER_EXTERNAL_TOOLCHAIN
             PREFIX_PATH FIND_ROOT_PATH)
    if(DEFINED CMAKE_${_var})
        string(REPLACE ";" "|" _encoded "${CMAKE_${_var}}")
        list(APPEND _qlementine_args "-DCMAKE_${_var}:STRING=${_encoded}")
    endif()
endforeach()
foreach(_module IN ITEMS "" Core Gui Widgets Svg)
    set(_qt_package "${_qlementine_qt}${_module}")
    if(DEFINED ${_qt_package}_DIR)
        list(APPEND _qlementine_args "-D${_qt_package}_DIR:PATH=${${_qt_package}_DIR}")
    endif()
endforeach()

set(_qlementine_install "${CMAKE_BINARY_DIR}/qlementine")
set(_qlementine_libdir "${_qlementine_install}/${CloudViewer_INSTALL_LIB_DIR}")
set(_qlementine_archive
    "${_qlementine_libdir}/${CMAKE_STATIC_LIBRARY_PREFIX}qlementine$<$<CONFIG:Debug>:_debug>${CMAKE_STATIC_LIBRARY_SUFFIX}")
ExternalProject_Add(ext_qlementine
    PREFIX "${CMAKE_BINARY_DIR}/qlementine-ep/${_qlementine_source_id}"
    INSTALL_DIR "${_qlementine_install}"
    URL "https://codeload.github.com/oclero/qlementine/tar.gz/refs/tags/${_qlementine_version}"
    URL_HASH "SHA256=${_qlementine_archive_hash}"
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/qlementine"
    DOWNLOAD_NAME "qlementine-${_qlementine_version}.tar.gz"
    UPDATE_COMMAND ""
    PATCH_COMMAND "${CMAKE_COMMAND}"
        "-DSOURCE_DIR=<SOURCE_DIR>" "-DPATCH_FILE=${_qlementine_patch}"
        "-DMANIFEST_FILE=${_qlementine_manifest}" "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
        "-DQLEMENTINE_VERSION=${_qlementine_version}" -P "${_qlementine_runner}"
    LIST_SEPARATOR "|"
    CMAKE_ARGS ${_qlementine_args}
        "-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>"
        "-DCMAKE_INSTALL_LIBDIR:STRING=${CloudViewer_INSTALL_LIB_DIR}"
        -DCMAKE_INSTALL_INCLUDEDIR:STRING=include
        -DCMAKE_POLICY_DEFAULT_CMP0091:STRING=NEW
        -DCMAKE_POLICY_DEFAULT_CMP0063:STRING=NEW
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        "-DUSE_QT6:BOOL=${USE_QT6}"
        -DQLEMENTINE_BUILD_STATIC:BOOL=ON
        -DQLEMENTINE_BUILD_SHARED:BOOL=OFF
        -DQLEMENTINE_BUILD_STYLE_PLUGIN:BOOL=OFF
        -DQLEMENTINE_SANDBOX:BOOL=OFF
        -DQLEMENTINE_SHOWCASE:BOOL=OFF
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config $<CONFIG> --parallel 4
    # CMake 3.24 has no INSTALL_BYPRODUCTS. Stage installation in the producing
    # step so Ninja can order both the imported archive and export header.
    COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR> --config $<CONFIG>
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${_qlementine_archive}"
        "${_qlementine_install}/include/oclero/qlementine/qlementine_export.h"
)

# A completed ExternalProject stamp is not evidence that its source is intact.
# Reuse must verify the whole patched state before configure/build, never silently
# repair source drift. ALWAYS intentionally reruns the cheap configure step too.
ExternalProject_Add_Step(ext_qlementine verify_patch
    COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=<SOURCE_DIR>"
        "-DPATCH_FILE=${_qlementine_patch}" "-DMANIFEST_FILE=${_qlementine_manifest}"
        "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}" "-DQLEMENTINE_VERSION=${_qlementine_version}"
        -DVERIFY_ONLY=ON -P "${_qlementine_runner}"
    DEPENDEES patch DEPENDERS configure ALWAYS 1 INDEPENDENT TRUE)

file(MAKE_DIRECTORY "${_qlementine_install}/include")
add_library(qlementine::qlementine STATIC IMPORTED)
set_target_properties(qlementine::qlementine PROPERTIES
    IMPORTED_LOCATION "${_qlementine_libdir}/${CMAKE_STATIC_LIBRARY_PREFIX}qlementine${CMAKE_STATIC_LIBRARY_SUFFIX}"
    IMPORTED_LOCATION_DEBUG "${_qlementine_libdir}/${CMAKE_STATIC_LIBRARY_PREFIX}qlementine_debug${CMAKE_STATIC_LIBRARY_SUFFIX}"
    INTERFACE_INCLUDE_DIRECTORIES "${_qlementine_install}/include"
    INTERFACE_COMPILE_DEFINITIONS QLEMENTINE_STATIC_DEFINE
    INTERFACE_LINK_LIBRARIES "Qt::Core;Qt::Widgets;Qt::Svg")
add_dependencies(qlementine::qlementine ext_qlementine)

install(FILES
    "${CMAKE_CURRENT_LIST_DIR}/LICENSE"
    "${CMAKE_CURRENT_LIST_DIR}/Inter-OFL.txt"
    "${CMAKE_CURRENT_LIST_DIR}/RobotoMono-OFL.txt"
    "${CMAKE_CURRENT_LIST_DIR}/upstream.json"
    DESTINATION share/ACloudViewer/licenses/qlementine
    COMPONENT Runtime)
