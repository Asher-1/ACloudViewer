if (APPLE)
    set(APP_EXTENSION ".app")
    set(PACKAGE_EXTENSION "dmg")
    set(CONFIG_POSTFIX "mac")
    # set(PACK_SCRIPTS ${PACK_SCRIPTS_PATH}/pack_macosx_bundle.sh)
elseif (UNIX)
    set(APP_EXTENSION "")
    set(PACKAGE_EXTENSION "run")
    set(CONFIG_POSTFIX "linux")
    set(PACK_SCRIPTS ${PACK_SCRIPTS_PATH}/pack_ubuntu.sh)
elseif(WIN32)
    set(APP_EXTENSION ".exe")
    set(PACKAGE_EXTENSION "exe")
    set(CONFIG_POSTFIX "win")
    set(PACK_SCRIPTS ${PACK_SCRIPTS_PATH}/pack_windows.ps1)
endif()

set(CONFIG_FILE_PATH ${DEPLOY_ROOT_PATH}/config/config_${CONFIG_POSTFIX}.xml)
set(DEPLOY_PACKAGES_PATH ${DEPLOY_ROOT_PATH}/packages)
set(MAIN_WORKING_DIRECTORY ${DEPLOY_ROOT_PATH})
set(COLMAP_DEPLOY_PATH ${DEPLOY_ROOT_PATH}/packages/${COLMAP_APP_NAME}/data)
set(MAIN_DEPLOY_PATH ${DEPLOY_ROOT_PATH}/packages/${MAIN_APP_NAME}/data)
set(CLOUDVIEWER_DEPLOY_PATH ${DEPLOY_ROOT_PATH}/packages/${CLOUDVIEWER_APP_NAME}/data)
set(DEPLOY_LIB_PATH ${MAIN_DEPLOY_PATH}/${LIBS_FOLDER_NAME})

function(replace_version_in_file file_path)
    # read contents
    file(READ "${file_path}" FILE_CONTENT)
    
    # replace version
    string(REPLACE "3.9.0" "${CLOUDVIEWER_VERSION}" UPDATED_CONTENT "${FILE_CONTENT}")
    
    # write back contents
    file(WRITE "${file_path}" "${UPDATED_CONTENT}")
endfunction()

function(replace_buildtime_in_file file_path)
    # read contents
    file(READ "${file_path}" FILE_CONTENT)
    # replace build time
    string(REPLACE "2024-09-18" "${BUILD_TIME}" UPDATED_CONTENT "${FILE_CONTENT}")
    # write back contents
    file(WRITE "${file_path}" "${UPDATED_CONTENT}")
endfunction()

function(copy_rename_files src_dir src_name dst_dir dst_name)
    file(COPY 
        "${src_dir}/${src_name}"
        DESTINATION "${dst_dir}"
        USE_SOURCE_PERMISSIONS
    )
    file(RENAME 
        "${dst_dir}/${src_name}"
        "${dst_dir}/${dst_name}"
    )
endfunction()

# macOS: ggml backend MODULE libs (.so) are dlopen'd at runtime and may not
# be inside the .app bundle after macdeployqt + lib_bundle_app.py runs.
# Core shared libs (.dylib) may also be missing if lib_bundle_app.py
# didn't discover them via otool (since they're loaded by dlopen).
# Explicitly copy them from the install lib dir into each deploy app's
# Frameworks directory so the app bundle is self-contained.
function(ensure_ggml_backends_in_app app_frameworks_dir install_lib_dir module_suffix)
    if(NOT IS_DIRECTORY "${install_lib_dir}" OR NOT IS_DIRECTORY "${app_frameworks_dir}")
        return()
    endif()
    # Collect both backend modules (.so on macOS) and core shared libs (.dylib)
    file(GLOB _ggml_modules "${install_lib_dir}/libggml-*${module_suffix}")
    file(GLOB _ggml_dylibs  "${install_lib_dir}/libggml*.dylib")
    list(APPEND _ggml_all ${_ggml_modules} ${_ggml_dylibs})
    list(REMOVE_DUPLICATES _ggml_all)
    set(_added 0)
    foreach(_mod IN LISTS _ggml_all)
        if(NOT EXISTS "${_mod}" OR IS_DIRECTORY "${_mod}")
            continue()
        endif()
        # Skip symlinks: resolve the real path and skip if different
        # (compatible with CMake < 3.28 which lacks IS_SYMLINK)
        get_filename_component(_realpath "${_mod}" REALPATH)
        if(NOT "${_realpath}" STREQUAL "${_mod}")
            continue()
        endif()
        get_filename_component(_name "${_mod}" NAME)
        set(_dst "${app_frameworks_dir}/${_name}")
        if(NOT EXISTS "${_dst}")
            file(COPY "${_mod}" DESTINATION "${app_frameworks_dir}"
                 FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                  GROUP_READ GROUP_EXECUTE
                                  WORLD_READ WORLD_EXECUTE)
            math(EXPR _added "${_added} + 1")
        endif()
    endforeach()
    if(_added GREATER 0)
        message(STATUS "PostInstall: copied ${_added} ggml backend(s) into ${app_frameworks_dir}")
    endif()
endfunction()

# 1. Config
## update ACloudViewer version and build time
replace_version_in_file("${CONFIG_FILE_PATH}")
replace_version_in_file("${DEPLOY_PACKAGES_PATH}/${MAIN_APP_NAME}/meta/package.xml")
replace_buildtime_in_file("${DEPLOY_PACKAGES_PATH}/${MAIN_APP_NAME}/meta/package.xml")
replace_version_in_file("${DEPLOY_PACKAGES_PATH}/${MAIN_APP_NAME}/meta/installscript.qs")
## update CloudViewer version and build time          
if (${BUILD_GUI} STREQUAL "ON")
    replace_version_in_file("${DEPLOY_PACKAGES_PATH}/${CLOUDVIEWER_APP_NAME}/meta/package.xml")
    replace_buildtime_in_file("${DEPLOY_PACKAGES_PATH}/${CLOUDVIEWER_APP_NAME}/meta/package.xml")
    replace_version_in_file("${DEPLOY_PACKAGES_PATH}/${CLOUDVIEWER_APP_NAME}/meta/installscript.qs")
endif()
## update Colmap version and build time
if (${BUILD_RECONSTRUCTION} STREQUAL "ON")
    replace_version_in_file("${DEPLOY_PACKAGES_PATH}/${COLMAP_APP_NAME}/meta/package.xml")
    replace_buildtime_in_file("${DEPLOY_PACKAGES_PATH}/${COLMAP_APP_NAME}/meta/package.xml")
    replace_version_in_file("${DEPLOY_PACKAGES_PATH}/${COLMAP_APP_NAME}/meta/installscript.qs")
endif()

# 2. Deploy
set(SOURCE_BIN_PATH ${CMAKE_INSTALL_PREFIX}/${CloudViewer_INSTALL_BIN_DIR})
if (APPLE AND GGML_MODULE_SUFFIX)
    set(_GGML_SRC_DIR "${CMAKE_INSTALL_PREFIX}/${CloudViewer_INSTALL_LIB_DIR}")
endif()
## deploy ACloudViewer
file(COPY "${SOURCE_BIN_PATH}/${MAIN_APP_NAME}/${MAIN_APP_NAME}${APP_EXTENSION}"
    DESTINATION "${MAIN_DEPLOY_PATH}"
    USE_SOURCE_PERMISSIONS)
if (APPLE AND GGML_MODULE_SUFFIX)
    ensure_ggml_backends_in_app(
        "${MAIN_DEPLOY_PATH}/${MAIN_APP_NAME}${APP_EXTENSION}/Contents/${LIBS_FOLDER_NAME}"
        "${_GGML_SRC_DIR}" "${GGML_MODULE_SUFFIX}")
    ensure_ggml_backends_in_app(
        "${SOURCE_BIN_PATH}/${MAIN_APP_NAME}/${MAIN_APP_NAME}${APP_EXTENSION}/Contents/${LIBS_FOLDER_NAME}"
        "${_GGML_SRC_DIR}" "${GGML_MODULE_SUFFIX}")
endif()
if (UNIX AND NOT APPLE)
    # deploy libs, plugins and translations for ACloudViewer
    file(COPY 
        "${CMAKE_INSTALL_PREFIX}/${LIBS_FOLDER_NAME}"
        "${CMAKE_INSTALL_PREFIX}/plugins"
        "${CMAKE_INSTALL_PREFIX}/translations"
        DESTINATION "${MAIN_DEPLOY_PATH}"
        USE_SOURCE_PERMISSIONS
    )

    # deploy SIBR plugin runtime assets (shaders, resources, config)
    foreach(_sibr_asset shaders sibr_resources)
        set(_sibr_asset_path "${SOURCE_BIN_PATH}/${_sibr_asset}")
        if(EXISTS "${_sibr_asset_path}")
            file(COPY "${_sibr_asset_path}"
                DESTINATION "${MAIN_DEPLOY_PATH}"
                USE_SOURCE_PERMISSIONS)
        endif()
    endforeach()
    set(_ibr_ini "${SOURCE_BIN_PATH}/ibr_resources.ini")
    if(EXISTS "${_ibr_ini}")
        file(COPY "${_ibr_ini}"
            DESTINATION "${MAIN_DEPLOY_PATH}"
            USE_SOURCE_PERMISSIONS)
    endif()
    
    if (${PLUGIN_PYTHON} STREQUAL "ON") 
        file(COPY 
            "${CMAKE_INSTALL_PREFIX}/plugins-python"
            DESTINATION "${MAIN_DEPLOY_PATH}"
            USE_SOURCE_PERMISSIONS
        )
    endif()

    # deploy c++ library dependency
    set(EXTERNAL_LIB_DIR ${EXTERNAL_INSTALL_DIRS}/${LIBS_FOLDER_NAME})
    if (${BUILD_WITH_CONDA} STREQUAL "ON")
        list(APPEND EXTERNAL_LIB_DIR "${CONDA_PREFIX}/lib")
    endif()
    execute_process(COMMAND bash ${PACK_SCRIPTS}
                    "${BUILD_LIB_PATH}" ${DEPLOY_LIB_PATH}
                    ${EXTERNAL_LIB_DIR}
                    WORKING_DIRECTORY ${BUILD_LIB_PATH})
    execute_process(COMMAND bash ${PACK_SCRIPTS}
                    "${BUILD_LIB_PATH}/plugins" ${DEPLOY_LIB_PATH}
                    ${EXTERNAL_LIB_DIR}
                    WORKING_DIRECTORY ${BUILD_LIB_PATH})
    
    set(QXCB_LIB_PATH "${MAIN_DEPLOY_PATH}/platforms/libqxcb.so")
    if(EXISTS ${QXCB_LIB_PATH})
        execute_process(COMMAND bash ${PACK_SCRIPTS}
                        ${QXCB_LIB_PATH} ${DEPLOY_LIB_PATH}
                        ${EXTERNAL_LIB_DIR}
                        WORKING_DIRECTORY ${MAIN_DEPLOY_PATH})
    else()
        message(WARNING "File ${QXCB_LIB_PATH} does not exist.")
    endif()
    set(SVGICON_LIB_PATH "${MAIN_DEPLOY_PATH}/iconengines/libqsvgicon.so")
    if(EXISTS "${SVGICON_LIB_PATH}")
        execute_process(COMMAND bash ${PACK_SCRIPTS}
                        ${SVGICON_LIB_PATH} ${DEPLOY_LIB_PATH}
                        ${EXTERNAL_LIB_DIR}
                        WORKING_DIRECTORY ${MAIN_DEPLOY_PATH})
    else()
        message(WARNING "File ${SVGICON_LIB_PATH} does not exist.")
    endif()

    # for ACloudViewer deps
    set(EXTERNAL_LIB_DIR2 ${EXTERNAL_LIB_DIR} ${BUILD_LIB_PATH})
    execute_process(COMMAND bash ${PACK_SCRIPTS}
                    "${BUILD_LIB_PATH}/${MAIN_APP_NAME}${APP_EXTENSION}"
                    ${DEPLOY_LIB_PATH}
                    ${EXTERNAL_LIB_DIR2}
                    WORKING_DIRECTORY ${BUILD_LIB_PATH})

    if(AICore_BUNDLE_CUDA_RUNTIME AND AICore_CUDA_ENABLED)
        message(STATUS
            "AICore_BUNDLE_CUDA_RUNTIME bundles libcublas.so.* into the "
            "installer so CUDA inference works on driver-only machines. "
            "Without this, the shipped libggml-cuda.so will silently fail" 
            "to load on machines without the CUDA toolkit (missing "
            "libcublas.so.11).")
        set(_bundled_cuda_src "${CMAKE_INSTALL_PREFIX}/${LIBS_FOLDER_NAME}/cuda-runtime")
        set(_bundled_cuda_dst "${DEPLOY_LIB_PATH}/cuda-runtime")
        if(EXISTS "${_bundled_cuda_src}")
            file(COPY "${_bundled_cuda_src}/"
                DESTINATION "${_bundled_cuda_dst}"
                USE_SOURCE_PERMISSIONS)
            message(STATUS "Deployed bundled CUDA runtime to ${_bundled_cuda_dst}")
        else()
            message(WARNING "AICore_BUNDLE_CUDA_RUNTIME=ON but ${_bundled_cuda_src} is missing")
        endif()
    endif()

elseif (WIN32)
    # deploy plugins and translations for ACloudViewer
    file(COPY 
        "${SOURCE_BIN_PATH}/${MAIN_APP_NAME}/plugins"
        "${SOURCE_BIN_PATH}/${MAIN_APP_NAME}/translations"
        DESTINATION "${MAIN_DEPLOY_PATH}"
        USE_SOURCE_PERMISSIONS
        )

    # deploy SIBR plugin runtime assets (shaders, resources, config)
    foreach(_sibr_asset shaders sibr_resources)
        set(_sibr_asset_path "${SOURCE_BIN_PATH}/${_sibr_asset}")
        if(EXISTS "${_sibr_asset_path}")
            file(COPY "${_sibr_asset_path}"
                DESTINATION "${MAIN_DEPLOY_PATH}"
                USE_SOURCE_PERMISSIONS)
        endif()
    endforeach()
    set(_ibr_ini "${SOURCE_BIN_PATH}/ibr_resources.ini")
    if(EXISTS "${_ibr_ini}")
        file(COPY "${_ibr_ini}"
            DESTINATION "${MAIN_DEPLOY_PATH}"
            USE_SOURCE_PERMISSIONS)
    endif()

    if (${PLUGIN_PYTHON} STREQUAL "ON")
        file(COPY 
            "${SOURCE_BIN_PATH}/${MAIN_APP_NAME}/plugins-python"
            DESTINATION "${MAIN_DEPLOY_PATH}"
            USE_SOURCE_PERMISSIONS
        )
    endif()

    # prepare search path for powershell
    set(EXTERNAL_DLL_DIR ${EXTERNAL_INSTALL_DIRS})
    if (${BUILD_WITH_CONDA} STREQUAL "ON")
        list(APPEND EXTERNAL_DLL_DIR ${CONDA_PREFIX}/Library/bin)
    endif()
    message(STATUS "Start search dependency from path: ${EXTERNAL_DLL_DIR}")
    string(REPLACE ";" "\",\"" PS_SEARCH_PATHS "${EXTERNAL_DLL_DIR}")
    set(PS_SEARCH_PATHS "\"${PS_SEARCH_PATHS}\"")
    message(STATUS "PS_SEARCH_PATHS: ${PS_SEARCH_PATHS}")
    # find powershell program
    find_program(POWERSHELL_PATH NAMES powershell pwsh)
    if(NOT POWERSHELL_PATH)
        message(FATAL_ERROR "PowerShell not found!")
    endif()
    # search dependency for ACloudViewer, CloudViewer and Colmap
    execute_process(
        COMMAND ${POWERSHELL_PATH} -ExecutionPolicy Bypass 
                -Command "& '${PACK_SCRIPTS}' '${SOURCE_BIN_PATH}/${MAIN_APP_NAME}' '${DEPLOY_LIB_PATH}' @(${PS_SEARCH_PATHS}) -Recursive"
    )

    if(AICore_BUNDLE_CUDA_RUNTIME AND AICore_CUDA_ENABLED)
        set(_bundled_cuda_src "${CMAKE_INSTALL_PREFIX}/${LIBS_FOLDER_NAME}/cuda-runtime")
        set(_bundled_cuda_dst "${DEPLOY_LIB_PATH}/cuda-runtime")
        if(EXISTS "${_bundled_cuda_src}")
            file(COPY "${_bundled_cuda_src}/"
                DESTINATION "${_bundled_cuda_dst}"
                USE_SOURCE_PERMISSIONS)
            message(STATUS "Deployed bundled CUDA runtime to ${_bundled_cuda_dst}")
        else()
            message(WARNING "AICore_BUNDLE_CUDA_RUNTIME=ON but ${_bundled_cuda_src} is missing")
        endif()
    endif()
endif()

## deploy CloudViewer
if (${BUILD_GUI} STREQUAL "ON")
    file(COPY "${SOURCE_BIN_PATH}/${CLOUDVIEWER_APP_NAME}/${CLOUDVIEWER_APP_NAME}${APP_EXTENSION}"
        DESTINATION "${CLOUDVIEWER_DEPLOY_PATH}"
        USE_SOURCE_PERMISSIONS)
    if (APPLE AND GGML_MODULE_SUFFIX)
        ensure_ggml_backends_in_app(
            "${CLOUDVIEWER_DEPLOY_PATH}/${CLOUDVIEWER_APP_NAME}${APP_EXTENSION}/Contents/${LIBS_FOLDER_NAME}"
            "${_GGML_SRC_DIR}" "${GGML_MODULE_SUFFIX}")
        ensure_ggml_backends_in_app(
            "${SOURCE_BIN_PATH}/${CLOUDVIEWER_APP_NAME}/${CLOUDVIEWER_APP_NAME}${APP_EXTENSION}/Contents/${LIBS_FOLDER_NAME}"
            "${_GGML_SRC_DIR}" "${GGML_MODULE_SUFFIX}")
    endif()
    if ((WIN32 OR UNIX) AND NOT APPLE)
        file(COPY "${SOURCE_BIN_PATH}/${CLOUDVIEWER_APP_NAME}/resources"
                DESTINATION "${CLOUDVIEWER_DEPLOY_PATH}"
                USE_SOURCE_PERMISSIONS)
    endif()
endif()
## deploy Colmap
if (${BUILD_RECONSTRUCTION} STREQUAL "ON")
    file(COPY "${SOURCE_BIN_PATH}/${COLMAP_APP_NAME}/${COLMAP_APP_NAME}${APP_EXTENSION}"
                DESTINATION "${COLMAP_DEPLOY_PATH}"
                USE_SOURCE_PERMISSIONS)
    if (APPLE AND GGML_MODULE_SUFFIX)
        ensure_ggml_backends_in_app(
            "${COLMAP_DEPLOY_PATH}/${COLMAP_APP_NAME}${APP_EXTENSION}/Contents/${LIBS_FOLDER_NAME}"
            "${_GGML_SRC_DIR}" "${GGML_MODULE_SUFFIX}")
        ensure_ggml_backends_in_app(
            "${SOURCE_BIN_PATH}/${COLMAP_APP_NAME}/${COLMAP_APP_NAME}${APP_EXTENSION}/Contents/${LIBS_FOLDER_NAME}"
            "${_GGML_SRC_DIR}" "${GGML_MODULE_SUFFIX}")
    endif()

    if (UNIX AND NOT APPLE)
        # for Colmap deps
        if(EXISTS "${LINK_GFLAGS_FILE_PATH}")
            file(COPY "${LINK_GFLAGS_FILE_PATH}"
                        DESTINATION "${DEPLOY_LIB_PATH}"
                        USE_SOURCE_PERMISSIONS)
            message(STATUS "Copied ${LINK_GFLAGS_FILE_PATH} to ${DEPLOY_LIB_PATH}")
        else()
            message(WARNING "File ${LINK_GFLAGS_FILE_PATH} does not exist.")
        endif()
    endif()
endif()

## 2.5. Patch version in deployed data files (.desktop, config, etc.)
set(MAIN_DESKTOP "${DEPLOY_PACKAGES_PATH}/${MAIN_APP_NAME}/data/${MAIN_APP_NAME}.desktop")
if (EXISTS "${MAIN_DESKTOP}")
    replace_version_in_file("${MAIN_DESKTOP}")
endif()
if (${BUILD_GUI} STREQUAL "ON")
    set(CV_DESKTOP "${DEPLOY_PACKAGES_PATH}/${CLOUDVIEWER_APP_NAME}/data/${CLOUDVIEWER_APP_NAME}.desktop")
    if (EXISTS "${CV_DESKTOP}")
        replace_version_in_file("${CV_DESKTOP}")
    endif()
endif()

## 3. Package
set(OUTPUT_CLOUDVIEWER_PACKAGE_PATH ${CMAKE_INSTALL_PREFIX}/${ACLOUDVIEWER_PACKAGE_NAME}.${PACKAGE_EXTENSION})
if (${PACKAGE} STREQUAL "ON") # package
    # PACKAGE_TOOL is set by install(CODE) from CMakeLists.txt find_program().
    # Fallback to "binarycreator" only if it wasn't resolved at configure time.
    if (NOT PACKAGE_TOOL OR PACKAGE_TOOL MATCHES "-NOTFOUND$")
        set(PACKAGE_TOOL "binarycreator")
    endif()
    if (APPLE)
        # Create the Qt IFW installer .app, then wrap in a DMG.
        # The IFW installerbase is a .app bundle (with Frameworks embedded via
        # macdeployqt), so binarycreator creates a nested bundle that we flatten.
        set(_QTIFW_APP_PATH "${CMAKE_INSTALL_PREFIX}/${ACLOUDVIEWER_PACKAGE_NAME}.app")
        set(_QTIFW_BUNDLE_EXE "${ACLOUDVIEWER_PACKAGE_NAME}")
        message(STATUS "Running: ${PACKAGE_TOOL} -c ${CONFIG_FILE_PATH} -p ${DEPLOY_PACKAGES_PATH} ${_QTIFW_APP_PATH}")
        message(STATUS "  Working directory: ${MAIN_WORKING_DIRECTORY}")
        execute_process(COMMAND ${PACKAGE_TOOL}
            -c ${CONFIG_FILE_PATH} -p ${DEPLOY_PACKAGES_PATH}
            "${_QTIFW_APP_PATH}"
            WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY}
            RESULT_VARIABLE _QTIFW_RESULT
            OUTPUT_VARIABLE _QTIFW_STDOUT
            ERROR_VARIABLE _QTIFW_STDERR)
        if(_QTIFW_STDOUT)
            message(STATUS "binarycreator stdout: ${_QTIFW_STDOUT}")
        endif()
        if(_QTIFW_STDERR)
            message(WARNING "binarycreator stderr: ${_QTIFW_STDERR}")
        endif()
        if(_QTIFW_RESULT)
            message(FATAL_ERROR
                "binarycreator failed!\n"
                "  Command: ${PACKAGE_TOOL} -c ${CONFIG_FILE_PATH} -p ${DEPLOY_PACKAGES_PATH} ${_QTIFW_APP_PATH}\n"
                "  Working directory: ${MAIN_WORKING_DIRECTORY}\n"
                "  Exit code: ${_QTIFW_RESULT}\n"
                "  stderr: ${_QTIFW_STDERR}")
        endif()

        # Fix nested bundle: if CFBundleExecutable is a directory (Qt IFW creates
        # a nested .app when the IFW installerbase is itself a .app bundle).
        # Flatten by moving the inner binary to the expected path AND promoting
        # Frameworks/PlugIns from the nested bundle to the top-level .app.
        set(_BUNDLE_MACOS "${_QTIFW_APP_PATH}/Contents/MacOS")
        set(_BUNDLE_CONTENTS "${_QTIFW_APP_PATH}/Contents")
        set(_BUNDLE_EXE "${_BUNDLE_MACOS}/${_QTIFW_BUNDLE_EXE}")
        if(IS_DIRECTORY "${_BUNDLE_EXE}")
            # Find the real binary inside the nested bundle
            set(_NESTED_BIN "${_BUNDLE_EXE}/Contents/MacOS/installerbase")
            if(NOT EXISTS "${_NESTED_BIN}")
                file(GLOB _NESTED_BIN "${_BUNDLE_EXE}/Contents/MacOS/*")
                list(GET _NESTED_BIN 0 _NESTED_BIN)
            endif()
            if(EXISTS "${_NESTED_BIN}")
                message(STATUS "Fixing nested Qt IFW bundle: ${_NESTED_BIN} -> ${_BUNDLE_EXE}")

                # Promote Frameworks/ from nested bundle to top-level .app
                set(_NESTED_FW "${_BUNDLE_EXE}/Contents/Frameworks")
                if(IS_DIRECTORY "${_NESTED_FW}")
                    message(STATUS "Promoting Frameworks/ from nested bundle to ${_BUNDLE_CONTENTS}/Frameworks/")
                    file(COPY "${_NESTED_FW}" DESTINATION "${_BUNDLE_CONTENTS}")
                endif()

                # Promote PlugIns/ from nested bundle to top-level .app
                set(_NESTED_PLUGINS "${_BUNDLE_EXE}/Contents/PlugIns")
                if(IS_DIRECTORY "${_NESTED_PLUGINS}")
                    message(STATUS "Promoting PlugIns/ from nested bundle to ${_BUNDLE_CONTENTS}/PlugIns/")
                    file(COPY "${_NESTED_PLUGINS}" DESTINATION "${_BUNDLE_CONTENTS}")
                endif()

                # Move the inner binary to the expected path
                set(_TMP_BIN "${_BUNDLE_MACOS}/_installerbase_tmp")
                file(COPY "${_NESTED_BIN}" DESTINATION "${_BUNDLE_MACOS}"
                     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                     GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
                get_filename_component(_NESTED_NAME "${_NESTED_BIN}" NAME)
                file(RENAME "${_BUNDLE_MACOS}/${_NESTED_NAME}" "${_TMP_BIN}")
                file(REMOVE_RECURSE "${_BUNDLE_EXE}")
                # Also remove the .QaTwuZ temporary directory if present
                file(GLOB _QTIFW_TEMPS "${_BUNDLE_MACOS}/${_QTIFW_BUNDLE_EXE}.*")
                foreach(_tmp IN LISTS _QTIFW_TEMPS)
                    if(IS_DIRECTORY "${_tmp}")
                        file(REMOVE_RECURSE "${_tmp}")
                    endif()
                endforeach()
                file(RENAME "${_TMP_BIN}" "${_BUNDLE_EXE}")
                message(STATUS "Fixed macOS installer bundle executable")
            else()
                message(WARNING "Cannot fix nested bundle: no binary found under ${_BUNDLE_EXE}")
            endif()
        endif()

        # Ad-hoc sign the installer app
        execute_process(COMMAND codesign --deep --force -s - --timestamp "${_QTIFW_APP_PATH}"
            RESULT_VARIABLE _SIGN_RESULT)
        if(_SIGN_RESULT)
            message(WARNING "Ad-hoc signing failed (code ${_SIGN_RESULT}), installer may not launch on some macOS versions")
        endif()

        # ── Create a polished DMG with background, icon, and Finder layout ──
        set(_DMG_PATH "${CMAKE_INSTALL_PREFIX}/${ACLOUDVIEWER_PACKAGE_NAME}.dmg")
        set(_DMG_VOL_NAME "${ACLOUDVIEWER_PACKAGE_NAME}")
        set(_DMG_RW_PATH "${CMAKE_INSTALL_PREFIX}/.tmp_installer_rw.dmg")
        set(_DMG_STAGING "${CMAKE_INSTALL_PREFIX}/.dmg_staging")

        # Locate DMG resources (background image + volume icon)
        # Config files are deployed to ${DEPLOY_ROOT_PATH}/config/ during install
        set(_DMG_RES_DIR "${MAIN_WORKING_DIRECTORY}/config")

        # 1. Build staging directory with .app + background
        file(REMOVE_RECURSE "${_DMG_STAGING}")
        file(MAKE_DIRECTORY "${_DMG_STAGING}/.background")
        file(COPY "${_QTIFW_APP_PATH}" DESTINATION "${_DMG_STAGING}")

        # Ensure the installer .app has the correct icon in its Resources.
        # IFW's Info.plist references CFBundleIconFile as <PackageName>.icns,
        # so we must copy the icon with that exact name.
        set(_STAGING_APP_RES "${_DMG_STAGING}/${ACLOUDVIEWER_PACKAGE_NAME}.app/Contents/Resources")
        set(_STAGING_APP_PLIST "${_DMG_STAGING}/${ACLOUDVIEWER_PACKAGE_NAME}.app/Contents/Info.plist")
        set(_DMG_ICON_SRC "${_DMG_RES_DIR}/logo_256.icns")
        if(EXISTS "${_DMG_ICON_SRC}" AND IS_DIRECTORY "${_STAGING_APP_RES}")
            # Extract the actual CFBundleIconFile name from Info.plist
            file(READ "${_STAGING_APP_PLIST}" _PLIST_CONTENT)
            string(REGEX MATCH "<key>CFBundleIconFile</key>[ \t\n]*<string>([^<]+)</string>" _ICON_MATCH "${_PLIST_CONTENT}")
            if(CMAKE_MATCH_1)
                set(_ICON_FILENAME "${CMAKE_MATCH_1}")
            else()
                # Fallback: use package name
                set(_ICON_FILENAME "${ACLOUDVIEWER_PACKAGE_NAME}.icns")
            endif()
            file(COPY "${_DMG_ICON_SRC}" DESTINATION "${_STAGING_APP_RES}")
            file(RENAME "${_STAGING_APP_RES}/logo_256.icns" "${_STAGING_APP_RES}/${_ICON_FILENAME}")
            # Re-sign so Finder picks up the new icon
            execute_process(COMMAND codesign --deep --force -s - "${_DMG_STAGING}/${ACLOUDVIEWER_PACKAGE_NAME}.app"
                OUTPUT_QUIET ERROR_QUIET)
            message(STATUS "DMG: refreshed .app icon as ${_ICON_FILENAME} (from ${_DMG_ICON_SRC})")
        endif()

        # Copy background image if available
        set(_DMG_BG "${_DMG_RES_DIR}/dmg_background.png")
        if(EXISTS "${_DMG_BG}")
            file(COPY "${_DMG_BG}" DESTINATION "${_DMG_STAGING}/.background")
            message(STATUS "DMG background: ${_DMG_BG}")
        endif()

        # 2. Use dmgbuild to create a polished DMG with background, icon positions, and window size.
        #    dmgbuild handles the full flow: create writable DMG → mount → AppleScript layout →
        #    unmount → convert to compressed read-only. This is the only reliable way to get
        #    Finder beautification on modern macOS (where .DS_Store injection alone doesn't work).
        set(_APP_NAME_IN_DMG "${ACLOUDVIEWER_PACKAGE_NAME}.app")
        set(_DMGBUILD_SETTINGS "${CMAKE_INSTALL_PREFIX}/.tmp_dmgbuild_settings.py")

        # Find Python with dmgbuild package
        set(_PYTHON_CMD "${_PYTHON_EXECUTABLE}")
        if(NOT _PYTHON_CMD)
            find_program(_PYTHON_CMD python3)
        endif()
        set(_DMGBUILD_OK FALSE)

        if(_PYTHON_CMD AND EXISTS "${_DMG_BG}")
            # Write dmgbuild settings file
            # Note: dmgbuild automatically copies the background image into the DMG
            # as '.background.png'. We only need to specify the installer .app in 'files'.
            # Window size matches the background image (640x360) so it fills perfectly.
            # No Applications symlink: this is an installer DMG, users double-click the .app.
            file(WRITE "${_DMGBUILD_SETTINGS}"
                "import os\n"
                "appname = '${_APP_NAME_IN_DMG}'\n"
                "icon_size = 96\n"
                "window_rect = ((200, 100), (640, 440))\n"
                "background = '${_DMG_STAGING}/.background/dmg_background.png'\n"
                "show_status_bar = False\n"
                "show_toolbar = False\n"
                "show_sidebar = False\n"
                "show_pathbar = False\n"
                "files = ['${_DMG_STAGING}/${_APP_NAME_IN_DMG}']\n"
                "icon_locations = {\n"
                "    appname: (320, 120),\n"
                "    '.background': (1000, 200),\n"
                "}\n")

            # First detach any stale volumes with the same name
            execute_process(COMMAND hdiutil detach "/Volumes/${_DMG_VOL_NAME}" -force OUTPUT_QUIET ERROR_QUIET)
            foreach(_suffix 1 2 3)
                execute_process(COMMAND hdiutil detach "/Volumes/${_DMG_VOL_NAME} ${_suffix}" -force OUTPUT_QUIET ERROR_QUIET)
            endforeach()

            # Run dmgbuild
            message(STATUS "DMG: running dmgbuild to create polished DMG: ${_DMG_PATH}")
            execute_process(
                COMMAND "${_PYTHON_CMD}" -m dmgbuild
                    -s "${_DMGBUILD_SETTINGS}"
                    "${_DMG_VOL_NAME}"
                    "${_DMG_PATH}"
                RESULT_VARIABLE _DMGBUILD_RESULT
                OUTPUT_VARIABLE _DMGBUILD_STDOUT
                ERROR_VARIABLE _DMGBUILD_STDERR
                TIMEOUT 120)
            file(REMOVE "${_DMGBUILD_SETTINGS}")

            if(_DMGBUILD_RESULT EQUAL 0)
                message(STATUS "DMG: dmgbuild succeeded: ${_DMGBUILD_STDOUT}")
                set(_DMGBUILD_OK TRUE)
            else()
                message(STATUS "DMG: dmgbuild failed (code ${_DMGBUILD_RESULT}): ${_DMGBUILD_STDERR}")
                message(STATUS "DMG: falling back to plain hdiutil approach")
                set(_DMGBUILD_OK FALSE)
            endif()
        endif()

        # Fallback: plain hdiutil approach (no beautification, but functional DMG)
        if(NOT _DMGBUILD_OK)
            message(STATUS "Creating plain DMG (no beautification): ${_DMG_PATH}")
            execute_process(COMMAND
                hdiutil create -srcfolder "${_DMG_STAGING}"
                -volname "${_DMG_VOL_NAME}" -ov -format UDZO "${_DMG_PATH}"
                RESULT_VARIABLE _DMG_RESULT
                OUTPUT_VARIABLE _DMG_STDOUT ERROR_VARIABLE _DMG_STDERR)
            if(_DMG_RESULT)
                message(FATAL_ERROR "hdiutil create failed:\n  code: ${_DMG_RESULT}\n  stderr: ${_DMG_STDERR}")
            endif()
        endif()

        # 3. Cleanup temporary files
        file(REMOVE_RECURSE "${_DMG_STAGING}")
        file(REMOVE "${_DMG_RW_PATH}")
        file(REMOVE_RECURSE "${_QTIFW_APP_PATH}")

        message(STATUS "${MAIN_APP_NAME} Installer DMG created: ${_DMG_PATH}")
        message(STATUS "Open the DMG and double-click the .app to launch the installer.")
    else()
        # Linux / Windows: create installer directly
        set(SHELL_CMD "${PACKAGE_TOOL} -c ${CONFIG_FILE_PATH} -p ${DEPLOY_PACKAGES_PATH} ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}")
        message(STATUS "Package with command: " ${SHELL_CMD})
        execute_process(COMMAND ${PACKAGE_TOOL} -c ${CONFIG_FILE_PATH} -p ${DEPLOY_PACKAGES_PATH} ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}
                        WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY}
                        RESULT_VARIABLE _ifw_result
                        OUTPUT_VARIABLE _ifw_stdout
                        ERROR_VARIABLE _ifw_stderr)
        if(_ifw_stdout)
            message(STATUS "${PACKAGE_TOOL} stdout: ${_ifw_stdout}")
        endif()
        if(_ifw_stderr)
            message(WARNING "${PACKAGE_TOOL} stderr: ${_ifw_stderr}")
        endif()
        if(_ifw_result)
            message(FATAL_ERROR
                "${PACKAGE_TOOL} failed!\n"
                "  Command: ${SHELL_CMD}\n"
                "  Working directory: ${MAIN_WORKING_DIRECTORY}\n"
                "  Exit code: ${_ifw_result}\n"
                "  stderr: ${_ifw_stderr}")
        endif()
        if(NOT EXISTS "${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}")
            message(FATAL_ERROR "${PACKAGE_TOOL} did not produce expected output: ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}")
        endif()
        message(STATUS "${MAIN_APP_NAME} Installer Package ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH} created.")
    endif()
else() # Do not package
    message(STATUS "Continue to publish installer package: cd ${MAIN_WORKING_DIRECTORY}.")
    message(STATUS "Then please execute: ${PACKAGE_TOOL} -c ${CONFIG_FILE_PATH} -p packages ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}")
endif()