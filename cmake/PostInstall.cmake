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
## deploy ACloudViewer
file(COPY "${SOURCE_BIN_PATH}/${MAIN_APP_NAME}/${MAIN_APP_NAME}${APP_EXTENSION}"
    DESTINATION "${MAIN_DEPLOY_PATH}"
    USE_SOURCE_PERMISSIONS)
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
    set(PACKAGE_TOOL "binarycreator")
    if (APPLE)
        # Qt IFW ≥4.x on macOS may create a nested app bundle where
        # CFBundleExecutable points to a directory instead of a binary.
        # Work around: create the .app first, fix the bundle, then wrap in DMG.
        set(_QTIFW_APP_PATH "${CMAKE_INSTALL_PREFIX}/${ACLOUDVIEWER_PACKAGE_NAME}.app")
        set(_QTIFW_BUNDLE_EXE "${ACLOUDVIEWER_PACKAGE_NAME}")
        message(STATUS "Creating macOS installer app: ${_QTIFW_APP_PATH}")
        execute_process(COMMAND ${PACKAGE_TOOL}
            -c ${CONFIG_FILE_PATH} -p ${DEPLOY_PACKAGES_PATH}
            "${_QTIFW_APP_PATH}"
            WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY}
            RESULT_VARIABLE _QTIFW_RESULT)
        if(_QTIFW_RESULT)
            message(FATAL_ERROR "binarycreator failed with code ${_QTIFW_RESULT}")
        endif()

        # Fix nested bundle: if CFBundleExecutable is a directory (Qt IFW bug),
        # flatten it by moving the inner installerbase binary to the expected path.
        set(_BUNDLE_MACOS "${_QTIFW_APP_PATH}/Contents/MacOS")
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

        # Create compressed DMG from the fixed app bundle
        message(STATUS "Creating DMG: ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}")
        file(REMOVE "${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}")
        execute_process(COMMAND hdiutil create
            -volname "${ACLOUDVIEWER_PACKAGE_NAME}"
            -srcfolder "${_QTIFW_APP_PATH}"
            -ov -format UDZO
            "${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}"
            RESULT_VARIABLE _DMG_RESULT)
        if(_DMG_RESULT)
            message(FATAL_ERROR "hdiutil create failed with code ${_DMG_RESULT}")
        endif()
        message(STATUS "${MAIN_APP_NAME} Installer Package ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH} created.")
    else()
        # Linux / Windows: create installer directly
        set(SHELL_CMD "${PACKAGE_TOOL} -c ${CONFIG_FILE_PATH} -p ${DEPLOY_PACKAGES_PATH} ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}")
        message(STATUS "Package with command: " ${SHELL_CMD})
        execute_process(COMMAND ${PACKAGE_TOOL} -c ${CONFIG_FILE_PATH} -p ${DEPLOY_PACKAGES_PATH} ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}
                        WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})
        message(STATUS "${MAIN_APP_NAME} Installer Package ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH} created.")
    endif()
else() # Do not package
    message(STATUS "Continue to publish installer package: cd ${MAIN_WORKING_DIRECTORY}.")
    message(STATUS "Then please execute: ${PACKAGE_TOOL} -c ${CONFIG_FILE_PATH} -p packages ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}")
endif()