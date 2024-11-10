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
    # deploy c++ library dependency
    execute_process(COMMAND bash ${PACK_SCRIPTS}
                    "${BUILD_LIB_PATH}" ${DEPLOY_LIB_PATH}
                    WORKING_DIRECTORY ${BUILD_LIB_PATH})
    execute_process(COMMAND bash ${PACK_SCRIPTS}
                    "${BUILD_LIB_PATH}/plugins" ${DEPLOY_LIB_PATH}
                    WORKING_DIRECTORY ${BUILD_LIB_PATH})
    execute_process(COMMAND bash ${PACK_SCRIPTS}
                    ${MAIN_DEPLOY_PATH}/platforms/libqxcb.so ${DEPLOY_LIB_PATH}
                    WORKING_DIRECTORY ${MAIN_DEPLOY_PATH})

elseif (WIN32)
    # deploy plugins and translations for ACloudViewer
    file(COPY 
        "${SOURCE_BIN_PATH}/${MAIN_APP_NAME}/plugins"
        "${SOURCE_BIN_PATH}/${MAIN_APP_NAME}/translations"
        DESTINATION "${MAIN_DEPLOY_PATH}"
        USE_SOURCE_PERMISSIONS
        )

    # prepare search path for powershell
    set(EXTERNAL_DLL_DIR ${EXTERNAL_INSTALL_DIRS} ${CONDA_PREFIX}/Library/bin)
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
                -Command "& '${PACK_SCRIPTS}' '${SOURCE_BIN_PATH}/${MAIN_APP_NAME}' '${DEPLOY_LIB_PATH}' @(${PS_SEARCH_PATHS})"
    )
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
    # fix gflags issues
    if (UNIX AND NOT APPLE)
        copy_rename_files(
            "${EXTERNAL_INSTALL_DIRS}/lib"
            "${GFLAGS_SRC_FILENAME}"
            "${DEPLOY_LIB_PATH}"
            "${GFLAGS_DST_FILENAME}"
        )
    endif()
endif()

## 3. Package
set(OUTPUT_CLOUDVIEWER_PACKAGE_PATH ${CMAKE_INSTALL_PREFIX}/${ACLOUDVIEWER_PACKAGE_NAME}.${PACKAGE_EXTENSION})
if (${PACKAGE} STREQUAL "ON") # package
    set(PACKAGE_TOOL "binarycreator")
    set(SHELL_CMD "${PACKAGE_TOOL} -c ${CONFIG_FILE_PATH} -p ${DEPLOY_PACKAGES_PATH} ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}")
    message(STATUS "Package with command: " ${SHELL_CMD})
    execute_process(COMMAND ${PACKAGE_TOOL} -c ${CONFIG_FILE_PATH} -p ${DEPLOY_PACKAGES_PATH} ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}
                    WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})
    message(STATUS "${MAIN_APP_NAME} Installer Package ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH} created.")
    # execute_process(COMMAND zip -r ${CMAKE_INSTALL_PREFIX}/${ACLOUDVIEWER_PACKAGE_NAME}.zip ${DEPLOY_ROOT_PATH}
    #                 WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX})
    # message(STATUS "Package ${CMAKE_INSTALL_PREFIX}/${ACLOUDVIEWER_PACKAGE_NAME}.zip created")
else() # Do not package
    message(STATUS "Continue to publish installer package: cd ${MAIN_WORKING_DIRECTORY}.")
    message(STATUS "Then please execute: ${PACKAGE_TOOL} -c ${CONFIG_FILE_PATH} -p packages ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}")
endif()