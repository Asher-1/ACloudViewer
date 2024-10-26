if (APPLE)
    set(APP_EXTENSION "dmg")
    set(CONFIG_POSTFIX "mac")
    set(PACK_SCRIPTS ${PACK_SCRIPTS_PATH}/pack_macosx_bundle.sh)
elseif (UNIX)
    set(APP_EXTENSION "run")
    set(CONFIG_POSTFIX "linux")
    set(PACK_SCRIPTS ${PACK_SCRIPTS_PATH}/pack_ubuntu.sh)
elseif(WIN32)
    set(APP_EXTENSION "exe")
    set(CONFIG_POSTFIX "win")
    set(PACK_SCRIPTS ${PACK_SCRIPTS_PATH}/pack_windows.bat)
endif()

set(CONFIG_FILE_PATH config/config_${CONFIG_POSTFIX}.xml)
set(MAIN_WORKING_DIRECTORY ${DEPLOY_ROOT_PATH})
set(MAIN_DEPLOY_PATH ${DEPLOY_ROOT_PATH}/packages/${MAIN_APP_NAME}/data)
set(COLMAP_DEPLOY_PATH ${DEPLOY_ROOT_PATH}/packages/${COLMAP_APP_NAME}/data)
set(CLOUDVIEWER_DEPLOY_PATH ${DEPLOY_ROOT_PATH}/packages/${CLOUDVIEWER_APP_NAME}/data)
set(DEPLOY_LIB_PATH ${MAIN_DEPLOY_PATH}/${LIBS_FOLDER_NAME})
if (APPLE)
    ## update version and build time
    execute_process(COMMAND sed -i "" "s/3.9.0/${CLOUDVIEWER_VERSION}/g"
                    ${CONFIG_FILE_PATH} packages/${MAIN_APP_NAME}/meta/package.xml packages/${MAIN_APP_NAME}/meta/installscript.qs
                    WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})
    execute_process(COMMAND sed -i "" "s/2024-09-18/${BUILD_TIME}/g"
                    packages/${MAIN_APP_NAME}/meta/package.xml
                    WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})
    # deploy ACloudViewer
    execute_process(COMMAND cp -r ${CMAKE_INSTALL_PREFIX}/${MAIN_APP_NAME}/${MAIN_APP_NAME}.app 
                    ${MAIN_DEPLOY_PATH} WORKING_DIRECTORY ${MAIN_DEPLOY_PATH})
    # execute_process(COMMAND python ${APP_SIGN_SCRIPT_PATH} ${MAIN_APP_NAME} ${MAIN_DEPLOY_PATH})
    
    # deploy and sign CloudViewer
    if (${BUILD_GUI} STREQUAL "ON")
        execute_process(COMMAND sed -i "" "s/3.9.0/${CLOUDVIEWER_VERSION}/g" packages/${CLOUDVIEWER_APP_NAME}/meta/package.xml 
                        WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})
        execute_process(COMMAND sed -i "" "s/2024-09-18/${BUILD_TIME}/g" packages/${CLOUDVIEWER_APP_NAME}/meta/package.xml
                        WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})
        execute_process(COMMAND cp -r ${CMAKE_INSTALL_PREFIX}/bin/${CLOUDVIEWER_APP_NAME}/${CLOUDVIEWER_APP_NAME}.app 
                        ${CLOUDVIEWER_DEPLOY_PATH} WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})
        # execute_process(COMMAND python ${APP_SIGN_SCRIPT_PATH} ${CLOUDVIEWER_APP_NAME} ${CLOUDVIEWER_DEPLOY_PATH})
    endif()

    # deploy and sig colmap
    if (${BUILD_RECONSTRUCTION} STREQUAL "ON")
        execute_process(COMMAND sed -i "" "s/3.9.0/${CLOUDVIEWER_VERSION}/g" packages/${COLMAP_APP_NAME}/meta/package.xml 
                        WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})
        execute_process(COMMAND sed -i "" "s/2024-09-18/${BUILD_TIME}/g" packages/${COLMAP_APP_NAME}/meta/package.xml
                        WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})
        execute_process(COMMAND cp -r ${CMAKE_INSTALL_PREFIX}/bin/${COLMAP_APP_NAME}/${COLMAP_APP_NAME}.app 
                        ${COLMAP_DEPLOY_PATH} WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})
        # execute_process(COMMAND python ${APP_SIGN_SCRIPT_PATH} ${COLMAP_APP_NAME} ${COLMAP_DEPLOY_PATH})
    endif()
elseif (UNIX)
    ## update version and build time
    execute_process(COMMAND sed -i "s/3.9.0/${CLOUDVIEWER_VERSION}/g"
                    ${CONFIG_FILE_PATH}
                    packages/${MAIN_APP_NAME}/meta/package.xml
                    packages/${MAIN_APP_NAME}/meta/installscript.qs
                    WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})
    execute_process(COMMAND sed -i "s/2024-09-18/${BUILD_TIME}/g" 
                    packages/${MAIN_APP_NAME}/meta/package.xml 
                    WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})

    # install ACloudViewer app
    execute_process(COMMAND cp -r ${CMAKE_INSTALL_PREFIX}/${MAIN_APP_NAME}  
                    ${MAIN_DEPLOY_PATH} WORKING_DIRECTORY ${MAIN_DEPLOY_PATH})
    # deploy extra libs, plugins and translations
    execute_process(COMMAND mv ${CMAKE_INSTALL_PREFIX}/${LIBS_FOLDER_NAME}
                    ${CMAKE_INSTALL_PREFIX}/plugins ${CMAKE_INSTALL_PREFIX}/translations
                    ${MAIN_DEPLOY_PATH} WORKING_DIRECTORY ${MAIN_DEPLOY_PATH})
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

    if (${BUILD_RECONSTRUCTION} STREQUAL "ON")
        # install colmap app
        execute_process(COMMAND cp -r ${CMAKE_INSTALL_PREFIX}/${COLMAP_APP_NAME} 
                        ${MAIN_DEPLOY_PATH} WORKING_DIRECTORY ${MAIN_DEPLOY_PATH})
        # fix gflags issues
        execute_process(COMMAND cp "${EXTERNAL_INSTALL_DIR}/lib/${GFLAGS_SRC_FILENAME}"
                        ${DEPLOY_LIB_PATH}/${GFLAGS_DST_FILENAME}
                        WORKING_DIRECTORY ${BUILD_LIB_PATH})
    endif()
elseif (WIN32)
endif()

set(OUTPUT_CLOUDVIEWER_PACKAGE_PATH ${CMAKE_INSTALL_PREFIX}/${CLOUDVIEWER_PACKAGE_NAME}.${APP_EXTENSION})
if (${PACKAGE} STREQUAL "ON") # package
    if (APPLE) # already packaged somewhere
        set(SHELL_CMD "binarycreator -c ${CONFIG_FILE_PATH} -p packages ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}")
        message(STATUS "Package with command: " ${SHELL_CMD})
        execute_process(COMMAND binarycreator -c ${CONFIG_FILE_PATH} -p packages ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}
                        WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})
        message(STATUS "${MAIN_APP_NAME} MacOS Installer Package ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH} created.")
    elseif (UNIX)
        set(SHELL_CMD "binarycreator -c ${CONFIG_FILE_PATH} -p packages ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}")
        message(STATUS "Package with command: " ${SHELL_CMD})
        execute_process(COMMAND binarycreator -c ${CONFIG_FILE_PATH} -p packages ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}
                        WORKING_DIRECTORY ${MAIN_WORKING_DIRECTORY})
        message(STATUS "${MAIN_APP_NAME} Linux Installer Package ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH} created.")
        # execute_process(COMMAND zip -r ${CMAKE_INSTALL_PREFIX}/${CLOUDVIEWER_PACKAGE_NAME}.zip ${DEPLOY_ROOT_PATH}
        #                 WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX})
        # message(STATUS "Package ${CMAKE_INSTALL_PREFIX}/${CLOUDVIEWER_PACKAGE_NAME}.zip created")
    elseif (WIN32)
        message(STATUS "${MAIN_APP_NAME} Windows Installer Package ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH} created.")
    endif()
else() # Do not package
    message(STATUS "Continue to publish installer package: cd ${MAIN_WORKING_DIRECTORY}.")
    message(STATUS "Then please execute: binarycreator -c ${CONFIG_FILE_PATH} -p packages ${OUTPUT_CLOUDVIEWER_PACKAGE_PATH}")
endif()