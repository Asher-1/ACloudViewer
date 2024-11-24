if (WIN32)
    SET(PACK_SCRIPTS pack_windows.bat)
elseif (UNIX AND NOT APPLE)
    SET(PACK_SCRIPTS pack_ubuntu.sh)
elseif(APPLE)
    SET(PACK_SCRIPTS pack_macos.sh)
endif ()

if (UNIX AND NOT APPLE)
    ## update version and build time
    execute_process(COMMAND sed -i "s/3.9.0/${CLOUDVIEWER_VERSION}/g" 
    config/config.xml packages/com.vendor.product/meta/package.xml packages/com.vendor.product/meta/installscript.qs
    WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/linux/${MAIN_APP_NAME})
    execute_process(COMMAND sed -i "s/2024-09-18/${BUILD_TIME}/g" packages/com.vendor.product/meta/package.xml
    WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/linux/${MAIN_APP_NAME})
    # install ACloudViewer and colmap app
    execute_process(COMMAND mv ${CMAKE_INSTALL_PREFIX}/${MAIN_APP_NAME} ${CMAKE_INSTALL_PREFIX}/${COLMAP_APP_NAME} 
                    ${DEPLOY_ROOT_PATH} WORKING_DIRECTORY ${DEPLOY_ROOT_PATH})
    # install extra libs, plugins and translations
    execute_process(COMMAND mv ${CMAKE_INSTALL_PREFIX}/${CloudViewer_INSTALL_LIB_DIR}
                    ${CMAKE_INSTALL_PREFIX}/plugins ${CMAKE_INSTALL_PREFIX}/translations
                    ${DEPLOY_ROOT_PATH} WORKING_DIRECTORY ${DEPLOY_ROOT_PATH})
    # install c++ library dependency
    execute_process(COMMAND bash ${DEPLOY_ROOT_PATH}/${PACK_SCRIPTS}
                    "${BUILD_LIB_PATH}" ${CLOUDVIEWER_INSTALL_LIB_DESTINATION}
                    WORKING_DIRECTORY ${BUILD_LIB_PATH})
    execute_process(COMMAND bash ${DEPLOY_ROOT_PATH}/${PACK_SCRIPTS}
                    "${BUILD_LIB_PATH}/plugins" ${CLOUDVIEWER_INSTALL_LIB_DESTINATION}
                    WORKING_DIRECTORY ${BUILD_LIB_PATH})
    execute_process(COMMAND bash ${DEPLOY_ROOT_PATH}/${PACK_SCRIPTS}
                    ${DEPLOY_ROOT_PATH}/platforms/libqxcb.so ${CLOUDVIEWER_INSTALL_LIB_DESTINATION}
                    WORKING_DIRECTORY ${DEPLOY_ROOT_PATH})
    # fix gflags issues
    if (${BUILD_RECONSTRUCTION} STREQUAL "ON")
        execute_process(COMMAND cp "${EXTERNAL_INSTALL_DIR}/lib/${GFLAGS_SRC_FILENAME}" 
                        ${CLOUDVIEWER_INSTALL_LIB_DESTINATION}/${GFLAGS_DST_FILENAME}
                        WORKING_DIRECTORY ${BUILD_LIB_PATH})
    endif()
endif()
if (${PACKAGE} STREQUAL "ON")
    set(SHELL_CMD "binarycreator -c config/config.xml -p packages ${CMAKE_INSTALL_PREFIX}/${CLOUDVIEWER_PACKAGE_NAME}.run")
    message(STATUS "Package with command: " ${SHELL_CMD})
    execute_process(COMMAND binarycreator -c config/config.xml -p packages ${CMAKE_INSTALL_PREFIX}/${CLOUDVIEWER_PACKAGE_NAME}.run
                    WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/linux/${MAIN_APP_NAME})
    message(STATUS "${MAIN_APP_NAME} Installer Package ${CMAKE_INSTALL_PREFIX}/${CLOUDVIEWER_PACKAGE_NAME}.run created")
    # execute_process(COMMAND zip -r ${CMAKE_INSTALL_PREFIX}/${CLOUDVIEWER_PACKAGE_NAME}.zip ${DEPLOY_ROOT_PATH}
    #                 WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX})
    # message(STATUS "Package ${CMAKE_INSTALL_PREFIX}/${CLOUDVIEWER_PACKAGE_NAME}.zip created")
endif()