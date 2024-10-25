macro(colmap_add_app PROJECT_ROOT_PATH APP_NAME TARGET_NAME)
    message(STATUS "ACloudViewer project root path: ${PROJECT_ROOT_PATH}")
    set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/ColmapApp")
    set(RESOURCE_DIR_NAME "Contents/Resources")
    set(EXE_DIR_NAME "Contents/MacOS")
    file(GLOB RESOURCE_FILES "${SOURCE_DIR}/*.icns")
    set(EXECUTABLE_SCRIPTS_NAME "colmap_gui.sh")
    set(EXECUTABLE_SCRIPTS "${SOURCE_DIR}/${EXECUTABLE_SCRIPTS_NAME}")
    set(INFO_PLIST "${SOURCE_DIR}/Info.plist.in")

    set(MACOSX_BUNDLE_NAME ${APP_NAME})
    set(MACOSX_BUNDLE_EXECUTABLE_NAME ${APP_NAME})
    set(MACOSX_BUNDLE_GUI_IDENTIFIER com.asher.${APP_NAME})
    set(MACOSX_BUNDLE_ICON_FILE "AppIcon")
    set(MACOSX_BUNDLE_LONG_VERSION_STRING ${PROJECT_VERSION_THREE_NUMBER})
    set(MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_THREE_NUMBER})
    set(MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION_THREE_NUMBER})
    set(MACOSX_BUNDLE_COPYRIGHT "Copyright (C) 2023 by ${APP_NAME}")

    set_target_properties(${TARGET_NAME} PROPERTIES
            MACOSX_BUNDLE TRUE
            INSTALL_RPATH "@executable_path/../Frameworks"
            MACOSX_BUNDLE_NAME ${APP_NAME}
            MACOSX_BUNDLE_GUI_IDENTIFIER com.asher.${APP_NAME}
            MACOSX_BUNDLE_INFO_PLIST "${INFO_PLIST}"
            XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "" # disable
            MACOSX_BUNDLE_ICON_FILE AppIcon.icns
            MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION_THREE_NUMBER}"
            MACOSX_BUNDLE_LONG_VERSION_STRING "${PROJECT_VERSION_THREE_NUMBER}"
            MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION_THREE_NUMBER}"
            OUTPUT_NAME ${APP_NAME})

    # $<TARGET_BUNDLE_DIR> does not exist at config time, so we need to
    # duplicate the post build step on macOS and non-macOS
    set(APP_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    find_program(mac_deploy_qt macdeployqt HINTS "${qt5_bin_dir}")

    if (NOT EXISTS "${mac_deploy_qt}")
        message(FATAL_ERROR "macdeployqt not found in ${qt5_bin_dir}")
    endif ()

    add_custom_command(TARGET "${TARGET_NAME}"
            POST_BUILD
            # copy the resource files into the bundle
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>/${RESOURCE_DIR_NAME}"
            COMMAND ${CMAKE_COMMAND} -E copy ${RESOURCE_FILES} "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>/${RESOURCE_DIR_NAME}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>/Contents/Frameworks"
            COMMAND "${mac_deploy_qt}" "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>" -verbose=1
            VERBATIM
            )

    set(APP_INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/${APP_NAME}")
    install(DIRECTORY "${APP_DIR}/${APP_NAME}.app"
            DESTINATION ${APP_INSTALL_DESTINATION}
            USE_SOURCE_PERMISSIONS)

    # copy external libraries (e.g. SDL into the bundle and fixup the search paths
    set(PACK_SCRIPTS_PATH "${PROJECT_ROOT_PATH}/scripts/platforms/mac/pack_macosx_bundle.sh")
    set(APP_INSTALL_EXE_DESTINATION "${APP_INSTALL_DESTINATION}/${APP_NAME}.app/Contents")
    message(STATUS "APP_INSTALL_EXE_DESTINATION: ${APP_INSTALL_EXE_DESTINATION}")
    message(STATUS "pack_macosx_bundle: ${PACK_SCRIPTS_PATH}")
    install(CODE "execute_process(COMMAND ${PACK_SCRIPTS_PATH} ${APP_INSTALL_EXE_DESTINATION}/MacOS)")
    install(CODE "execute_process(COMMAND ${PACK_SCRIPTS_PATH} ${APP_INSTALL_EXE_DESTINATION}/Frameworks)")
    install(CODE "execute_process(COMMAND ${PACK_SCRIPTS_PATH} ${APP_INSTALL_EXE_DESTINATION}/Frameworks)")
endmacro(colmap_add_app)