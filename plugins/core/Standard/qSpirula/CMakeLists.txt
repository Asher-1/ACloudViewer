cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QSPIRULA "Install qSpirula (Spirula Studio 3DGS training) plugin" OFF )

if( PLUGIN_STANDARD_QSPIRULA )
    project( QSPIRULA_PLUGIN )

    # The plugin is meaningless without the standalone executable — enable the
    # component automatically instead of silently skipping the plugin when a
    # user passes only -DPLUGIN_STANDARD_QSPIRULA=ON. A plain set() (not CACHE)
    # keeps explicit -DBUILD_SPIRULA=OFF out of the way while spirula.cmake's
    # own option() stays visible to later scopes.
    if(NOT TARGET ext_spirula)
        set(BUILD_SPIRULA ON)
        include(${CloudViewer_3RDPARTY_DIR}/spirula/spirula.cmake)
    endif()
    if(NOT TARGET ext_spirula)
        message(WARNING "qSpirula: ext_spirula target not available — skipping plugin. "
                        "Enable -DBUILD_SPIRULA=ON and re-run CMake.")
        return()
    endif()

    # Translation catalogs (.ts → .qm) embedded as plugin resources; the
    # constructor loads the catalog matching the persisted language setting.
    set(QSPIRULA_TRANSLATIONS
        translations/qspirula_zh_CN.ts)
    set(TRANSLATION_RC_FILES)
    if(Qt${QT_VERSION_MAJOR}LinguistTools_FOUND)
        if(QT_VERSION_MAJOR EQUAL 6)
            qt6_add_translation(QM_FILES ${QSPIRULA_TRANSLATIONS})
        else()
            qt5_add_translation(QM_FILES ${QSPIRULA_TRANSLATIONS})
        endif()
        set(TRANSLATION_ENTRIES)
        foreach(_qm ${QM_FILES})
            get_filename_component(_qm_name "${_qm}" NAME)
            string(APPEND TRANSLATION_ENTRIES
                "        <file alias=\"${_qm_name}\">${_qm}</file>\n")
        endforeach()
        configure_file(translations/translations.qrc.in
                       ${CMAKE_CURRENT_BINARY_DIR}/translations.qrc @ONLY)
        if(QT_VERSION_MAJOR EQUAL 6)
            qt6_add_resources(TRANSLATION_RC_FILES
                ${CMAKE_CURRENT_BINARY_DIR}/translations.qrc)
        else()
            qt5_add_resources(TRANSLATION_RC_FILES
                ${CMAKE_CURRENT_BINARY_DIR}/translations.qrc)
        endif()
    else()
        message(STATUS "Qt LinguistTools not found: qSpirula language "
                       "catalogs will not be built")
    endif()

    AddPlugin( NAME ${PROJECT_NAME} )

    if(TRANSLATION_RC_FILES)
        target_sources(${PROJECT_NAME} PRIVATE ${TRANSLATION_RC_FILES})
    endif()

    add_subdirectory( include )
    add_subdirectory( src )

    target_include_directories( ${PROJECT_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # mesh.ply import lives in spirula_base (shared with qJSonRPCPlugin);
    # linking it brings SpirulaLocator/SpirulaLauncher/SplatImporter headers.
    if(TARGET spirula_base)
        target_link_libraries(${PROJECT_NAME} spirula_base)
    endif()
    target_link_libraries(${PROJECT_NAME} CV_IO_LIB)

    # Build-order guarantee + runtime path fallback for SpirulaLocator.
    add_dependencies(${PROJECT_NAME} ext_spirula)
    get_filename_component(SPIRULA_BIN_DIR "${SPIRULA_EXECUTABLE}" DIRECTORY)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        QSPIRULA_BUILD_PATH="${SPIRULA_BIN_DIR}")

    # Carry the executable next to the application so the packaged layout
    # contains it automatically (plugins sit in bin/plugins; ../ is bin).
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${SPIRULA_EXECUTABLE}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/../spirula${CMAKE_EXECUTABLE_SUFFIX}"
        COMMENT "Copying spirula executable next to the application")

    add_subdirectory( tests )
endif()
