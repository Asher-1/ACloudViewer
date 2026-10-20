cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option(PLUGIN_STANDARD_QANYLABELING "Install the X-AnyLabeling-compatible annotation workspace" OFF)

if(PLUGIN_STANDARD_QANYLABELING)
    project(QANYLABELING_PLUGIN)
    if(NOT TARGET AICore)
        message(WARNING "qAnyLabeling requires AICore_ENABLED=ON")
        return()
    endif()

    # Source language-menu parity: .ts catalogs are compiled to .qm and
    # embedded in the plugin resources; the workspace loads the catalog
    # matching the persisted language setting at startup.
    set(QANYLABELING_TRANSLATIONS
        translations/qanylabeling_zh_CN.ts
        translations/qanylabeling_ja_JP.ts
        translations/qanylabeling_ko_KR.ts)
    set(QM_FILES)
    if(Qt${QT_VERSION_MAJOR}LinguistTools_FOUND)
        set(TRANSLATION_ENTRIES)
        if(QT_VERSION_MAJOR EQUAL 6)
            qt6_add_translation(QM_FILES ${QANYLABELING_TRANSLATIONS})
        else()
            qt5_add_translation(QM_FILES ${QANYLABELING_TRANSLATIONS})
        endif()
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
        message(STATUS "Qt LinguistTools not found: qAnyLabeling language "
                       "catalogs will not be built")
    endif()

    AddPlugin(NAME ${PROJECT_NAME})
    add_subdirectory(include)
    add_subdirectory(src)
    if(TRANSLATION_RC_FILES)
        target_sources(${PROJECT_NAME} PRIVATE ${TRANSLATION_RC_FILES})
    endif()
    target_include_directories(${PROJECT_NAME} PRIVATE
        include
        ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../qRelateAnything/include)
    target_link_libraries(${PROJECT_NAME} AICore CVPluginAPI video_base Qt::OpenGL)
    target_link_libraries(${PROJECT_NAME} Qt${QT_VERSION_MAJOR}::Concurrent
        Qt${QT_VERSION_MAJOR}::Network)
    cloudviewer_plugin_link_zlib_minizip(${PROJECT_NAME} PLUGIN_NAME ${PROJECT_NAME})
    if(TARGET 3rdparty_opencv)
        target_link_libraries(${PROJECT_NAME} 3rdparty_opencv)
        target_compile_definitions(${PROJECT_NAME} PRIVATE QANYLABELING_HAS_OPENCV_MASK)
    endif()
    if(QT_VERSION_MAJOR GREATER_EQUAL 6)
        target_link_libraries(${PROJECT_NAME} Qt::OpenGLWidgets)
    endif()

    if(BUILD_UNIT_TESTS)
        add_subdirectory(tests)
    endif()
endif()
