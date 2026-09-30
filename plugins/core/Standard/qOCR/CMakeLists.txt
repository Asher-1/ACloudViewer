cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QOCR "Install qOCR (PP-OCR text detection/recognition via AICore) plugin" OFF )

if( PLUGIN_STANDARD_QOCR )
    project( QOCR_PLUGIN )

    # AICore owns the ocr catalog; skip the plugin entirely if unavailable
    if(NOT TARGET AICore)
        message(WARNING "qOCR: AICore target not available — skipping plugin. "
                        "Enable AICore_ENABLED and re-run CMake.")
        return()
    endif()

    AddPlugin( NAME ${PROJECT_NAME} )

    add_subdirectory( include )
    add_subdirectory( src )

    target_include_directories( ${PROJECT_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Catalog enumeration and the inference runtime both live in libAICore;
    # the worker/dialog only need the public aicore/* headers.
    target_link_libraries(${PROJECT_NAME} AICore CVPluginAPI
        Qt${QT_VERSION_MAJOR}::Widgets)

    find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets QUIET)
    if(Qt${QT_VERSION_MAJOR}Widgets_FOUND)
        # The workspace dialog is a Q_OBJECT class; make sure its moc is
        # generated for the standalone test target regardless of the
        # global automoc scope.
        set(CMAKE_AUTOMOC ON)
        add_executable(test_qocr tests/test_qocr.cpp
            src/OCRDialog.cpp src/OCRWorker.cpp include/OCRDialog.h
            include/OCRWorker.h)
        target_include_directories(test_qocr PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_link_libraries(test_qocr PRIVATE
            Qt${QT_VERSION_MAJOR}::Widgets AICore CVPluginAPI)
        set_target_properties(test_qocr PROPERTIES
            POSITION_INDEPENDENT_CODE ON AUTOMOC ON)
        if(UNIX AND NOT APPLE)
            target_compile_options(test_qocr PRIVATE -fPIC)
            target_link_options(test_qocr PRIVATE "-no-pie")
        endif()
        cloudviewer_set_bin_runtime_output_directory(test_qocr)
        add_test(NAME test_qocr COMMAND test_qocr)
    endif()
endif()
