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
    target_sources(${PROJECT_NAME} PRIVATE
        include/qAnyLabeling.h
        include/AnnotationWindow.h
        include/AnnotationCanvas.h
        include/AnnotationPopup.h
        include/AnnotationLabelDialog.h
        include/AnnotationNavigator.h
        include/AnnotationWorker.h
        include/SlicedDetection.h
        include/AnnotationSemantic.h
        include/MaskPolygon.h
        include/AnnotationExporter.h
        include/AnnotationImporter.h
        include/AnnotationMot.h
        include/AnnotationOdvg.h
        include/AnnotationVlm.h
        include/AnnotationPpocr.h
        include/AnnotationMask.h
        include/VideoSidecar.h
        include/VideoSegmentExporter.h
        include/VideoDatasetExporter.h
        include/VideoFrameExtraction.h
        include/VideoMaskImport.h
        include/VideoAnnotationDialog.h
        include/PointCloudDocument.h
        include/PointCloudWindow.h
        include/LomaBridgeDialog.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/include/SAM3Types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/include/VideoCanvas.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/include/VideoTimeline.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/include/VideoWorker.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/include/VideoTab.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/include/VideoMaskArchive.h
        src/qAnyLabeling.cpp
        src/AnnotationWindow.cpp
        src/AnnotationCanvas.cpp
        src/AnnotationPopup.cpp
        src/AnnotationLabelDialog.cpp
        src/AnnotationNavigator.cpp
        src/AnnotationWorker.cpp
        src/SlicedDetection.cpp
        src/AnnotationSemantic.cpp
        src/MaskPolygon.cpp
        src/AnnotationExporter.cpp
        src/AnnotationPose.cpp
        src/AnnotationYaml.cpp
        src/AnnotationImporter.cpp
        src/AnnotationMot.cpp
        src/AnnotationOdvg.cpp
        src/AnnotationVlm.cpp
        src/AnnotationPpocr.cpp
        src/AnnotationMask.cpp
        src/VideoSidecar.cpp
        src/VideoSegmentExporter.cpp
        src/VideoDatasetExporter.cpp
        src/VideoFrameExtraction.cpp
        src/VideoMaskImport.cpp
        src/VideoAnnotationDialog.cpp
        src/PointCloudDocument.cpp
        src/PointCloudWindow.cpp
        src/LomaBridgeDialog.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/src/VideoCanvas.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/src/VideoTimeline.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/src/VideoWorker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/src/VideoTab.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/src/VideoMaskArchive.cpp
    )
    if(TRANSLATION_RC_FILES)
        target_sources(${PROJECT_NAME} PRIVATE ${TRANSLATION_RC_FILES})
    endif()
    target_include_directories(${PROJECT_NAME} PRIVATE
        include
        ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/include)
    target_link_libraries(${PROJECT_NAME} AICore video_base Qt::OpenGL)
    cloudviewer_plugin_link_zlib_minizip(${PROJECT_NAME} PLUGIN_NAME ${PROJECT_NAME})
    if(TARGET 3rdparty_opencv)
        target_link_libraries(${PROJECT_NAME} 3rdparty_opencv)
        target_compile_definitions(${PROJECT_NAME} PRIVATE QANYLABELING_HAS_OPENCV_MASK)
    endif()
    if(QT_VERSION_MAJOR GREATER_EQUAL 6)
        target_link_libraries(${PROJECT_NAME} Qt::OpenGLWidgets)
    endif()

    if(BUILD_UNIT_TESTS)
        if(TARGET 3rdparty_opencv)
            add_executable(test_qanylabeling_video_ui
                tests/test_video_ui.cpp src/VideoAnnotationDialog.cpp
                src/VideoSidecar.cpp src/VideoSegmentExporter.cpp
                src/VideoDatasetExporter.cpp
                include/VideoAnnotationDialog.h qAnyLabeling.qrc)
            target_include_directories(test_qanylabeling_video_ui PRIVATE include)
            target_link_libraries(test_qanylabeling_video_ui video_base Qt${QT_VERSION_MAJOR}::Widgets)
            cloudviewer_plugin_link_zlib_minizip(test_qanylabeling_video_ui PLUGIN_NAME test_qanylabeling_video_ui)
            set_target_properties(test_qanylabeling_video_ui PROPERTIES AUTOMOC ON AUTORCC ON
                POSITION_INDEPENDENT_CODE ON)
            if(UNIX AND NOT APPLE)
                target_compile_options(test_qanylabeling_video_ui PRIVATE -fPIC)
                target_link_options(test_qanylabeling_video_ui PRIVATE "-no-pie")
            endif()
            cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_video_ui)
            add_test(NAME test_qanylabeling_video_ui COMMAND test_qanylabeling_video_ui)
            set_tests_properties(test_qanylabeling_video_ui PROPERTIES
                ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
        endif()

        if(TARGET 3rdparty_opencv AND TARGET video_base)
            add_executable(test_qanylabeling_sam3_frames
                tests/test_video_frame_extraction.cpp
                src/VideoFrameExtraction.cpp src/VideoMaskImport.cpp
                src/MaskPolygon.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/src/VideoMaskArchive.cpp)
            target_include_directories(test_qanylabeling_sam3_frames PRIVATE
                include ${CMAKE_CURRENT_SOURCE_DIR}/../qSAM3/include
                ${CMAKE_SOURCE_DIR}/core/AICore/include)
            target_link_libraries(test_qanylabeling_sam3_frames PRIVATE
                video_base Qt${QT_VERSION_MAJOR}::Widgets)
            set_target_properties(test_qanylabeling_sam3_frames PROPERTIES
                POSITION_INDEPENDENT_CODE ON)
            if(UNIX AND NOT APPLE)
                target_compile_options(test_qanylabeling_sam3_frames PRIVATE -fPIC)
                target_link_options(test_qanylabeling_sam3_frames PRIVATE "-no-pie")
            endif()
            cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_sam3_frames)
            add_test(NAME test_qanylabeling_sam3_frames
                COMMAND test_qanylabeling_sam3_frames)

            add_executable(test_qanylabeling_video_dataset
                tests/test_video_dataset_exporter.cpp
                src/VideoDatasetExporter.cpp src/VideoSegmentExporter.cpp
                src/VideoSidecar.cpp)
            target_include_directories(test_qanylabeling_video_dataset PRIVATE include)
            target_link_libraries(test_qanylabeling_video_dataset
                video_base Qt${QT_VERSION_MAJOR}::Core)
            cloudviewer_plugin_link_zlib_minizip(test_qanylabeling_video_dataset
                PLUGIN_NAME test_qanylabeling_video_dataset)
            set_target_properties(test_qanylabeling_video_dataset PROPERTIES
                POSITION_INDEPENDENT_CODE ON)
            if(UNIX AND NOT APPLE)
                target_compile_options(test_qanylabeling_video_dataset PRIVATE -fPIC)
                target_link_options(test_qanylabeling_video_dataset PRIVATE "-no-pie")
            endif()
            cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_video_dataset)
            add_test(NAME test_qanylabeling_video_dataset
                COMMAND test_qanylabeling_video_dataset)
        endif()

        add_executable(test_qanylabeling_video_sidecar
            tests/test_video_sidecar.cpp src/VideoSidecar.cpp)
        target_include_directories(test_qanylabeling_video_sidecar PRIVATE include)
        target_link_libraries(test_qanylabeling_video_sidecar PRIVATE Qt${QT_VERSION_MAJOR}::Core)
        set_target_properties(test_qanylabeling_video_sidecar PROPERTIES POSITION_INDEPENDENT_CODE ON)
        if(UNIX AND NOT APPLE)
            target_compile_options(test_qanylabeling_video_sidecar PRIVATE -fPIC)
            target_link_options(test_qanylabeling_video_sidecar PRIVATE "-no-pie")
        endif()
        cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_video_sidecar)
        add_test(NAME test_qanylabeling_video_sidecar COMMAND test_qanylabeling_video_sidecar)

        add_executable(test_qanylabeling_loma_bridge
            tests/test_loma_bridge.cpp src/LomaBridgeDialog.cpp
            include/LomaBridgeDialog.h)
        target_include_directories(test_qanylabeling_loma_bridge PRIVATE
            include ${CMAKE_SOURCE_DIR}/core/AICore/include)
        target_link_libraries(test_qanylabeling_loma_bridge PRIVATE
            AICore CVPluginAPI Qt${QT_VERSION_MAJOR}::Widgets)
        set_target_properties(test_qanylabeling_loma_bridge PROPERTIES
            AUTOMOC ON POSITION_INDEPENDENT_CODE ON)
        if(UNIX AND NOT APPLE)
            target_compile_options(test_qanylabeling_loma_bridge PRIVATE -fPIC)
            target_link_options(test_qanylabeling_loma_bridge PRIVATE "-no-pie")
        endif()
        cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_loma_bridge)
        add_test(NAME test_qanylabeling_loma_bridge
            COMMAND test_qanylabeling_loma_bridge)
        set_tests_properties(test_qanylabeling_loma_bridge PROPERTIES
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
            SKIP_RETURN_CODE 77)

        add_executable(test_qanylabeling_exporter
            tests/test_exporter.cpp src/AnnotationExporter.cpp src/AnnotationPose.cpp
            src/AnnotationYaml.cpp
            src/AnnotationImporter.cpp src/AnnotationMask.cpp src/MaskPolygon.cpp
            src/AnnotationMot.cpp src/AnnotationOdvg.cpp src/AnnotationVlm.cpp
            src/AnnotationPpocr.cpp)
        target_include_directories(test_qanylabeling_exporter PRIVATE include)
        target_link_libraries(test_qanylabeling_exporter PRIVATE Qt${QT_VERSION_MAJOR}::Gui)
        if(TARGET 3rdparty_opencv)
            target_link_libraries(test_qanylabeling_exporter PRIVATE 3rdparty_opencv)
            target_compile_definitions(test_qanylabeling_exporter PRIVATE
                QANYLABELING_HAS_OPENCV_MASK)
        endif()
        set_target_properties(test_qanylabeling_exporter PROPERTIES POSITION_INDEPENDENT_CODE ON)
        if(UNIX AND NOT APPLE)
            target_compile_options(test_qanylabeling_exporter PRIVATE -fPIC)
            target_link_options(test_qanylabeling_exporter PRIVATE "-no-pie")
        endif()
        cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_exporter)
        add_test(NAME test_qanylabeling_exporter COMMAND test_qanylabeling_exporter)

        add_executable(test_qanylabeling_pointcloud
            tests/test_pointcloud.cpp src/PointCloudDocument.cpp)
        target_include_directories(test_qanylabeling_pointcloud PRIVATE include)
        target_link_libraries(test_qanylabeling_pointcloud PRIVATE Qt${QT_VERSION_MAJOR}::Gui)
        set_target_properties(test_qanylabeling_pointcloud PROPERTIES POSITION_INDEPENDENT_CODE ON)
        if(UNIX AND NOT APPLE)
            target_compile_options(test_qanylabeling_pointcloud PRIVATE -fPIC)
            target_link_options(test_qanylabeling_pointcloud PRIVATE "-no-pie")
        endif()
        cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_pointcloud)
        add_test(NAME test_qanylabeling_pointcloud COMMAND test_qanylabeling_pointcloud)

        find_program(QANYLABELING_XVFB_RUN xvfb-run)
        if(QANYLABELING_XVFB_RUN)
            add_executable(test_qanylabeling_pointcloud_ui
                tests/test_pointcloud_ui.cpp src/PointCloudDocument.cpp
                src/PointCloudWindow.cpp)
            target_include_directories(test_qanylabeling_pointcloud_ui PRIVATE include)
            target_link_libraries(test_qanylabeling_pointcloud_ui PRIVATE
                Qt${QT_VERSION_MAJOR}::Widgets Qt::OpenGL)
            if(QT_VERSION_MAJOR GREATER_EQUAL 6)
                target_link_libraries(test_qanylabeling_pointcloud_ui PRIVATE Qt::OpenGLWidgets)
            endif()
            set_target_properties(test_qanylabeling_pointcloud_ui PROPERTIES
                AUTOMOC ON POSITION_INDEPENDENT_CODE ON)
            if(UNIX AND NOT APPLE)
                target_compile_options(test_qanylabeling_pointcloud_ui PRIVATE -fPIC)
                target_link_options(test_qanylabeling_pointcloud_ui PRIVATE "-no-pie")
            endif()
            cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_pointcloud_ui)
            add_test(NAME test_qanylabeling_pointcloud_ui COMMAND
                ${QANYLABELING_XVFB_RUN} -a $<TARGET_FILE:test_qanylabeling_pointcloud_ui>)
        endif()

        add_executable(test_qanylabeling_canvas
            tests/test_canvas.cpp
            src/AnnotationCanvas.cpp
            src/MaskPolygon.cpp
            include/AnnotationCanvas.h
        )
        target_include_directories(test_qanylabeling_canvas PRIVATE include)
        target_link_libraries(test_qanylabeling_canvas PRIVATE
            Qt${QT_VERSION_MAJOR}::Widgets)
        set_target_properties(test_qanylabeling_canvas PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
        if(UNIX AND NOT APPLE)
            target_compile_options(test_qanylabeling_canvas PRIVATE -fPIC)
            target_link_options(test_qanylabeling_canvas PRIVATE "-no-pie")
        endif()
        cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_canvas)
        add_test(NAME test_qanylabeling_canvas COMMAND test_qanylabeling_canvas)

        add_executable(test_qanylabeling_sliced_detection
            tests/test_sliced_detection.cpp src/SlicedDetection.cpp)
        target_include_directories(test_qanylabeling_sliced_detection PRIVATE include)
        target_link_libraries(test_qanylabeling_sliced_detection PRIVATE Qt${QT_VERSION_MAJOR}::Core)
        set_target_properties(test_qanylabeling_sliced_detection PROPERTIES POSITION_INDEPENDENT_CODE ON)
        if(UNIX AND NOT APPLE)
            target_compile_options(test_qanylabeling_sliced_detection PRIVATE -fPIC)
            target_link_options(test_qanylabeling_sliced_detection PRIVATE "-no-pie")
        endif()
        cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_sliced_detection)
        add_test(NAME test_qanylabeling_sliced_detection COMMAND test_qanylabeling_sliced_detection)
        set_tests_properties(test_qanylabeling_canvas PROPERTIES
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

        add_executable(test_qanylabeling_inference
            tests/test_inference.cpp
            src/AnnotationWorker.cpp
            src/AnnotationSemantic.cpp
            src/SlicedDetection.cpp
            src/MaskPolygon.cpp
            src/AnnotationCanvas.cpp
            include/AnnotationWorker.h
            include/AnnotationCanvas.h
        )
        target_include_directories(test_qanylabeling_inference PRIVATE include)
        target_link_libraries(test_qanylabeling_inference PRIVATE
            AICore Qt${QT_VERSION_MAJOR}::Widgets)
        set_target_properties(test_qanylabeling_inference PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
        if(UNIX AND NOT APPLE)
            target_compile_options(test_qanylabeling_inference PRIVATE -fPIC)
            target_link_options(test_qanylabeling_inference PRIVATE "-no-pie")
        endif()
        cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_inference)
        # This probe requires explicit MODEL and IMAGE arguments and is kept
        # as a manual real-checkpoint smoke test rather than a CTest row.

        add_executable(test_qanylabeling_semantic
            tests/test_semantic.cpp src/AnnotationSemantic.cpp src/MaskPolygon.cpp)
        target_include_directories(test_qanylabeling_semantic PRIVATE include)
        target_link_libraries(test_qanylabeling_semantic PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
        set_target_properties(test_qanylabeling_semantic PROPERTIES POSITION_INDEPENDENT_CODE ON)
        if(UNIX AND NOT APPLE)
            target_compile_options(test_qanylabeling_semantic PRIVATE -fPIC)
            target_link_options(test_qanylabeling_semantic PRIVATE "-no-pie")
        endif()
        cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_semantic)
        add_test(NAME test_qanylabeling_semantic COMMAND test_qanylabeling_semantic)

        add_executable(test_qanylabeling_mask_polygon
            tests/test_mask_polygon.cpp src/MaskPolygon.cpp)
        target_include_directories(test_qanylabeling_mask_polygon PRIVATE include)
        target_link_libraries(test_qanylabeling_mask_polygon PRIVATE Qt${QT_VERSION_MAJOR}::Core)
        set_target_properties(test_qanylabeling_mask_polygon PROPERTIES POSITION_INDEPENDENT_CODE ON)
        if(UNIX AND NOT APPLE)
            target_compile_options(test_qanylabeling_mask_polygon PRIVATE -fPIC)
            target_link_options(test_qanylabeling_mask_polygon PRIVATE "-no-pie")
        endif()
        cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_mask_polygon)
        add_test(NAME test_qanylabeling_mask_polygon COMMAND test_qanylabeling_mask_polygon)

        add_executable(test_qanylabeling_plugin tests/test_plugin.cpp)
        target_link_libraries(test_qanylabeling_plugin PRIVATE
            CVPluginAPI CVPluginStub Qt${QT_VERSION_MAJOR}::Widgets)
        set_target_properties(test_qanylabeling_plugin PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
        if(UNIX AND NOT APPLE)
            target_compile_options(test_qanylabeling_plugin PRIVATE -fPIC)
            target_link_options(test_qanylabeling_plugin PRIVATE "-no-pie")
        endif()
        cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_plugin)
        add_test(NAME test_qanylabeling_plugin
            COMMAND test_qanylabeling_plugin $<TARGET_FILE:QANYLABELING_PLUGIN>)
        set_tests_properties(test_qanylabeling_plugin PROPERTIES
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

        add_executable(test_qanylabeling_sam3_archive
            tests/test_video_mask_import.cpp
            src/VideoMaskImport.cpp src/MaskPolygon.cpp)
        target_include_directories(test_qanylabeling_sam3_archive PRIVATE include)
        target_link_libraries(test_qanylabeling_sam3_archive PRIVATE
            Qt${QT_VERSION_MAJOR}::Gui)
        set_target_properties(test_qanylabeling_sam3_archive PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
        if(UNIX AND NOT APPLE)
            target_compile_options(test_qanylabeling_sam3_archive PRIVATE -fPIC)
            target_link_options(test_qanylabeling_sam3_archive PRIVATE "-no-pie")
        endif()
        cloudviewer_set_bin_runtime_output_directory(test_qanylabeling_sam3_archive)
        add_test(NAME test_qanylabeling_sam3_archive
                 COMMAND test_qanylabeling_sam3_archive)
    endif()
endif()
