# qAnyLabeling unit tests. Registered from the plugin's main CMakeLists
# under if(BUILD_UNIT_TESTS). Every target uses the repository-wide
# cloudviewer_add_plugin_test boilerplate plus
# cloudviewer_set_plugin_test_env_offscreen, so the per-target blocks only
# carry what actually differs: sources, include dirs, libraries and the
# ctest row itself.

set(QAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Match the plugin's bundled/system zlib selection. The test helper uses
# keyword linking, while cloudviewer_plugin_link_zlib_minizip uses plain.
if(TARGET 3rdparty_zlib)
    set(QAL_ARCHIVE_ZLIB 3rdparty_zlib)
elseif(TARGET 3rdparty_minizip)
    set(QAL_ARCHIVE_ZLIB 3rdparty_minizip)
else()
    find_package(ZLIB REQUIRED)
    set(QAL_ARCHIVE_ZLIB ZLIB::ZLIB)
endif()

cloudviewer_add_plugin_test(test_qanylabeling_ppocr_document
    SOURCES test_ppocr_document.cpp
        ${QAL_DIR}/src/PpocrDocument.cpp
        ${QAL_DIR}/src/PpocrDocumentExport.cpp
        ${QAL_DIR}/src/AnnotationArchive.cpp
        ${QAL_DIR}/src/PpocrServiceClient.cpp
        ${QAL_DIR}/src/PpocrDialog.cpp
        ${QAL_DIR}/src/PpocrTableEditor.cpp
        ${QAL_DIR}/src/PpocrRichTextEditor.cpp
        ${QAL_DIR}/include/PpocrServiceClient.h
        ${QAL_DIR}/include/PpocrDialog.h
        ${QAL_DIR}/include/PpocrTableEditor.h
        ${QAL_DIR}/include/PpocrRichTextEditor.h
    INCLUDES ${QAL_DIR}/include ${CMAKE_SOURCE_DIR}/core/include
    LIBS ${QAL_ARCHIVE_ZLIB} Qt${QT_VERSION_MAJOR}::Widgets Qt${QT_VERSION_MAJOR}::Network)
add_test(NAME test_qanylabeling_ppocr_document COMMAND test_qanylabeling_ppocr_document)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_ppocr_document)

cloudviewer_add_plugin_test(test_qanylabeling_ppocr_table
    SOURCES test_ppocr_table.cpp
        ${QAL_DIR}/src/PpocrTableEditor.cpp
        ${QAL_DIR}/include/PpocrTableEditor.h
    INCLUDES ${QAL_DIR}/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_ppocr_table COMMAND test_qanylabeling_ppocr_table)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_ppocr_table)

cloudviewer_add_plugin_test(test_qanylabeling_ppocr_rich
    SOURCES test_ppocr_rich.cpp
        ${QAL_DIR}/src/PpocrRichTextEditor.cpp
        ${QAL_DIR}/include/PpocrRichTextEditor.h
    INCLUDES ${QAL_DIR}/include ${CMAKE_SOURCE_DIR}/core/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_ppocr_rich COMMAND test_qanylabeling_ppocr_rich)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_ppocr_rich)

cloudviewer_add_plugin_test(test_qanylabeling_chat
    SOURCES test_chat.cpp ${QAL_DIR}/src/ChatServiceClient.cpp
        ${QAL_DIR}/include/ChatServiceClient.h
        ${QAL_DIR}/src/LocalChatEngine.cpp ${QAL_DIR}/include/LocalChatEngine.h
    INCLUDES ${QAL_DIR}/include
    LIBS AICore CVPluginAPI CVPluginStub Qt${QT_VERSION_MAJOR}::Widgets
         Qt${QT_VERSION_MAJOR}::Network Threads::Threads)
add_test(NAME test_qanylabeling_chat
    COMMAND test_qanylabeling_chat $<TARGET_FILE:QANYLABELING_PLUGIN>)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_chat)

cloudviewer_add_plugin_test(test_qanylabeling_tagging
    SOURCES probe_tagging.cpp ${QAL_DIR}/qAnyLabeling.qrc
    INCLUDES ${QAL_DIR}/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_tagging COMMAND test_qanylabeling_tagging)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_tagging)

cloudviewer_add_plugin_test(test_qanylabeling_config
    SOURCES probe_config.cpp
    INCLUDES ${QAL_DIR}/include
    LIBS AICore CVPluginAPI CVPluginStub Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_config
    COMMAND test_qanylabeling_config $<TARGET_FILE:QANYLABELING_PLUGIN>)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_config)

# Loads the real plugin .so through QPluginLoader and exercises the
# workspace object graph (review-toast open-time-hidden contract included).
cloudviewer_add_plugin_test(test_qanylabeling_plugin
    SOURCES test_plugin.cpp
    LIBS CVPluginAPI CVPluginStub Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_plugin
    COMMAND test_qanylabeling_plugin $<TARGET_FILE:QANYLABELING_PLUGIN>)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_plugin)

# Reflection-based probe that bypasses QPluginLoader metadata validation and
# pins the toast placement end to end (open-time hidden, centered at 100
# client pixels, timed auto-close), optionally with a real image.
cloudviewer_add_plugin_test(test_qanylabeling_popup_probe
    SOURCES probe_popup.cpp
    LIBS CVPluginAPI CVPluginStub Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_popup_probe
    COMMAND test_qanylabeling_popup_probe $<TARGET_FILE:QANYLABELING_PLUGIN>)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_popup_probe)

# Remote inference client against an in-process QTcpServer mock: models +
# predict round trips, video session endpoints, SSE propagation and the
# one-shot tracker reset.
cloudviewer_add_plugin_test(test_qanylabeling_remote
    SOURCES test_remote_server.cpp
        ${QAL_DIR}/src/RemoteServerClient.cpp
        ${QAL_DIR}/include/RemoteServerClient.h
    INCLUDES ${QAL_DIR}/include ${CMAKE_SOURCE_DIR}/core/AICore/include
    LIBS CVPluginAPI Qt${QT_VERSION_MAJOR}::Widgets
         Qt${QT_VERSION_MAJOR}::Network)
add_test(NAME test_qanylabeling_remote COMMAND test_qanylabeling_remote)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_remote)

if(TARGET 3rdparty_opencv)
    # cloudviewer_plugin_link_zlib_minizip links with the plain signature,
    # so this target's libraries stay plain as well (CMake forbids mixing
    # keyword and plain forms on one target).
    cloudviewer_add_plugin_test(test_qanylabeling_video_ui
        SOURCES test_video_ui.cpp
            ${QAL_DIR}/src/VideoAnnotationDialog.cpp
            ${QAL_DIR}/src/VideoSidecar.cpp
            ${QAL_DIR}/src/VideoSegmentExporter.cpp
            ${QAL_DIR}/src/VideoDatasetExporter.cpp
            ${QAL_DIR}/src/AnnotationArchive.cpp
            ${QAL_DIR}/include/VideoAnnotationDialog.h
            ${QAL_DIR}/qAnyLabeling.qrc
        INCLUDES ${QAL_DIR}/include)
    target_link_libraries(test_qanylabeling_video_ui PRIVATE
        video_base ${QAL_ARCHIVE_ZLIB} Qt${QT_VERSION_MAJOR}::Widgets)
    # NO_LINK: the shared helper links with the PLAIN signature (it also
    # serves all-plain targets), while cloudviewer_add_plugin_test already
    # linked this target with the KEYWORD signature — CMake forbids mixing,
    # so keep the helper's existence guard and link zlib explicitly above.
    cloudviewer_plugin_link_zlib_minizip(test_qanylabeling_video_ui
        PLUGIN_NAME test_qanylabeling_video_ui NO_LINK)
    add_test(NAME test_qanylabeling_video_ui
        COMMAND test_qanylabeling_video_ui)
    cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_video_ui)
endif()

if(TARGET 3rdparty_opencv AND TARGET video_base)
    cloudviewer_add_plugin_test(test_qanylabeling_sam3_frames
        SOURCES test_video_frame_extraction.cpp
            ${QAL_DIR}/src/VideoFrameExtraction.cpp
            ${QAL_DIR}/src/VideoMaskImport.cpp
            ${QAL_DIR}/src/MaskPolygon.cpp
            ${QAL_DIR}/../qSAM3/src/VideoMaskArchive.cpp
        INCLUDES ${QAL_DIR}/include ${QAL_DIR}/../qSAM3/include
            ${CMAKE_SOURCE_DIR}/core/AICore/include
        LIBS video_base Qt${QT_VERSION_MAJOR}::Widgets)
    add_test(NAME test_qanylabeling_sam3_frames
        COMMAND test_qanylabeling_sam3_frames)
    cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_sam3_frames)

    cloudviewer_add_plugin_test(test_qanylabeling_video_dataset
        SOURCES test_video_dataset_exporter.cpp
            ${QAL_DIR}/src/VideoDatasetExporter.cpp
            ${QAL_DIR}/src/AnnotationArchive.cpp
            ${QAL_DIR}/src/VideoSegmentExporter.cpp
            ${QAL_DIR}/src/VideoSidecar.cpp
        INCLUDES ${QAL_DIR}/include)
    target_link_libraries(test_qanylabeling_video_dataset PRIVATE
        video_base ${QAL_ARCHIVE_ZLIB} Qt${QT_VERSION_MAJOR}::Core)
    cloudviewer_plugin_link_zlib_minizip(test_qanylabeling_video_dataset
        PLUGIN_NAME test_qanylabeling_video_dataset NO_LINK)
    add_test(NAME test_qanylabeling_video_dataset
        COMMAND test_qanylabeling_video_dataset)
    cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_video_dataset)
endif()

cloudviewer_add_plugin_test(test_qanylabeling_video_sidecar
    SOURCES test_video_sidecar.cpp ${QAL_DIR}/src/VideoSidecar.cpp
    INCLUDES ${QAL_DIR}/include
    LIBS Qt${QT_VERSION_MAJOR}::Core)
add_test(NAME test_qanylabeling_video_sidecar
    COMMAND test_qanylabeling_video_sidecar)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_video_sidecar)

# LoMa uses the shared pinned image pair and catalog models. Missing or
# corrupt fixtures fail; they must not silently remove regression coverage.
cloudviewer_add_plugin_test(test_qanylabeling_loma_bridge
    SOURCES test_loma_bridge.cpp
        ${QAL_DIR}/src/LomaBridgeDialog.cpp
        ${QAL_DIR}/include/LomaBridgeDialog.h
    INCLUDES ${QAL_DIR}/include ${CMAKE_SOURCE_DIR}/core/AICore/include
    LIBS AICore CVPluginAPI Qt${QT_VERSION_MAJOR}::Widgets)
target_compile_definitions(test_qanylabeling_loma_bridge PRIVATE
    QANY_VALIDATION_MANIFEST="${CMAKE_SOURCE_DIR}/core/AICore/scripts/validation_manifest.json")
add_test(NAME test_qanylabeling_loma_bridge
    COMMAND test_qanylabeling_loma_bridge)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_loma_bridge)
set_tests_properties(test_qanylabeling_loma_bridge PROPERTIES
    TIMEOUT 600)

cloudviewer_add_plugin_test(test_qanylabeling_exporter
    SOURCES test_exporter.cpp
        ${QAL_DIR}/src/AnnotationExporter.cpp
        ${QAL_DIR}/src/AnnotationPose.cpp
        ${QAL_DIR}/src/AnnotationYaml.cpp
        ${QAL_DIR}/src/AnnotationImporter.cpp
        ${QAL_DIR}/src/AnnotationMask.cpp
        ${QAL_DIR}/src/MaskPolygon.cpp
        ${QAL_DIR}/src/AnnotationMot.cpp
        ${QAL_DIR}/src/AnnotationOdvg.cpp
        ${QAL_DIR}/src/AnnotationVlm.cpp
        ${QAL_DIR}/src/AnnotationPpocr.cpp
    # core/include: QtCompat.h (Qt 5.12 split behaviors); it pulls in
    # QtWidgets headers, hence Widgets rather than Gui.
    INCLUDES ${QAL_DIR}/include ${CMAKE_SOURCE_DIR}/core/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)
if(TARGET 3rdparty_opencv)
    target_link_libraries(test_qanylabeling_exporter PRIVATE 3rdparty_opencv)
    target_compile_definitions(test_qanylabeling_exporter PRIVATE
        QANYLABELING_HAS_OPENCV_MASK)
endif()
add_test(NAME test_qanylabeling_exporter
    COMMAND test_qanylabeling_exporter)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_exporter)

cloudviewer_add_plugin_test(test_qanylabeling_pointcloud
    SOURCES test_pointcloud.cpp ${QAL_DIR}/src/PointCloudDocument.cpp
        ${QAL_DIR}/src/PointCloudCuboid.cpp
        ${QAL_DIR}/include/PointCloudCuboid.h
    INCLUDES ${QAL_DIR}/include
    LIBS Qt${QT_VERSION_MAJOR}::Gui)
add_test(NAME test_qanylabeling_pointcloud
    COMMAND test_qanylabeling_pointcloud)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_pointcloud)

# Camera-calibration sidecar math kernel (schema validation, pinhole
# projection, opencv5 distortion, near-plane and image-rect clipping).
cloudviewer_add_plugin_test(test_qanylabeling_pointcloud_camera
    SOURCES test_pointcloud_camera.cpp
        ${QAL_DIR}/src/PointCloudCamera.cpp
        ${QAL_DIR}/include/PointCloudCamera.h
        ${QAL_DIR}/src/PointCloudCuboid.cpp
        ${QAL_DIR}/include/PointCloudCuboid.h
    INCLUDES ${QAL_DIR}/include
    LIBS Qt${QT_VERSION_MAJOR}::Gui)
add_test(NAME test_qanylabeling_pointcloud_camera
    COMMAND test_qanylabeling_pointcloud_camera)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_pointcloud_camera)

# GL widget workflow needs a real (virtual) X server: run under xvfb-run
# when available; without it the test simply cannot execute.
find_program(QANYLABELING_XVFB_RUN xvfb-run)
if(QANYLABELING_XVFB_RUN)
    cloudviewer_add_plugin_test(test_qanylabeling_pointcloud_ui
        SOURCES test_pointcloud_ui.cpp
            ${QAL_DIR}/src/PointCloudDocument.cpp
            ${QAL_DIR}/src/PointCloudWindow.cpp
            ${QAL_DIR}/src/PointCloudCuboid.cpp
            ${QAL_DIR}/src/PointCloudCamera.cpp
            ${QAL_DIR}/include/PointCloudCuboid.h
            ${QAL_DIR}/include/PointCloudCamera.h
        INCLUDES ${QAL_DIR}/include
        LIBS Qt${QT_VERSION_MAJOR}::Widgets Qt::OpenGL
            Qt${QT_VERSION_MAJOR}::Concurrent)
    if(QT_VERSION_MAJOR GREATER_EQUAL 6)
        target_link_libraries(test_qanylabeling_pointcloud_ui PRIVATE
            Qt::OpenGLWidgets)
    endif()
    add_test(NAME test_qanylabeling_pointcloud_ui
        COMMAND ${QANYLABELING_XVFB_RUN} -a
            $<TARGET_FILE:test_qanylabeling_pointcloud_ui>)
endif()

cloudviewer_add_plugin_test(test_qanylabeling_canvas
    SOURCES test_canvas.cpp
        ${QAL_DIR}/src/AnnotationCanvas.cpp
        ${QAL_DIR}/src/MaskPolygon.cpp
        ${QAL_DIR}/include/AnnotationCanvas.h
    INCLUDES ${QAL_DIR}/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_canvas COMMAND test_qanylabeling_canvas)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_canvas)

cloudviewer_add_plugin_test(test_qanylabeling_sliced_detection
    SOURCES test_sliced_detection.cpp ${QAL_DIR}/src/SlicedDetection.cpp
    INCLUDES ${QAL_DIR}/include
    LIBS Qt${QT_VERSION_MAJOR}::Core)
add_test(NAME test_qanylabeling_sliced_detection
    COMMAND test_qanylabeling_sliced_detection)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_sliced_detection)

# Real-checkpoint smoke probe: requires explicit MODEL and IMAGE arguments,
# so it is intentionally not a ctest row.
cloudviewer_add_plugin_test(test_qanylabeling_inference
    SOURCES test_inference.cpp
        ${QAL_DIR}/qAnyLabeling.qrc
        ${QAL_DIR}/src/AnnotationWorker.cpp
        ${QAL_DIR}/src/AnnotationSemantic.cpp
        ${QAL_DIR}/src/SlicedDetection.cpp
        ${QAL_DIR}/src/MaskPolygon.cpp
        ${QAL_DIR}/src/AnnotationCanvas.cpp
        ${QAL_DIR}/include/AnnotationWorker.h
        ${QAL_DIR}/include/AnnotationCanvas.h
    INCLUDES ${QAL_DIR}/include ${CMAKE_SOURCE_DIR}/core/include
    LIBS AICore Qt${QT_VERSION_MAJOR}::Widgets)

cloudviewer_add_plugin_test(test_qanylabeling_semantic
    SOURCES test_semantic.cpp
        ${QAL_DIR}/src/AnnotationSemantic.cpp
        ${QAL_DIR}/src/MaskPolygon.cpp
    INCLUDES ${QAL_DIR}/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_semantic
    COMMAND test_qanylabeling_semantic)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_semantic)

cloudviewer_add_plugin_test(test_qanylabeling_mask_polygon
    SOURCES test_mask_polygon.cpp ${QAL_DIR}/src/MaskPolygon.cpp
    INCLUDES ${QAL_DIR}/include
    LIBS Qt${QT_VERSION_MAJOR}::Core)
add_test(NAME test_qanylabeling_mask_polygon
    COMMAND test_qanylabeling_mask_polygon)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_mask_polygon)

cloudviewer_add_plugin_test(test_qanylabeling_sam3_archive
    SOURCES test_video_mask_import.cpp
        ${QAL_DIR}/src/VideoMaskImport.cpp
        ${QAL_DIR}/src/MaskPolygon.cpp
    INCLUDES ${QAL_DIR}/include
    LIBS Qt${QT_VERSION_MAJOR}::Gui)
add_test(NAME test_qanylabeling_sam3_archive
    COMMAND test_qanylabeling_sam3_archive)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_sam3_archive)

target_sources(test_qanylabeling_canvas PRIVATE ${QAL_DIR}/src/AnnotationCuboid.cpp)

cloudviewer_add_plugin_test(probe_qanylabeling_render
    SOURCES probe_render.cpp ${QAL_DIR}/src/AnnotationCanvas.cpp
        ${QAL_DIR}/src/AnnotationCuboid.cpp ${QAL_DIR}/src/MaskPolygon.cpp
        ${QAL_DIR}/include/AnnotationCanvas.h
    INCLUDES ${QAL_DIR}/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)

target_sources(test_qanylabeling_inference PRIVATE ${QAL_DIR}/src/AnnotationCuboid.cpp)

target_include_directories(test_qanylabeling_inference PRIVATE ${QAL_DIR}/../qRelateAnything/include)

if(TARGET test_qanylabeling_pointcloud_ui)
    target_sources(test_qanylabeling_pointcloud_ui PRIVATE
        ${QAL_DIR}/include/PointCloudWindow.h
        ${QAL_DIR}/src/PointCloudCuboidView.cpp
        ${QAL_DIR}/src/PointCloudCameraPanel.cpp
        ${QAL_DIR}/include/PointCloudCameraPanel.h
        ${QAL_DIR}/src/PointCloudDatasetExporter.cpp
        ${QAL_DIR}/src/PointCloudDatasetImporter.cpp)
    cloudviewer_plugin_link_zlib_minizip(test_qanylabeling_pointcloud_ui
        PLUGIN_NAME test_qanylabeling_pointcloud_ui NO_LINK)
    target_link_libraries(test_qanylabeling_pointcloud_ui PRIVATE 3rdparty_zlib)
endif()

cloudviewer_add_plugin_test(test_qanylabeling_shape_batch
    SOURCES test_shape_batch.cpp
        ${QAL_DIR}/src/AnnotationShapeBatch.cpp
    INCLUDES ${QAL_DIR}/include ${QAL_DIR}/../qRelateAnything/include
        ${CMAKE_SOURCE_DIR}/core/AICore/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_shape_batch COMMAND test_qanylabeling_shape_batch)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_shape_batch)

cloudviewer_add_plugin_test(test_qanylabeling_document
    SOURCES test_document.cpp
        ${QAL_DIR}/src/AnnotationDocument.cpp
    INCLUDES ${QAL_DIR}/include ${QAL_DIR}/../qRelateAnything/include
        ${CMAKE_SOURCE_DIR}/core/AICore/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_document COMMAND test_qanylabeling_document)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_document)

cloudviewer_add_plugin_test(test_qanylabeling_pointcloud_cuboid
    SOURCES test_pointcloud_cuboid.cpp
        ${QAL_DIR}/src/PointCloudDocument.cpp
        ${QAL_DIR}/src/PointCloudCuboid.cpp
    INCLUDES ${QAL_DIR}/include ${QAL_DIR}/../qRelateAnything/include
        ${CMAKE_SOURCE_DIR}/core/AICore/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_pointcloud_cuboid COMMAND test_qanylabeling_pointcloud_cuboid)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_pointcloud_cuboid)

cloudviewer_add_plugin_test(test_qanylabeling_pointcloud_dataset
    SOURCES test_pointcloud_dataset.cpp
        ${QAL_DIR}/src/PointCloudDatasetExporter.cpp
        ${QAL_DIR}/src/PointCloudCuboid.cpp
    INCLUDES ${QAL_DIR}/include ${QAL_DIR}/../qRelateAnything/include
        ${CMAKE_SOURCE_DIR}/core/AICore/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_pointcloud_dataset COMMAND test_qanylabeling_pointcloud_dataset)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_pointcloud_dataset)
cloudviewer_plugin_link_zlib_minizip(test_qanylabeling_pointcloud_dataset PLUGIN_NAME test_qanylabeling_pointcloud_dataset NO_LINK)
target_link_libraries(test_qanylabeling_pointcloud_dataset PRIVATE 3rdparty_zlib)

cloudviewer_add_plugin_test(test_qanylabeling_pointcloud_dataset_importer
    SOURCES test_pointcloud_dataset_importer.cpp
        ${QAL_DIR}/src/PointCloudDatasetImporter.cpp
        ${QAL_DIR}/src/PointCloudDatasetExporter.cpp
        ${QAL_DIR}/src/PointCloudCuboid.cpp
    INCLUDES ${QAL_DIR}/include ${QAL_DIR}/../qRelateAnything/include
        ${CMAKE_SOURCE_DIR}/core/AICore/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_pointcloud_dataset_importer COMMAND test_qanylabeling_pointcloud_dataset_importer)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_pointcloud_dataset_importer)
cloudviewer_plugin_link_zlib_minizip(test_qanylabeling_pointcloud_dataset_importer PLUGIN_NAME test_qanylabeling_pointcloud_dataset_importer NO_LINK)
target_link_libraries(test_qanylabeling_pointcloud_dataset_importer PRIVATE 3rdparty_zlib)

cloudviewer_add_plugin_test(test_qanylabeling_relation_filter
    SOURCES test_relation_filter.cpp
        ${QAL_DIR}/src/RelationFilterWidget.cpp
        ${QAL_DIR}/include/RelationFilterWidget.h
    INCLUDES ${QAL_DIR}/include ${QAL_DIR}/../qRelateAnything/include
        ${CMAKE_SOURCE_DIR}/core/AICore/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_relation_filter COMMAND test_qanylabeling_relation_filter)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_relation_filter)

cloudviewer_add_plugin_test(test_qanylabeling_relate_options
    SOURCES test_relate_options.cpp
        ${QAL_DIR}/src/RelateOptionsDialog.cpp
        ${QAL_DIR}/src/RelateDetectors.cpp
        ${QAL_DIR}/include/RelateOptionsDialog.h
    INCLUDES ${QAL_DIR}/include ${QAL_DIR}/../qRelateAnything/include
        ${CMAKE_SOURCE_DIR}/core/AICore/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets AICore CVPluginAPI)
add_test(NAME test_qanylabeling_relate_options COMMAND test_qanylabeling_relate_options)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_relate_options)

cloudviewer_add_plugin_test(test_qanylabeling_cuboid
    SOURCES test_cuboid.cpp
        ${QAL_DIR}/src/AnnotationCanvas.cpp
        ${QAL_DIR}/src/AnnotationCuboid.cpp
        ${QAL_DIR}/src/MaskPolygon.cpp
        ${QAL_DIR}/include/AnnotationCanvas.h
    INCLUDES ${QAL_DIR}/include ${QAL_DIR}/../qRelateAnything/include
        ${CMAKE_SOURCE_DIR}/core/AICore/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_cuboid COMMAND test_qanylabeling_cuboid)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_cuboid)

cloudviewer_add_plugin_test(test_qanylabeling_run_path
    SOURCES test_run_path.cpp
    INCLUDES ${QAL_DIR}/include ${QAL_DIR}/../qRelateAnything/include
        ${CMAKE_SOURCE_DIR}/core/AICore/include
    LIBS Qt${QT_VERSION_MAJOR}::Widgets CVPluginAPI CVPluginStub)
add_test(NAME test_qanylabeling_run_path COMMAND test_qanylabeling_run_path $<TARGET_FILE:QANYLABELING_PLUGIN>)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_run_path)
set_tests_properties(test_qanylabeling_run_path PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200)
add_test(NAME test_qanylabeling_tagged_run_path COMMAND test_qanylabeling_run_path $<TARGET_FILE:QANYLABELING_PLUGIN> --tagged)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_tagged_run_path)
set_tests_properties(test_qanylabeling_tagged_run_path PROPERTIES TIMEOUT 600)
add_test(NAME test_qanylabeling_classifier_run_path COMMAND test_qanylabeling_run_path $<TARGET_FILE:QANYLABELING_PLUGIN> --classifier)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_classifier_run_path)
set_tests_properties(test_qanylabeling_classifier_run_path PROPERTIES TIMEOUT 600)
add_test(NAME test_qanylabeling_ocr_run_path COMMAND test_qanylabeling_run_path $<TARGET_FILE:QANYLABELING_PLUGIN> --ocr)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_ocr_run_path)
set_tests_properties(test_qanylabeling_ocr_run_path PROPERTIES TIMEOUT 600)
# 本地属性模型由测试环境显式注入；缺失时失败，不下载或跳过。
add_test(NAME test_qanylabeling_attribute_run_path
    COMMAND test_qanylabeling_run_path $<TARGET_FILE:QANYLABELING_PLUGIN> --attribute)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_attribute_run_path)
set_tests_properties(test_qanylabeling_attribute_run_path PROPERTIES TIMEOUT 600)

# 真实插件注册的原生命令；外部本地模型与输入缺失必须失败，不自动下载。
cloudviewer_add_plugin_test(test_qanylabeling_attribute_batch
    SOURCES test_attribute_batch.cpp
    LIBS Qt${QT_VERSION_MAJOR}::Widgets CVPluginAPI CVPluginStub)
add_test(NAME test_qanylabeling_attribute_batch
    COMMAND test_qanylabeling_attribute_batch $<TARGET_FILE:QANYLABELING_PLUGIN>)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_attribute_batch)
set_tests_properties(test_qanylabeling_attribute_batch PROPERTIES
    TIMEOUT 600 LABELS "model;e2e;cvpluginapi")

add_test(NAME test_qanylabeling_lightglue_bridge COMMAND test_qanylabeling_loma_bridge)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_lightglue_bridge)
set_property(TEST test_qanylabeling_lightglue_bridge APPEND PROPERTY
    ENVIRONMENT "QANY_LIGHTGLUE_TEST=1")
set_tests_properties(test_qanylabeling_lightglue_bridge PROPERTIES TIMEOUT 600)
