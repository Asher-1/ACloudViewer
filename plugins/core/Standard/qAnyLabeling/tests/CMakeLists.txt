# qAnyLabeling unit tests. Registered from the plugin's main CMakeLists
# under if(BUILD_UNIT_TESTS). Every target uses the repository-wide
# cloudviewer_add_plugin_test boilerplate plus
# cloudviewer_set_plugin_test_env_offscreen, so the per-target blocks only
# carry what actually differs: sources, include dirs, libraries and the
# ctest row itself.

set(QAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

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
            ${QAL_DIR}/include/VideoAnnotationDialog.h
            ${QAL_DIR}/qAnyLabeling.qrc
        INCLUDES ${QAL_DIR}/include)
    target_link_libraries(test_qanylabeling_video_ui PRIVATE
        video_base 3rdparty_zlib Qt${QT_VERSION_MAJOR}::Widgets)
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
            ${QAL_DIR}/src/VideoSegmentExporter.cpp
            ${QAL_DIR}/src/VideoSidecar.cpp
        INCLUDES ${QAL_DIR}/include)
    target_link_libraries(test_qanylabeling_video_dataset PRIVATE
        video_base 3rdparty_zlib Qt${QT_VERSION_MAJOR}::Core)
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

# LomaBridge needs its live service; the test itself exits 77 when the
# bridge data is absent, which ctest reports as Skipped.
cloudviewer_add_plugin_test(test_qanylabeling_loma_bridge
    SOURCES test_loma_bridge.cpp
        ${QAL_DIR}/src/LomaBridgeDialog.cpp
        ${QAL_DIR}/include/LomaBridgeDialog.h
    INCLUDES ${QAL_DIR}/include ${CMAKE_SOURCE_DIR}/core/AICore/include
    LIBS AICore CVPluginAPI Qt${QT_VERSION_MAJOR}::Widgets)
add_test(NAME test_qanylabeling_loma_bridge
    COMMAND test_qanylabeling_loma_bridge)
cloudviewer_set_plugin_test_env_offscreen(test_qanylabeling_loma_bridge)
set_tests_properties(test_qanylabeling_loma_bridge PROPERTIES
    SKIP_RETURN_CODE 77)

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
