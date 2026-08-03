target_sources( ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/qFaceDetect.cpp
        ${CMAKE_CURRENT_LIST_DIR}/FaceDetectDialog.cpp
        ${CMAKE_CURRENT_LIST_DIR}/FaceDetectWorker.cpp
        ${CMAKE_CURRENT_LIST_DIR}/FaceDetectEmbedHelpers.cpp
        ${CMAKE_CURRENT_LIST_DIR}/FaceDetectModelContext.cpp
        ${CMAKE_CURRENT_LIST_DIR}/FaceLiveDetectInferWorker.cpp
        ${CMAKE_CURRENT_LIST_DIR}/FaceLiveDetectWidget.cpp
        ${CMAKE_CURRENT_LIST_DIR}/FaceRegistryStore.cpp
        ${CMAKE_CURRENT_LIST_DIR}/FaceRegistryWidget.cpp
        ${CMAKE_CURRENT_LIST_DIR}/FaceDetectTestData.cpp
        ${CMAKE_CURRENT_LIST_DIR}/FaceDetectTestDataWorker.cpp
)
