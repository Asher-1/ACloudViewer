target_sources( ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/qFaceDetect.h
        ${CMAKE_CURRENT_LIST_DIR}/FaceDetectDialog.h
        ${CMAKE_CURRENT_LIST_DIR}/FaceDetectWorker.h
        ${CMAKE_CURRENT_LIST_DIR}/FaceDetectEmbedHelpers.h
        ${CMAKE_CURRENT_LIST_DIR}/FaceDetectModelContext.h
        ${CMAKE_CURRENT_LIST_DIR}/FaceLiveDetectInferWorker.h
        ${CMAKE_CURRENT_LIST_DIR}/FaceLiveDetectWidget.h
        ${CMAKE_CURRENT_LIST_DIR}/FaceRegistryStore.h
        ${CMAKE_CURRENT_LIST_DIR}/FaceRegistryWidget.h
        ${CMAKE_CURRENT_LIST_DIR}/FaceDetectTestData.h
        ${CMAKE_CURRENT_LIST_DIR}/FaceDetectTestDataWorker.h
)
