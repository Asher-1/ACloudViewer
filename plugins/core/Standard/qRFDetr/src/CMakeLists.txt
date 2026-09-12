target_sources( ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/qRFDetr.cpp
        ${CMAKE_CURRENT_LIST_DIR}/RFDetrModelCatalog.cpp
        ${CMAKE_CURRENT_LIST_DIR}/RFDetrWorker.cpp
        ${CMAKE_CURRENT_LIST_DIR}/RFDetrLiveInferWorker.cpp
        ${CMAKE_CURRENT_LIST_DIR}/RFDetrLiveWidget.cpp
        ${CMAKE_CURRENT_LIST_DIR}/RFDetrDialog.cpp
)
