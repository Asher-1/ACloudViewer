target_sources( ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/qYOLO.cpp
        ${CMAKE_CURRENT_LIST_DIR}/YOLOModelCatalog.cpp
        ${CMAKE_CURRENT_LIST_DIR}/YOLOWorker.cpp
        ${CMAKE_CURRENT_LIST_DIR}/YOLOLiveInferWorker.cpp
        ${CMAKE_CURRENT_LIST_DIR}/YOLOLiveWidget.cpp
        ${CMAKE_CURRENT_LIST_DIR}/YOLODialog.cpp
)
