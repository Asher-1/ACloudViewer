target_sources( ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/qYOLO.cpp
        ${CMAKE_CURRENT_LIST_DIR}/YOLOModelCatalog.cpp
        ${CMAKE_CURRENT_LIST_DIR}/YOLOWorker.cpp
        ${CMAKE_CURRENT_LIST_DIR}/YOLOLiveInferWorker.cpp
        ${CMAKE_CURRENT_LIST_DIR}/YOLOLiveWidget.cpp
        ${CMAKE_CURRENT_LIST_DIR}/YOLOVisualPromptLabel.cpp
        ${CMAKE_CURRENT_LIST_DIR}/YOLODialog.cpp
        # Multi-object tracking (algorithm-for-algorithm port of the
        # ultralytics-ggml runtime's six trackers; see tracking/tracker.hpp).
        ${CMAKE_CURRENT_LIST_DIR}/tracking/tracker.cpp
)
