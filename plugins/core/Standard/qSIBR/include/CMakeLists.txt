target_sources( ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/qSIBR.h
        ${CMAKE_CURRENT_LIST_DIR}/SIBRViewerThread.h
        ${CMAKE_CURRENT_LIST_DIR}/SIBROptionsDialog.h
)

target_include_directories( ${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
