set(SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/qSAM3D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Sam3dDialog.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Sam3dWorker.cpp
)

target_sources(${PROJECT_NAME}
    PRIVATE
        ${SOURCE_FILES}
)
