
target_sources( ${PROJECT_NAME}
        PRIVATE
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCPlugin.cpp
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCCommon.h
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCCommon.cpp
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCChildProcess.h
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCChildProcess.cpp
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCMethodsScene.cpp
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCMethodsUi.cpp
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCMethodsView.cpp
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCMethodsCloud.cpp
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCMethodsCloudScalarFields.cpp
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCMethodsCloudNormals.cpp
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCMethodsMesh.cpp
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCMethodsProcess.cpp
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCMethodsAI.cpp
                ${CMAKE_CURRENT_LIST_DIR}/JsonRPCMethodsSpirula.cpp
                ${CMAKE_CURRENT_LIST_DIR}/jsonrpcserver.cpp
)
