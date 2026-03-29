project(imgui)

file(GLOB SOURCES "*.cpp" "*.h" "examples/opengl3_example/imgui_impl_glfw_gl3.cpp" "examples/opengl3_example/imgui_impl_glfw_gl3.h")
source_group("Source Files" FILES ${SOURCES})

## Specify target rules
add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_link_libraries(${PROJECT_NAME}
    ${GLEW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    glfw
)

add_definitions(-DIMGUI_INCLUDE_IMCONFIG_H -DIMGUI_EXPORTS)

target_include_directories(${PROJECT_NAME} PUBLIC . "examples/opengl3_example")

