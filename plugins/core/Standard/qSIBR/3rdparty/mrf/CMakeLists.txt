project(mrf)

file(GLOB SOURCES "*.cpp" "*.h" "*.inc")
source_group("Source Files" FILES ${SOURCES})

## Specify target rules
add_library(${PROJECT_NAME} SHARED ${SOURCES})

add_definitions(-DMRF_EXPORTS)
