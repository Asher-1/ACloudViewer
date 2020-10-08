file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")

include_directories(".")

include_directories( ${OPENGL_ENGINE_LIB_SOURCE_DIR} )
if( MSVC )
   include_directories( ${OPENGL_ENGINE_LIB_SOURCE_DIR}/msvc )
endif()

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

add_executable(ManuallyAlignPointCloud ${HEADER_FILES} ${SOURCE_FILES})
ShowAndAbortOnWarning(ManuallyAlignPointCloud)

target_link_libraries( ManuallyAlignPointCloud ECV_DB_LIB )
target_link_libraries( ManuallyAlignPointCloud ECV_IO_LIB )
target_link_libraries( ManuallyAlignPointCloud  OPENGL_ENGINE_LIB )

set_target_properties(ManuallyAlignPointCloud PROPERTIES FOLDER "Tools")
