file( GLOB internal_header_list ${CMAKE_CURRENT_SOURCE_DIR}/*.h )
set(header_list
   ${header_list}
   ${internal_header_list}
   PARENT_SCOPE
)

file( GLOB internal_source_list ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )
set(source_list
   ${source_list}
   ${internal_source_list}
   PARENT_SCOPE
)

set( qrc_list
    ${qrc_list}
    ${CMAKE_CURRENT_SOURCE_DIR}/resources.qrc
    PARENT_SCOPE
)
