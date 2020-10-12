project( CVPluginStub )

#include_directories( ${CV_CORE_LIB_SOURCE_DIR}/include )
#include_directories( ${ECV_DB_LIB_SOURCE_DIR} )
#if( MSVC )
#   include_directories( ${ECV_DB_LIB_SOURCE_DIR}/msvc )
#endif()
#include_directories( ${ECV_IO_LIB_SOURCE_DIR} )

add_library( ${PROJECT_NAME} STATIC )

set_target_properties( ${PROJECT_NAME} PROPERTIES
    AUTOMOC OFF
    AUTORCC OFF
    AUTOUIC OFF
)

add_subdirectory( include )
add_subdirectory( src )

target_link_libraries( ${PROJECT_NAME}
	PRIVATE
		ECV_DB_LIB
)
