cmake_minimum_required(VERSION 3.10)

project( CVPluginStub )

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
