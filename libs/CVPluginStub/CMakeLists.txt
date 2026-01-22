cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

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
		CV_DB_LIB
)
