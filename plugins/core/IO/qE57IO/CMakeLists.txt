
option(PLUGIN_IO_QE57 "Install qE57_IO plugin to read & write E57 files using libE57Format" OFF)

if (PLUGIN_IO_QE57)
    project(QE57_IO_PLUGIN)

    AddPlugin(NAME ${PROJECT_NAME} TYPE io)

    add_subdirectory(include)
    add_subdirectory(src)
	
	if (WIN32)
		message(STATUS "Using prebuilt third-party library libE57Format and xercesC")
        include( cmake/E57Format_XERCES_download.cmake )
		include( cmake/E57Support.cmake )
		copy_shared_library(${PROJECT_NAME}
			LIB_DIR      ${XERCES_ROOT}/bin
			LIBRARIES    xerces-c_3_2$<$<CONFIG:Debug>:D>)
		target_link_LIBE57FORMAT( ${PROJECT_NAME} )
	else ()
		include( cmake/E57Format_build.cmake )
		import_3rdparty_library(3rdparty_libE57Format
				INCLUDE_DIRS ${E57Format_INCLUDE_DIRS}
				LIB_DIR ${E57Format_LIB_DIR}
				LIBRARIES ${E57Format_LIBRARIES}
				DEPENDS ext_libE57Format ext_xerces
				)
		add_dependencies(${PROJECT_NAME} 3rdparty_libE57Format)
		set_target_properties(3rdparty_libE57Format PROPERTIES
				AUTOMOC OFF
				AUTORCC OFF
				AUTOUIC OFF
				)
		target_link_libraries(${PROJECT_NAME} 3rdparty_libE57Format)

		if (USING_STATIC_XERCES)
			target_compile_definitions(${PROJECT_NAME} PRIVATE XERCES_STATIC_LIBRARY)
		endif ()
	endif()


endif ()
