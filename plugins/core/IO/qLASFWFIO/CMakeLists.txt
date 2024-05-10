# Windows only
if (WIN32)
	option( PLUGIN_IO_QLAS_FWF "Check to install qLAS FWF I/O plugin (deprecated, prefer using qLAS I/O now)" OFF )

    if (PLUGIN_IO_QLAS_FWF)
		message( WARNING "The qLAS_FWF plugin is now deprecated, prefer using qLAS instead" )

        #CloudViewer LAS FWF (i.e. version >= 1.3) I/O plugin
        project(QLAS_FWF_IO_PLUGIN)

        AddPlugin(NAME ${PROJECT_NAME} TYPE io)

        # LasLibZip
        include(cmake/LasLibZip_download.cmake)
        import_3rdparty_library(3rdparty_lasLibzip
                INCLUDE_DIRS ${LASLIBZIP_DIR}/
                DEPENDS ext_lasLibzip
                )
        add_dependencies(${PROJECT_NAME} 3rdparty_lasLibzip)
		
		# LASLib include directory
		set(LASLIB_INCLUDE_DIR "${LASLIBZIP_DIR}/LASlib/inc" CACHE PATH "LASLib include directory")
		set(LASZIP_INCLUDE_DIR "${LASLIBZIP_DIR}/LASzip/src" CACHE PATH "LASZip include directory")
		set(LASLIB_RELEASE_LIBRARY "${LASLIBZIP_DIR}/LASlib/lib/Release/LASlib.lib" CACHE FILEPATH "LASLib release library file")
		set(LASLIB_DEBUG_LIBRARY "${LASLIBZIP_DIR}/LASlib/lib/Debug/LASlib.lib" CACHE FILEPATH "LASLib debug library file")
		if (NOT LASLIB_INCLUDE_DIR OR NOT LASZIP_INCLUDE_DIR)
			message(SEND_ERROR "No LASLib or LASZip include directories specified (LASLIB_INCLUDE_DIR / LASZIP_INCLUDE_DIR)")
		else ()
			include_directories(${LASLIB_INCLUDE_DIR})
			include_directories(${LASZIP_INCLUDE_DIR})
		endif ()

        add_subdirectory(include)
        add_subdirectory(src)
        add_subdirectory(ui)

        # Uses headers from the qPDALIO plugin...
        target_include_directories(${PROJECT_NAME}
                PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/../qPDALIO/include"
                )
		
		if (LASLIB_RELEASE_LIBRARY)
			target_link_libraries(${PROJECT_NAME} optimized ${LASLIB_RELEASE_LIBRARY})
		endif ()

		if (CMAKE_CONFIGURATION_TYPES)
			set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS_RELEASE WITH_LASLIB)
			if (LASLIB_DEBUG_LIBRARY)
				target_link_libraries(${PROJECT_NAME} debug ${LASLIB_DEBUG_LIBRARY})
				set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS_DEBUG WITH_LASLIB)
			endif ()
		else ()
			set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS WITH_LASLIB)
		endif ()

    endif ()

endif ()
