set_directory_properties( PROPERTIES CLEAN_NO_CUSTOM 1 )

# Qt5/Qt6 compatibility
if(USE_QT6)
		find_package( Qt6 COMPONENTS LinguistTools REQUIRED )
    set( QT_PATH ${_qt6_linguisttools_install_prefix} )
    if( UNIX AND NOT APPLE )
        set( QT_PATH "${QT_PATH}/share/qt6" )
    endif()
    # Create unified Qt::lconvert alias
    if(NOT TARGET Qt::lconvert)
        add_executable(Qt::lconvert IMPORTED)
        get_target_property(lconvert_location Qt6::lconvert IMPORTED_LOCATION)
        if(lconvert_location)
            set_target_properties(Qt::lconvert PROPERTIES IMPORTED_LOCATION ${lconvert_location})
        endif()
    endif()
    set( QT_LCONVERT_EXE Qt::lconvert )
else()
    find_package( Qt5LinguistTools REQUIRED )
    # Note that _qt5_linguisttools_install_prefix is from Qt5LinguistToolsConfig.cmake
    #	and probably a private variable...
    set( QT_PATH ${_qt5_linguisttools_install_prefix} )
    if( UNIX AND NOT APPLE )
        set( QT_PATH "${QT_PATH}/share/qt5" )
    endif()
    # Create unified Qt::lconvert alias
    if(NOT TARGET Qt::lconvert)
        add_executable(Qt::lconvert IMPORTED)
        get_target_property(lconvert_location Qt5::lconvert IMPORTED_LOCATION)
        if(lconvert_location)
            set_target_properties(Qt::lconvert PROPERTIES IMPORTED_LOCATION ${lconvert_location})
        endif()
    endif()
    set( QT_LCONVERT_EXE Qt::lconvert )
endif()

# Now handle our own .ts files
file( GLOB TRANSLATION_FILES ${CMAKE_CURRENT_LIST_DIR}/*.ts )

# Qt5/Qt6 compatibility
if(USE_QT6)
    qt6_add_translation( TRANSLATION_QM ${TRANSLATION_FILES} )
else()
    qt5_add_translation( TRANSLATION_QM ${TRANSLATION_FILES} )
endif()
add_custom_target( translations ALL DEPENDS ${TRANSLATION_QM} )

# For each language we have .ts files for, process Qt's qm files
# (they need to be concatenated into one file)
foreach( TRANSLATION_FILE ${TRANSLATION_FILES} )
	get_filename_component( FILENAME ${TRANSLATION_FILE} NAME )
	
	# Our files are named like this: ACloudViewer_fr.qm
	# We want just the language part - "fr" so we can find the right files from the Qt install
	string( REGEX REPLACE "[^_]+_([^._]+).+" "\\1" LANG_CODE ${FILENAME} )
	
	unset( QT_LANG_FILES )
	unset( TRANSLATION_FILES_QT_EXIST )
	
	# These are the Qt files we want to add if they exist
	set( QT_LANG_FILES 
		${QT_PATH}/translations/qtbase_${LANG_CODE}.qm
		)
	
	# Now check that they exist
	foreach ( QT_LANG_FILE IN ITEMS ${QT_LANG_FILES} )
		if ( EXISTS ${QT_LANG_FILE} )
			list( APPEND TRANSLATION_FILES_QT_EXIST ${QT_LANG_FILE} )
		endif()
	endforeach()

	if ( TRANSLATION_FILES_QT_EXIST )
		# Finally, add a command to generate the qt language file for this language
		set( QT_LANG_FILE qt_${LANG_CODE}.qm )
		set( TRANSLATION_QM ${TRANSLATION_QM} ${CMAKE_CURRENT_BINARY_DIR}/${QT_LANG_FILE} )
		
		add_custom_command(
			OUTPUT
				${QT_LANG_FILE}
			DEPENDS
				${TRANSLATION_FILES_QT_EXIST}
			COMMAND
				"${QT_LCONVERT_EXE}" -o ${QT_LANG_FILE} ${TRANSLATION_FILES_QT_EXIST}
			COMMENT
				"Generating Qt language file for ${LANG_CODE}"
			VERBATIM
			)
		
		add_custom_target( translations_${LANG_CODE} ALL DEPENDS qt_${LANG_CODE}.qm )
	endif()
endforeach()

if (CHANGE_TARGET_GENERATION_PATH_FOR_DEBUGGING)
	if( MSVC )
	    set( TRANSLATION_OUTPUT_DIR "${CMAKE_BINARY_DIR}/app/$<CONFIGURATION>/translations" )
		add_custom_command(TARGET translations POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E make_directory
			${TRANSLATION_OUTPUT_DIR}
			VERBATIM
			)
		add_custom_command(TARGET translations POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${TRANSLATION_QM}
			${TRANSLATION_OUTPUT_DIR}
			DEPENDS ${TRANSLATION_QM}
			COMMENT "Copying ${TRANSLATION_QM}"
			VERBATIM
			)
	else()

	endif()
endif()
		
if ( APPLE )
	install( FILES ${TRANSLATION_QM} DESTINATION ${ACloudViewer_MAC_BASE_DIR}/Contents/translations )
elseif ( UNIX )
  install( FILES ${TRANSLATION_QM} DESTINATION translations )
elseif ( WIN32 )
	if( CMAKE_CONFIGURATION_TYPES )
		install( FILES ${TRANSLATION_QM} CONFIGURATIONS Release DESTINATION ${ACloudViewer_DEST_FOLDER}/translations )
		install( FILES ${TRANSLATION_QM} CONFIGURATIONS Debug DESTINATION ${ACloudViewer_DEST_FOLDER}_debug/translations )
	elseif()
		install( FILES ${TRANSLATION_QM} DESTINATION ${ACloudViewer_DEST_FOLDER}/translations )
	endif()
endif()
