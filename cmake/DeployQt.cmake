# Support both Qt5 and Qt6 based on USE_QT6 option
if(USE_QT6)
    find_package( Qt6
        COMPONENTS
            Core
        REQUIRED
    )
    get_target_property( qmake_location Qt6::qmake IMPORTED_LOCATION )
    get_filename_component( qt_bin_dir ${qmake_location} DIRECTORY )
else()
    find_package( Qt5
        COMPONENTS
            Core
        REQUIRED
    )
    get_target_property( qmake_location Qt5::qmake IMPORTED_LOCATION )
    get_filename_component( qt_bin_dir ${qmake_location} DIRECTORY )
endif()

if ( APPLE )
	find_program( mac_deploy_qt macdeployqt HINTS "${qt_bin_dir}" )
	if( NOT EXISTS "${mac_deploy_qt}" )
		message( FATAL_ERROR "macdeployqt not found in ${qt_bin_dir}" )
	endif()
elseif( WIN32 )
	find_program( win_deploy_qt windeployqt HINTS "${qt_bin_dir}" )
	if( NOT EXISTS "${win_deploy_qt}" )
		message( FATAL_ERROR "windeployqt not found in ${qt_bin_dir}" )
	endif()
endif()

function( DeployQt )
	if ( NOT APPLE AND NOT WIN32 )
		message( FATAL_ERROR "DeployQt only supports Windows and Mac OS X" )
		return()
	endif()
	
	cmake_parse_arguments(
			DEPLOY_QT
			""
			"TARGET;DEPLOY_PATH"
			""
			${ARGN}
	)

	if( NOT DEPLOY_QT_DEPLOY_PATH )
		message( FATAL_ERROR "DeployQt: DEPLOY_PATH not set" )
	endif()

	# For readability
	set( deploy_path "${DEPLOY_QT_DEPLOY_PATH}" )
	
	message( STATUS "Installing ${DEPLOY_QT_TARGET} to ${deploy_path}" )
	
	get_target_property( name ${DEPLOY_QT_TARGET} NAME )
		
	if ( APPLE )
		set( app_name "${name}.app" )
		# fix macdeployqt package issues with rpath and external dynamic libraries
		# copy external libraries (e.g. SDL into the bundle and fixup the search paths
		set(INSTALL_DEPLOY_PATH "${CMAKE_INSTALL_PREFIX}/${deploy_path}")
    set(deploy_app_contents "${INSTALL_DEPLOY_PATH}/${app_name}/Contents")

		if (CMAKE_CONFIGURATION_TYPES)
			set(app_path "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${app_name}")
			set(temp_dir "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/deployqt")
		else ()
			set(app_path "${CMAKE_CURRENT_BINARY_DIR}/${app_name}")
			set(temp_dir "${CMAKE_CURRENT_BINARY_DIR}/deployqt")
		endif ()
		set( temp_app_path "${temp_dir}/$<CONFIG>/${app_name}" )
		add_custom_command(
				TARGET ${DEPLOY_QT_TARGET}
				POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E remove_directory "${temp_dir}"
				COMMAND ${CMAKE_COMMAND} -E make_directory "${temp_dir}"
				COMMAND ${CMAKE_COMMAND} -E copy_directory ${app_path} ${temp_app_path}
				COMMAND "${mac_deploy_qt}" ${temp_app_path} -verbose=1
				VERBATIM
		)

		install(
				DIRECTORY ${temp_app_path}
				DESTINATION ${INSTALL_DEPLOY_PATH}
				USE_SOURCE_PERMISSIONS
		)

		set(PACK_SCRIPTS_PATH "${PROJECT_ROOT_PATH}/scripts/platforms/mac/bundle/lib_bundle_app.py")
		set(APP_SIGN_SCRIPT_PATH "${PROJECT_ROOT_PATH}/scripts/platforms/mac/bundle/signature_app.py")
		if (PLUGIN_PYTHON AND BUILD_PYTHON_MODULE)
			install(CODE "execute_process(COMMAND python ${PACK_SCRIPTS_PATH} ${name} ${INSTALL_DEPLOY_PATH} --embed_python)")
			install(CODE "execute_process(COMMAND python ${APP_SIGN_SCRIPT_PATH} ${name} ${INSTALL_DEPLOY_PATH} --embed_python)")
		else()
			install(CODE "execute_process(COMMAND python ${PACK_SCRIPTS_PATH} ${name} ${INSTALL_DEPLOY_PATH})")
			install(CODE "execute_process(COMMAND python ${APP_SIGN_SCRIPT_PATH} ${name} ${INSTALL_DEPLOY_PATH})")
		endif()
	elseif( WIN32 )
		set( app_name "${name}.exe" )
		if( CMAKE_CONFIGURATION_TYPES )
			set( app_path "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${app_name}" )
			set( temp_dir "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/deployqt" )
		else()
			set( app_path "${CMAKE_CURRENT_BINARY_DIR}/${app_name}" )
			set( temp_dir "${CMAKE_CURRENT_BINARY_DIR}/deployqt" )
		endif()
		set( temp_app_path "${temp_dir}/${app_name}" )


		add_custom_command(
			TARGET ${DEPLOY_QT_TARGET}
			POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E remove_directory "${temp_dir}"
			COMMAND ${CMAKE_COMMAND} -E make_directory "${temp_dir}"
			COMMAND ${CMAKE_COMMAND} -E copy ${app_path} ${temp_app_path}
			COMMAND "${win_deploy_qt}"
				${temp_app_path}
				--no-angle
				--no-opengl-sw
				--no-quick-import
				--no-system-d3d-compiler
				--concurrent
				--verbose=1
			VERBATIM
		)
	
		if( NOT CMAKE_CONFIGURATION_TYPES )
			install(
				DIRECTORY ${temp_dir}/
				DESTINATION ${deploy_path}
			)
		else()
			install(
				DIRECTORY ${temp_dir}/
				CONFIGURATIONS Debug
				DESTINATION ${deploy_path}_debug
			)
		
			install(
				DIRECTORY ${temp_dir}/
				CONFIGURATIONS Release
				DESTINATION ${deploy_path}
			)
		
			install(
				DIRECTORY ${temp_dir}/
				CONFIGURATIONS RelWithDebInfo
				DESTINATION ${deploy_path}_withDebInfo
			)
		endif()
	endif()
endfunction()

unset( qmake_location )
unset( qt_bin_dir )
