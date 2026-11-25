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
	
	# The deploy step consumes the plugin binaries (macOS copies them into
	# the bundle before macdeployqt; Windows scans them below), so they must
	# be built before this target's POST_BUILD runs. Filter by TARGET: this
	# function also runs for CloudViewer before plugins/ is processed, where
	# the cached plugin list may still name targets from a previous configure.
	set( _deploy_qt_plugins "" )
	foreach( _plugin_target ${CV_PLUGIN_TARGET_LIST} )
		if( TARGET ${_plugin_target} )
			list( APPEND _deploy_qt_plugins ${_plugin_target} )
		endif()
	endforeach()
	if( _deploy_qt_plugins )
		add_dependencies( ${DEPLOY_QT_TARGET} ${_deploy_qt_plugins} )
	endif()
	
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
		
		# Single source of truth for the backend-module suffix is
		# AICoreRuntimeLayout.cmake (GGML_MODULE_SUFFIX: ".so" on all
		# non-Windows platforms, cache value from ggml configure takes
		# precedence). Backend modules are MODULE libraries and are
		# "*.so" even on macOS — NOT CMAKE_SHARED_LIBRARY_SUFFIX
		# (".dylib"): ggml-backend-reg dlopen()s ".so" on all UNIX, so
		# copying "*.dylib" here would miss libggml-cpu.so /
		# libggml-metal.so and leave the bundle without any ggml device.
		if(NOT DEFINED GGML_MODULE_SUFFIX)
		        set(acv_ggml_module_suffix ".so")
		else()
		        set(acv_ggml_module_suffix "${GGML_MODULE_SUFFIX}")
		endif()
		add_custom_command(
				TARGET ${DEPLOY_QT_TARGET}
				POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E remove_directory "${temp_dir}"
				COMMAND ${CMAKE_COMMAND} -E make_directory "${temp_dir}"
				COMMAND ${CMAKE_COMMAND} -E copy_directory ${app_path} ${temp_app_path}
				COMMAND "${mac_deploy_qt}" ${temp_app_path} -verbose=1
				# ggml backend modules are dlopen'd at runtime and
				# invisible to macdeployqt's otool scan. libAICore
				# resolves its module dir via dladdr (the directory
				# containing libAICore.dylib inside the bundle), so a
				# bundle without these modules reports zero ggml
				# devices (no CPU, no Metal) and every AICore task
				# fails at model load.
				COMMAND sh -c "cp ${CMAKE_BINARY_DIR}/bin/libggml-*${acv_ggml_module_suffix} '${temp_app_path}/Contents/Frameworks/'"
				# Remove PostgreSQL SQL driver plugin which links against libpq
				# (uses private macOS APIs, not Mac App Store compliant).
				# cmake -E remove never fails if the file doesn't exist.
				COMMAND ${CMAKE_COMMAND} -E remove "${temp_app_path}/Contents/PlugIns/sqldrivers/libqsqlpsql.dylib"
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
			# Resolve the Python prefix that was configured at CMake time so the
			# bundler embeds the correct Python even when make install is run
			# from a different conda env or system Python.
			execute_process(
				COMMAND "${Python3_EXECUTABLE}" -c "import sys; print(sys.exec_prefix, end='')"
				OUTPUT_VARIABLE _PY_BUNDLE_PREFIX
				OUTPUT_STRIP_TRAILING_WHITESPACE
			)
			install(CODE "execute_process(COMMAND python ${PACK_SCRIPTS_PATH} ${name} ${INSTALL_DEPLOY_PATH} --embed_python --python_prefix \"${_PY_BUNDLE_PREFIX}\")")
			install(CODE "execute_process(COMMAND python ${APP_SIGN_SCRIPT_PATH} ${name} ${INSTALL_DEPLOY_PATH} --embed_python)")
		else()
			install(CODE "execute_process(COMMAND python ${PACK_SCRIPTS_PATH} ${name} ${INSTALL_DEPLOY_PATH})")
			install(CODE "execute_process(COMMAND python ${APP_SIGN_SCRIPT_PATH} ${name} ${INSTALL_DEPLOY_PATH})")
		endif()
	elseif( WIN32 )
		set( app_name "${name}.exe" )
		set( _deploy_qt_plugin_commands "" )
		foreach( _plugin_target ${_deploy_qt_plugins} )
			list( APPEND _deploy_qt_plugin_commands
					COMMAND "${win_deploy_qt}"
						$<TARGET_FILE:${_plugin_target}>
						--dir ${temp_app_path}
						--no-angle
						--no-opengl-sw
						--no-quick-import
						--no-system-d3d-compiler
						--verbose=1 )
		endforeach()
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
			# windeployqt only scans the main executable, so Qt modules linked
			# exclusively by plugins (e.g. WebSockets for qJSONRPC) would be
			# missing from the deploy tree. Scan every plugin dll into the same
			# directory so the tree stays self-contained without IFW packaging.
			${_deploy_qt_plugin_commands}
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
