project(ErowCloudViewer VERSION ${PROJECT_VERSION})

OPTION( UPDATE_TRANSLATION_FILES "Generate or update ts files" OFF )

include( CMakePolicies NO_POLICY_SCOPE )

if (MSVC)
    if (CVCORELIB_USE_TBB)
		include_directories( ${TBB_INCLUDE_DIRS})
	endif()
endif()

if (BUILD_PYTHON_MODULE)
	include_directories( ${ECV_PYTHON_LIB_SOURCE_DIR} )
endif()

include_directories( ${ErowCloudViewerPlugins_SOURCE_DIR} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/db_tree )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/ui_templates )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

file( GLOB header_list *.h  )
file( GLOB source_list *.cpp )

file( GLOB ui_list ui_templates/*.ui )
file( GLOB qrc_list *.qrc Resources/themes/*.qrc )

file( GLOB txt_list TODO.txt ${ErowCloudViewerProjects_SOURCE_DIR}/CHANGELOG.md )

add_subdirectory( db_tree )
add_subdirectory( pluginManager )

add_subdirectory( extern/QCustomPlot EXCLUDE_FROM_ALL )

qt5_wrap_ui( generated_ui_list ${ui_list} )
qt5_add_resources( generated_qrc_list ${qrc_list} )

if( MSVC )
	# App icon with MSVC
	set( rc_list Resources/images/icon/ecv_icon.rc )
	
	#to get rid of the (system) console
	add_executable( ${PROJECT_NAME} WIN32 ${header_list} ${source_list} ${generated_ui_list} ${generated_qrc_list} ${rc_list} ${txt_list})
	
	# generate or update translation files(ts)
	if(UPDATE_TRANSLATION_FILES)
	    file(GLOB_RECURSE SOURCE_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../*.h ${CMAKE_CURRENT_SOURCE_DIR}/../*.ui)
		# message("SOURCE_LIST : ${SOURCE_LIST}")
		find_package(Qt5Core REQUIRED)
		find_package(Qt5LinguistTools REQUIRED)
		#qt5_use_modules(${PROJECT_NAME} Core)
		set( TRANSLATION_FILES ${CMAKE_CURRENT_SOURCE_DIR}/translations/ErowCloudViewer_zh.ts)
		#file( GLOB TRANSLATION_FILES ${CMAKE_CURRENT_LIST_DIR}/translations/*.ts )
		foreach( TRANSLATION_FILE ${TRANSLATION_FILES} )
			message("TRANSLATION_FILE : ${TRANSLATION_FILE}")
			qt5_update_translation(QMs ${PROJECT_NAME} ${SOURCE_LIST} ${TRANSLATION_FILE})
			#qt5_create_translation(QMs ${PROJECT_NAME} ${SOURCE_LIST} ${TRANSLATION_FILE})
			message("QMs: ${QMs}")
		endforeach()
	endif()
elseif( APPLE )
        add_executable( ${PROJECT_NAME} MACOSX_BUNDLE ${header_list} ${source_list} ${generated_ui_list} ${generated_qrc_list} ${txt_list} )
else()
	add_executable( ${PROJECT_NAME} ${header_list} ${source_list} ${generated_ui_list} ${generated_qrc_list} ${rc_list} ${txt_list} )
endif()

if (WIN32)
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/ErowCloudViewer.exe ${CMAKE_BINARY_DIR}/eCV/$<CONFIG>/ErowCloudViewer.exe
            VERBATIM
    )
else()
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/bin/ErowCloudViewer ${CMAKE_BINARY_DIR}/eCV/$<CONFIG>/ErowCloudViewer
            VERBATIM
    )
endif()


DeployQt(
	TARGET
		${PROJECT_NAME}
	DEPLOY_PATH
		${PROJECT_NAME}
)

target_link_libraries( ${PROJECT_NAME}
    CVAppCommon
    QCustomPlot
    Qt5::PrintSupport
)

if( CVCORELIB_USE_QT_CONCURRENT )
	# We want *deployqt to handle this dependency, so instead of just copying the file
	# link to the target.
	target_link_libraries( ${PROJECT_NAME}
		Qt5::Concurrent
	)
endif()

if (BUILD_PYTHON_MODULE)
	target_link_libraries( ${PROJECT_NAME} ECV_PYTHON_LIB )
	# Add custom preprocessor definitions
	target_compile_definitions( ${PROJECT_NAME} PRIVATE USE_PYTHON_MODULE )
endif()	

# Add custom preprocessor definitions
if( WIN32 )
	set_property( TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS)
	if (MSVC)
		set_target_properties( ${PROJECT_NAME} PROPERTIES LINK_FLAGS " /MANIFEST:NO" )
	endif()
endif()

# App icon with Code::Blocks/MinGW
if( WIN32 )
	if( MINGW )
		if (NOT CMAKE_RC_COMPILER)
			set( CMAKE_RC_COMPILER windres )
		endif()
		add_custom_command( TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_RC_COMPILER} -i ${CMAKE_CURRENT_SOURCE_DIR}/Resources/images/icon/ecv_icon.rc --input-format=rc -o ${CMAKE_CURRENT_BINARY_DIR}/ecv_icon.res -O coff )
	endif()
endif()

# install program
install_ext( TARGETS ${PROJECT_NAME} ${EROWCLOUDVIEWER_DEST_FOLDER} "" )

# Auxiliary files
set( auxFiles  ${ErowCloudViewerProjects_SOURCE_DIR}/CHANGELOG.md bin_other/license.txt bin_other/global_shift_list_template.txt )

if( WIN32 )
	# Additional auxiliary file(s)
	file( GLOB batch_list bin_other/*.bat )
	list( APPEND auxFiles ${batch_list} )
endif()

# Install auxiliary files
if( WIN32 OR APPLE )
	foreach( filename ${auxFiles} )
		install_ext( FILES ${filename} ${EROWCLOUDVIEWER_DEST_FOLDER} "" )
	endforeach()
else( UNIX )
	foreach( filename ${auxFiles} )
		install_ext( FILES ${filename} share/ErowCloudViewer "" )
	endforeach()
endif()

# Translation
add_subdirectory( translations )

# Put together our Application bundle on macOS
if( APPLE )    
    add_subdirectory( Mac )
endif()

# Install plugins and shaders in the correct folder for each platform
if( APPLE )	
	InstallPlugins(
		DEST_PATH "${EROWCLOUDVIEWER_MAC_PLUGIN_DIR}"
		SHADER_DEST_PATH "${EROWCLOUDVIEWER_MAC_BASE_DIR}/Contents/Shaders"
	)
elseif( UNIX )
	InstallPlugins(
		DEST_PATH "${CMAKE_INSTALL_LIBDIR}/ErowCloudViewer/plugins"
		SHADER_DEST_PATH share/ErowCloudViewer/shaders
	)
else()
	InstallPlugins(
		DEST_PATH "${EROWCLOUDVIEWER_DEST_FOLDER}"
		DEST_FOLDER "plugins"
		SHADER_DEST_PATH "${EROWCLOUDVIEWER_DEST_FOLDER}"
		SHADER_DEST_FOLDER "shaders"
	)
endif()
