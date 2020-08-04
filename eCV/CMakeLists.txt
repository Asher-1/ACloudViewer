cmake_minimum_required(VERSION 3.5)

project(ErowCloudViewer VERSION ${PROJECT_VERSION})

OPTION( UPDATE_TRANSLATION_FILES "Generate or update ts files" OFF )

include ( ${ECV_CMAKE_SCRIPTS_DIR}/CMakePolicies.cmake NO_POLICY_SCOPE )

include_directories( ${CV_CORE_LIB_SOURCE_DIR}/include )

include_directories( ${ECV_DB_LIB_SOURCE_DIR} )
if( MSVC )
   include_directories( ${ECV_DB_LIB_SOURCE_DIR}/msvc )
endif()

include_directories( ${ECV_IO_LIB_SOURCE_DIR} )

if (USE_PCL_BACKEND) 
	include_directories( ${PCL_INCLUDE_DIRS} )
	include_directories( ${QPCL_ENGINE_LIB_SOURCE_DIR})
endif()

if (BUILD_PYTHON_MODULE)
	include_directories( ${ECV_PYTHON_LIB_SOURCE_DIR} )
endif()

include_directories( ${ErowCloudViewerPlugins_SOURCE_DIR} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../common )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/db_tree )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/ui_templates )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../libs/qcustomplot )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

file( GLOB header_list *.h  )
file( GLOB source_list *.cpp )

file( GLOB ui_list ui_templates/*.ui )
file( GLOB qrc_list *.qrc Resources/themes/*.qrc )

file( GLOB txt_list TODO.txt ${ErowCloudViewerProjects_SOURCE_DIR}/CHANGELOG.md )

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../common ${CMAKE_CURRENT_BINARY_DIR}/common)
add_subdirectory( db_tree )
add_subdirectory( pluginManager )

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
	
else()
	add_executable( ${PROJECT_NAME} ${header_list} ${source_list} ${generated_ui_list} ${generated_qrc_list} ${rc_list} ${txt_list} )
endif()

target_link_libraries( ${PROJECT_NAME} CV_CORE_LIB )
target_link_libraries( ${PROJECT_NAME} ECV_DB_LIB )
target_link_libraries( ${PROJECT_NAME} ECV_IO_LIB )
target_link_libraries( ${PROJECT_NAME} qcustomplot )

# Qt
target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets Qt5::OpenGL Qt5::PrintSupport Qt5::Svg)
if (WIN32)
	target_link_libraries( ${PROJECT_NAME} Qt5::WinMain )
endif()

if (USE_PCL_BACKEND)
	link_directories( ${PCL_LIBRARY_DIRS} )
	link_directories( ${VTK_LIBRARIES_DIRS} )
	# PCL
	target_link_libraries( ${PROJECT_NAME} ${PCL_LIBRARIES} )
	# VTK
	target_link_libraries( ${PROJECT_NAME} ${VTK_LIBRARIES} )
	target_link_libraries( ${PROJECT_NAME} QPCL_ENGINE_LIB )
	# Add custom preprocessor definitions
	target_compile_definitions( ${PROJECT_NAME} PRIVATE ECV_PCL_ENGINE_LIBRARY_BUILD )
	
endif()	

if (BUILD_PYTHON_MODULE)
	target_link_libraries( ${PROJECT_NAME} ${3RDPARTY_LIBRARIES} )
	target_link_libraries( ${PROJECT_NAME} ECV_PYTHON_LIB )
	# Add custom preprocessor definitions
	target_compile_definitions( ${PROJECT_NAME} PRIVATE ECV_PYTHON_LIBRARY_BUILD )
endif()	

# Add custom preprocessor definitions
if( WIN32 )
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
install_ext( TARGETS ${PROJECT_NAME} ${EROWCLOUDVIEWER_DEST_FOLDER} )

# Auxiliary files
set( auxFiles  ${ErowCloudViewerProjects_SOURCE_DIR}/CHANGELOG.md bin_other/license.txt bin_other/global_shift_list_template.txt )

if( WIN32 )
	# Export Qt dlls
	install_Qt_Dlls( ${EROWCLOUDVIEWER_DEST_FOLDER} ${OPTION_SUPPORT_GAMEPADS} )
	install_Qt_ImageFormats( ${EROWCLOUDVIEWER_DEST_FOLDER} )
	install_Qt5_plugins( ${EROWCLOUDVIEWER_DEST_FOLDER} )

	# Additional auxiliary file(s)
	file( GLOB batch_list bin_other/*.bat )
	list( APPEND auxFiles ${batch_list} )
endif()

# Install auxiliary files
if( WIN32 OR APPLE )
	foreach( filename ${auxFiles} )
		install_ext( FILES ${filename} ${EROWCLOUDVIEWER_DEST_FOLDER} )
	endforeach()
else( UNIX )
	foreach( filename ${auxFiles} )
		install_ext( FILES ${filename} share/ErowCloudViewer )
	endforeach()
endif()

# Translation
add_subdirectory( translations )
