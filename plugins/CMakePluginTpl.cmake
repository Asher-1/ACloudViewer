### DEFAULT ECV "STANDARD" PLUGIN CMAKE SCRIPT ###

include ( ${ECV_CMAKE_SCRIPTS_DIR}/CMakePolicies.cmake NO_POLICY_SCOPE )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )
include_directories( ${ErowCloudViewerPlugins_SOURCE_DIR} )
include_directories( ${ErowCloudViewer_SOURCE_DIR}/../common )
include_directories( ${CV_CORE_LIB_SOURCE_DIR}/include )
include_directories( ${ECV_IO_LIB_SOURCE_DIR} )
include_directories( ${ECV_DB_LIB_SOURCE_DIR} )
if( MSVC )
    include_directories( ${ECV_DB_LIB_SOURCE_DIR}/msvc )
endif()
if (USE_PCL_BACKEND) 
    include_directories( ${PCL_INCLUDE_DIRS} )
	include_directories( ${QPCL_ENGINE_LIB_SOURCE_DIR})
endif()	
include_directories( ${EXTERNAL_LIBS_INCLUDE_DIR} )

file( GLOB header_list *.h)
file( GLOB source_list *.cpp)

# force link with interface implementations
list( APPEND source_list ${ErowCloudViewerPlugins_SOURCE_DIR}/ecvDefaultPluginInterface.cpp )

file( GLOB json_list *.json)
file( GLOB ui_list *.ui )
file( GLOB qrc_list *.qrc )
file( GLOB rc_list *.rc )

if ( ECV_PLUGIN_CUSTOM_HEADER_LIST )
    list( APPEND header_list ${ECV_PLUGIN_CUSTOM_HEADER_LIST} )
endif()

if ( ECV_PLUGIN_CUSTOM_SOURCE_LIST )
    list( APPEND source_list ${ECV_PLUGIN_CUSTOM_SOURCE_LIST} )
endif()

if (ECV_PLUGIN_CUSTOM_UI_LIST)
    list( APPEND ui_list ${ECV_PLUGIN_CUSTOM_UI_LIST} )
endif()


qt5_wrap_ui( generated_ui_list ${ui_list} )
qt5_add_resources( generated_qrc_list ${qrc_list} )

add_library( ${PROJECT_NAME} SHARED ${header_list} ${source_list} ${moc_list} ${generated_ui_list} ${generated_qrc_list} ${json_list} )

# Plugins need the QT_NO_DEBUG preprocessor in release!
if( WIN32 )
    set_property( TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS CV_USE_AS_DLL ECV_DB_USE_AS_DLL ECV_IO_USE_AS_DLL QPCL_ENGINE_USE_AS_DLL )
endif()

# Plugins need the QT_NO_DEBUG preprocessor in release!
if( WIN32 )
	if( NOT CMAKE_CONFIGURATION_TYPES )
		set_property( TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS QT_NO_DEBUG )
	else()
		#Anytime we use COMPILE_DEFINITIONS_XXX we must define this policy!
		#(and setting it outside of the function/file doesn't seem to work...)
		#cmake_policy(SET CMP0043 OLD)
	
		set_property( TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS_RELEASE QT_NO_DEBUG)
	endif()
endif()


target_link_libraries( ${PROJECT_NAME} CV_CORE_LIB )
target_link_libraries( ${PROJECT_NAME} ECV_DB_LIB )
target_link_libraries( ${PROJECT_NAME} ECV_IO_LIB )

if (USE_PCL_BACKEND)
	link_directories( ${PCL_LIBRARY_DIRS} )
	link_directories(${VTK_LIBRARIES_DIRS})
	# PCL
	target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})
	# VTK
	target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
	target_link_libraries( ${PROJECT_NAME} QPCL_ENGINE_LIB )
endif()	

# Qt
target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets Qt5::OpenGL Qt5::Concurrent)


if( APPLE )
    # put all the plugins we build into one directory
    set( PLUGINS_OUTPUT_DIR "${CMAKE_BINARY_DIR}/ECVPlugins" )

    file( MAKE_DIRECTORY "${PLUGINS_OUTPUT_DIR}" )

    set_target_properties( ${PROJECT_NAME}
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PLUGINS_OUTPUT_DIR}"
    )

    install( TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${EROWCLOUDVIEWER_MAC_PLUGIN_DIR} COMPONENT Runtime )
    set( EROWCLOUDVIEWER_PLUGINS ${EROWCLOUDVIEWER_PLUGINS} ${EROWCLOUDVIEWER_MAC_PLUGIN_DIR}/lib${PROJECT_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX} CACHE INTERNAL "EROWCLOUDVIEWER plugin list")
  elseif( UNIX )
    install_shared( ${PROJECT_NAME} ${CMAKE_INSTALL_LIBDIR}/EROWCLOUDVIEWER/plugins 0 )
  else()
    install_shared( ${PROJECT_NAME} ${EROWCLOUDVIEWER_DEST_FOLDER} 1 /plugins )
endif()
### END OF DEFAULT ECV PLUGIN CMAKE SCRIPT ###
