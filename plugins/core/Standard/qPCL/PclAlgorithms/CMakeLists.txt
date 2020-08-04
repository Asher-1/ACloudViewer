set_property(GLOBAL PROPERTY USE_FOLDERS On)
cmake_minimum_required (VERSION 3.0)

# https://cmake.org/cmake/help/git-stage/policy/CMP0074.html
# cmake_policy(SET CMP0074 OLD)

if( NOT MSVC )
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -lpthread" )
endif()

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${ErowCloudViewerPlugins_SOURCE_DIR} )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )
include_directories( ${EXTERNAL_LIBS_INCLUDE_DIR} )
include_directories( ${CV_CORE_LIB_SOURCE_DIR}/include )
include_directories( ${ECV_DB_LIB_SOURCE_DIR} )
if( MSVC )
  include_directories( ${ECV_DB_LIB_SOURCE_DIR}/msvc )
endif()

project( QPCL_PLUGIN_ALGORIGHM_LIB )

include ( ${ECV_CMAKE_SCRIPTS_DIR}/CMakePolicies.cmake NO_POLICY_SCOPE )

if (USE_PCL_BACKEND) 
	include_directories( ${PCL_INCLUDE_DIRS} )
	include_directories( ${QPCL_ENGINE_LIB_SOURCE_DIR})

	# CURRENT DIR
	file( GLOB header_list *.h)
	file( GLOB source_list *.cpp)
	file(GLOB_RECURSE ui_list  ${CMAKE_CURRENT_SOURCE_DIR}/*.ui)
	qt5_wrap_ui(ui_list_all ${ui_list})
	
	# PCL FILTERS
	file( GLOB_RECURSE filter_list 			${CMAKE_CURRENT_SOURCE_DIR}/Filters/*)
	source_group("PclAlgorithms\\Filters" FILES ${filter_list} )

	# PCL FEATURES
	file( GLOB_RECURSE feature_list 			${CMAKE_CURRENT_SOURCE_DIR}/Features/*)
	source_group("PclAlgorithms\\Features" FILES ${feature_list} )
	
	# PCL SURFACES
	file( GLOB_RECURSE surface_list 		${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/*)
	source_group("PclAlgorithms\\Surfaces" FILES ${surface_list} )

	# PCL SEGMENTATIONS
	file( GLOB_RECURSE segment_list 		${CMAKE_CURRENT_SOURCE_DIR}/Segmentations/*)
	source_group("PclAlgorithms\\Segmentations" FILES ${segment_list} )

	# PCL RECOGNITIONS
	file( GLOB_RECURSE recognition_list 	${CMAKE_CURRENT_SOURCE_DIR}/Recognitions/*)
	source_group("PclAlgorithms\\Recognitions" FILES ${recognition_list} )
	
	# add library
	add_library(
	${PROJECT_NAME} STATIC
	${header_list} 
	${source_list} 
	${ui_list_all}
	${filter_list}
	${feature_list}
	${surface_list} 
	${segment_list}
	${recognition_list}
	)

	link_directories( ${PCL_LIBRARY_DIRS} )
	add_definitions( ${PCL_DEFINITIONS} )
	target_link_libraries( ${PROJECT_NAME} ${PCL_LIBRARIES})
	target_link_libraries( ${PROJECT_NAME} QPCL_ENGINE_LIB )
	
endif()	

# Qt
target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets)

