cmake_minimum_required(VERSION 3.5)

# if( MSVC )
    # add_compile_options(-bigobj)
# endif()

# PCL ENGINE BACKEND
if (USE_PCL_BACKEND)
	# find package
	link_directories( ${PCL_LIBRARY_DIRS} )
	link_directories( ${VTK_LIBRARIES_DIRS} )

	include_directories( ${CV_CORE_LIB_SOURCE_DIR}/include )
	include_directories( ${ECV_DB_LIB_SOURCE_DIR} )
	if( MSVC )
	   include_directories( ${ECV_DB_LIB_SOURCE_DIR}/msvc )
	endif()
	include_directories( ${ECV_IO_LIB_SOURCE_DIR} )
	include_directories( ${qcustomplot_SOURCE_DIR} )

	# PCL ENGINE
	project( QPCL_ENGINE_LIB )

	# CURRENT DIR
	file( GLOB header_list 	*.h)
	file(GLOB_RECURSE ui_list    ${CMAKE_CURRENT_SOURCE_DIR}/*.ui)

	set( CMAKE_CXX_VISIBILITY_PRESET hidden )

	qt5_wrap_ui(ui_list_all ${ui_list})
	
	# VTK EXTENSIONS MODULE
	SET( QVTK_EXTENSIONS ${CMAKE_CURRENT_SOURCE_DIR}/VTKExtensions )
	# core module
	file( GLOB_RECURSE VtkExtensions_core_list ${QVTK_EXTENSIONS}/Core/*)
	source_group("VTKExtensions\\Core" FILES ${VtkExtensions_core_list} )
	# Extraction module
	file( GLOB_RECURSE VtkExtensions_extraction_list ${QVTK_EXTENSIONS}/Extraction/*)
	source_group("VTKExtensions\\Extraction" FILES ${VtkExtensions_extraction_list} )
	# CosmoTools module
	file( GLOB_RECURSE VtkExtensions_cosmoTools_list ${QVTK_EXTENSIONS}/CosmoTools/*)
	source_group("VTKExtensions\\CosmoTools" FILES ${VtkExtensions_cosmoTools_list} )
	# interaction style module
	file( GLOB_RECURSE VtkExtensions_interactionStyle_list ${QVTK_EXTENSIONS}/InteractionStyle/*)
	source_group("VTKExtensions\\InteractionStyle" FILES ${VtkExtensions_interactionStyle_list} )
	# views module
	file( GLOB_RECURSE VtkExtensions_views_list ${QVTK_EXTENSIONS}/Views/*)
	source_group("VTKExtensions\\Views" FILES ${VtkExtensions_views_list} )

	set(VtkExtensions_list 
		${VtkExtensions_core_list}  
		${VtkExtensions_extraction_list}  
		${VtkExtensions_cosmoTools_list}  
		${VtkExtensions_interactionStyle_list}  
		${VtkExtensions_views_list}
	)
	
	# VTK UTILS MODULE
	SET( QVTK_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/VtkUtils )
	file( GLOB_RECURSE VtkUtils_list 		${QVTK_UTILS}/*)
	source_group("VtkUtils" FILES ${VtkUtils_list} )
	
	# TOOLS MODULE
	SET( QPCL_TOOLS ${CMAKE_CURRENT_SOURCE_DIR}/Tools )
	file(GLOB Basic_list ${QPCL_TOOLS}/*.h ${QPCL_TOOLS}/*.cpp ${QPCL_TOOLS}/*.hpp)
	source_group("Tools" FILES ${Basic_list} )
	file( GLOB_RECURSE AnnotationTools_list     ${QPCL_TOOLS}/AnnotationTools/*)
	source_group("Tools\\AnnotationTools" FILES ${AnnotationTools_list} )
	file( GLOB_RECURSE FilterTools_list 		${QPCL_TOOLS}/FilterTools/*)
	source_group("Tools\\FilterTools" FILES ${FilterTools_list} )
	file( GLOB_RECURSE TransformTools_list 		${QPCL_TOOLS}/TransformTools/*)
	source_group("Tools\\TransformTools" FILES ${TransformTools_list} )
	
	set(Tools_list ${Basic_list} ${AnnotationTools_list} ${FilterTools_list} ${TransformTools_list})
	
	# PCL VISUALIZATION MODULE
    if( WIN32 OR UNIX)
        add_subdirectory(visualization)
        SET( QPCL_VIS	${CMAKE_CURRENT_SOURCE_DIR}/visualization )
        file( GLOB_RECURSE PclVis_list 		${QPCL_VIS}/*)
        source_group("visualization" FILES ${PclVis_list} )
    endif()
	
	# PCL UTILS MODULE
	SET( QPCL_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/PclUtils )
	file( GLOB_RECURSE PclUtils_list 		${QPCL_UTILS}/*)
	source_group("PclUtils" FILES ${PclUtils_list} )
	

	# VTK WIDGETS MODULE
	SET( QVTK_WIDGETS ${CMAKE_CURRENT_SOURCE_DIR}/VtkWidgets )
	# filter module
	file( GLOB_RECURSE VtkWidgets_filters_list 		${QVTK_WIDGETS}/filters/*)
	source_group("VtkWidgets\\filters" FILES ${VtkWidgets_filters_list} )
	
	# surface module
	file( GLOB_RECURSE VtkWidgets_surface_list 		${QVTK_WIDGETS}/surface/*)	
	source_group("VtkWidgets\\surface" FILES ${VtkWidgets_surface_list} )
	
	# smallWidgets module
	file( GLOB_RECURSE VtkWidgets_smallWidgets_list ${QVTK_WIDGETS}/smallWidgets/*)
	source_group("VtkWidgets\\smallWidgets" FILES ${VtkWidgets_smallWidgets_list} )
	
	set(Widget_list ${VtkWidgets_filters_list} 
					${VtkWidgets_surface_list}
					${VtkWidgets_smallWidgets_list})
	
	add_library( ${PROJECT_NAME} SHARED 
									${header_list} 
									${ui_list_all} 
									${VtkUtils_list}
									${VtkExtensions_list}
									${Tools_list} 
									${PclVis_list} 
									${PclUtils_list} 
									${Widget_list}
	)

	#define the PCL_VER_1_6_OR_OLDER preprocessor to compile qPCL with older versions of PCL
	if ( PCL_VERSION VERSION_LESS  1.7 ) # VERSION_GREATER Works just like "greater or equal"
		set_property( TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS PCL_VER_1_6_OR_OLDER )
	endif()

	# Luca's PCL patch support
	if( PCL_VERSION VERSION_GREATER  1.7 ) #from 1.7 the patch was merged
		set_property( TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS LP_PCL_PATCH_ENABLED )
	endif()

	# INTERNAL LIB
	target_link_libraries( ${PROJECT_NAME} CV_CORE_LIB )
	target_link_libraries( ${PROJECT_NAME} ECV_DB_LIB )
	target_link_libraries( ${PROJECT_NAME} ECV_IO_LIB )
	target_link_libraries( ${PROJECT_NAME} qcustomplot )
	
	# Qt & PCL & VTK
	target_link_libraries(${PROJECT_NAME} 
	Qt5::Core 
	Qt5::Gui 
	Qt5::Widgets
	Qt5::PrintSupport
	${PCL_LIBRARIES} 
	${VTK_LIBRARIES}
	)

	# Add custom preprocessor definitions
	target_compile_definitions( ${PROJECT_NAME} PRIVATE ECV_PCL_ENGINE_LIBRARY_BUILD )

	# install (shared) lib to specified destinations
	if( WIN32)
            foreach( dest ${INSTALL_DESTINATIONS} )
                    install_shared( ${PROJECT_NAME} ${dest} 1 )
            endforeach()
	else()
            install_shared( ${PROJECT_NAME} ${CMAKE_INSTALL_LIBDIR}/ErowCloudViewer 0 ) #default destination: /usr/lib
	endif()
endif()
