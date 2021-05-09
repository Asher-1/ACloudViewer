cmake_minimum_required( VERSION 3.18 )

# PCL ENGINE BACKEND
if (USE_PCL_BACKEND)

    # PCL ENGINE
    project( QPCL_ENGINE_LIB )

    # CURRENT DIR
    file( GLOB header_list *.h)
    file(GLOB_RECURSE ui_list ${CMAKE_CURRENT_SOURCE_DIR}/*.ui)

    set( CMAKE_CXX_VISIBILITY_PRESET hidden )

    qt5_wrap_ui(ui_list_all ${ui_list})

    # VTK EXTENSIONS MODULE
    SET( QVTK_EXTENSIONS ${CMAKE_CURRENT_SOURCE_DIR}/VTKExtensions )
    # Core module
    file( GLOB_RECURSE VtkExtensions_Core_List ${QVTK_EXTENSIONS}/Core/*)
    source_group("VTKExtensions\\Core" FILES ${VtkExtensions_Core_List} )
    # Points module
    file( GLOB_RECURSE VtkExtensions_Points_List ${QVTK_EXTENSIONS}/Points/*)
    source_group("VTKExtensions\\Points" FILES ${VtkExtensions_Points_List} )
    # Widgets module
    file( GLOB_RECURSE VtkExtensions_Widgets_List ${QVTK_EXTENSIONS}/Widgets/*)
    source_group("VTKExtensions\\Widgets" FILES ${VtkExtensions_Widgets_List} )
    # CallbackTools module
    file( GLOB_RECURSE VtkExtensions_CallbackTools_List ${QVTK_EXTENSIONS}/CallbackTools/*)
    source_group("VTKExtensions\\CallbackTools" FILES ${VtkExtensions_CallbackTools_List} )
    # CosmoTools module
    file( GLOB_RECURSE VtkExtensions_CosmoTools_List ${QVTK_EXTENSIONS}/CosmoTools/*)
    source_group("VTKExtensions\\CosmoTools" FILES ${VtkExtensions_CosmoTools_List} )
    # Interaction style module
    file( GLOB_RECURSE VtkExtensions_InteractionStyle_List ${QVTK_EXTENSIONS}/InteractionStyle/*)
    source_group("VTKExtensions\\InteractionStyle" FILES ${VtkExtensions_InteractionStyle_List} )
    # Views module
    file( GLOB_RECURSE VtkExtensions_Views_List ${QVTK_EXTENSIONS}/Views/*)
    source_group("VTKExtensions\\Views" FILES ${VtkExtensions_Views_List} )
    # Utility module
    file( GLOB_RECURSE VtkExtensions_Utility_List ${QVTK_EXTENSIONS}/Utility/*)
    source_group("VTKExtensions\\Utility" FILES ${VtkExtensions_Utility_List} )

    set(VtkExtensions_list
            ${VtkExtensions_Core_List}
            ${VtkExtensions_Points_List}
            ${VtkExtensions_Widgets_List}
            ${VtkExtensions_CallbackTools_List}
            ${VtkExtensions_CosmoTools_List}
            ${VtkExtensions_InteractionStyle_List}
            ${VtkExtensions_Views_List}
            ${VtkExtensions_Utility_List}
    )

    # VTK UTILS MODULE
    SET( QVTK_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/VtkUtils )
    file( GLOB_RECURSE VtkUtils_list ${QVTK_UTILS}/*)
    source_group("VtkUtils" FILES ${VtkUtils_list} )

    # TOOLS MODULE
    SET( QPCL_TOOLS ${CMAKE_CURRENT_SOURCE_DIR}/Tools )
    file(GLOB Basic_list ${QPCL_TOOLS}/*.h ${QPCL_TOOLS}/*.cpp ${QPCL_TOOLS}/*.hpp)
    source_group("Tools" FILES ${Basic_list} )
    file( GLOB_RECURSE AnnotationTools_list ${QPCL_TOOLS}/AnnotationTools/*)
    source_group("Tools\\AnnotationTools" FILES ${AnnotationTools_list} )
    file( GLOB_RECURSE FilterTools_list ${QPCL_TOOLS}/FilterTools/*)
    source_group("Tools\\FilterTools" FILES ${FilterTools_list} )
    file( GLOB_RECURSE TransformTools_list ${QPCL_TOOLS}/TransformTools/*)
    source_group("Tools\\TransformTools" FILES ${TransformTools_list} )

    set(Tools_list
        ${Basic_list}
        ${AnnotationTools_list}
        ${FilterTools_list}
        ${TransformTools_list})

    # PCL VISUALIZATION MODULE
    if( WIN32 OR UNIX)
        add_subdirectory(visualization)
        SET( QPCL_VIS ${CMAKE_CURRENT_SOURCE_DIR}/visualization )
        file( GLOB_RECURSE PclVis_list ${QPCL_VIS}/*)
        source_group("visualization" FILES ${PclVis_list} )
    endif()
	
    # PCL UTILS MODULE
    SET( QPCL_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/PclUtils )
    file( GLOB_RECURSE PclUtils_list ${QPCL_UTILS}/*)
    source_group("PclUtils" FILES ${PclUtils_list} )


    # VTK WIDGETS MODULE
    SET( QVTK_WIDGETS ${CMAKE_CURRENT_SOURCE_DIR}/VtkWidgets )
    # filter module
    file( GLOB_RECURSE VtkWidgets_filters_list ${QVTK_WIDGETS}/filters/*)
    source_group("VtkWidgets\\filters" FILES ${VtkWidgets_filters_list} )

    # smallWidgets module
    file( GLOB_RECURSE VtkWidgets_smallWidgets_list ${QVTK_WIDGETS}/smallWidgets/*)
    source_group("VtkWidgets\\smallWidgets" FILES ${VtkWidgets_smallWidgets_list} )

    set(Widget_list ${VtkWidgets_filters_list}
                ${VtkWidgets_smallWidgets_list})

    add_library(${PROJECT_NAME} SHARED
                ${header_list}
                ${ui_list_all}
                ${VtkUtils_list}
                ${VtkExtensions_list}
                ${Tools_list}
                ${PclVis_list}
                ${PclUtils_list}
                ${Widget_list})

    if (MSVC)
        # fix compiling error on windows platform for deprecated flag in pcl
        target_compile_options(${PROJECT_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
    endif()

    target_include_directories( ${PROJECT_NAME} PUBLIC
                                ${PCL_INCLUDE_DIRS} )

    if( CMAKE_VERSION VERSION_LESS 3.13 )
            link_directories( ${PCL_LIBRARY_DIRS} )
            link_directories( ${VTK_LIBRARIES_DIRS} )
    else()
            target_link_directories(${PROJECT_NAME} PUBLIC
                                    ${PCL_LIBRARY_DIRS}
                                    ${VTK_LIBRARIES_DIRS})
    endif()

    # TBB
    if (MSVC)
        if (CVCORELIB_USE_TBB)
            include_directories( ${TBB_INCLUDE_DIRS})
        endif()
    endif()

    #define the PCL_VER_1_6_OR_OLDER preprocessor to compile qPCL with older versions of PCL
    if ( PCL_VERSION VERSION_LESS  1.7 ) # VERSION_GREATER Works just like "greater or equal"
            target_compile_definitions( ${PROJECT_NAME} PUBLIC PCL_VER_1_6_OR_OLDER )
    endif()

    # Luca's PCL patch support
    if( PCL_VERSION VERSION_GREATER  1.7 ) #from 1.7 the patch was merged
            target_compile_definitions( ${PROJECT_NAME} PUBLIC LP_PCL_PATCH_ENABLED )
            #set_property( TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS LP_PCL_PATCH_ENABLED )
    endif()

    # INTERNAL LIB
    target_link_libraries( ${PROJECT_NAME}
                            ECV_DB_LIB
                            ECV_IO_LIB )

    # Qt & PCL & VTK
    target_link_libraries( ${PROJECT_NAME}
                            Qt5::Core
                            Qt5::Gui
                            Qt5::Widgets
                            Qt5::PrintSupport
                            Qt5::OpenGL
                            ${PCL_LIBRARIES}
                            ${VTK_LIBRARIES} )

    target_compile_definitions( ${PROJECT_NAME} PRIVATE ECV_PCL_ENGINE_LIBRARY_BUILD )
    target_compile_definitions( ${PROJECT_NAME} PUBLIC USE_PCL_BACKEND )

    # install (shared) lib to specified destinations
    InstallSharedLibrary( TARGET ${PROJECT_NAME} )

	#import PCL dlls (if any, WIN32 only)
    include( cmake/ExportPCLDlls.cmake )
    export_PCL_dlls( ${EROWCLOUDVIEWER_DEST_FOLDER} )

	#import VTK dlls (if any, WIN32 only)
    include( cmake/ExportVTKDlls.cmake )
    export_VTK_dlls( ${EROWCLOUDVIEWER_DEST_FOLDER} )

endif()
