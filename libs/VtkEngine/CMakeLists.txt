cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

# VTK ENGINE BACKEND
if (USE_VTK_BACKEND)
    # VTK ENGINE
    project(QVTK_ENGINE_LIB)

    # CURRENT DIR
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
    file(GLOB header_list *.h)
    file(GLOB_RECURSE ui_list ${CMAKE_CURRENT_SOURCE_DIR}/*.ui)

    # Qt5/Qt6 compatibility
    if(USE_QT6)
        qt6_wrap_ui(ui_list_all ${ui_list})
    else()
        qt5_wrap_ui(ui_list_all ${ui_list})
    endif()

    # VTK EXTENSIONS MODULE
    SET(QVTK_EXTENSIONS ${CMAKE_CURRENT_SOURCE_DIR}/VTKExtensions)
    # Core module
    file(GLOB_RECURSE VtkExtensions_Core_List ${QVTK_EXTENSIONS}/Core/*)
    source_group("VTKExtensions\\Core" FILES ${VtkExtensions_Core_List})
    # Points module
    file(GLOB_RECURSE VtkExtensions_Points_List ${QVTK_EXTENSIONS}/Points/*)
    source_group("VTKExtensions\\Points" FILES ${VtkExtensions_Points_List})
    # Widgets module
    file(GLOB_RECURSE VtkExtensions_Widgets_List ${QVTK_EXTENSIONS}/Widgets/*)
    source_group("VTKExtensions\\Widgets" FILES ${VtkExtensions_Widgets_List})
    # CallbackTools module
    file(GLOB_RECURSE VtkExtensions_CallbackTools_List ${QVTK_EXTENSIONS}/CallbackTools/*)
    source_group("VTKExtensions\\CallbackTools" FILES ${VtkExtensions_CallbackTools_List})
    # CosmoTools module
    file(GLOB_RECURSE VtkExtensions_CosmoTools_List ${QVTK_EXTENSIONS}/CosmoTools/*)
    source_group("VTKExtensions\\CosmoTools" FILES ${VtkExtensions_CosmoTools_List})
    # Interaction style module
    file(GLOB_RECURSE VtkExtensions_InteractionStyle_List ${QVTK_EXTENSIONS}/InteractionStyle/*)
    source_group("VTKExtensions\\InteractionStyle" FILES ${VtkExtensions_InteractionStyle_List})
    # Views module
    file(GLOB_RECURSE VtkExtensions_Views_List ${QVTK_EXTENSIONS}/Views/*)
    source_group("VTKExtensions\\Views" FILES ${VtkExtensions_Views_List})
    # Utility module
    file(GLOB_RECURSE VtkExtensions_Utility_List ${QVTK_EXTENSIONS}/Utility/*)
    source_group("VTKExtensions\\Utility" FILES ${VtkExtensions_Utility_List})
    # ConstrainedWidgets module
    file(GLOB_RECURSE VtkExtensions_ConstrainedWidgets_List ${QVTK_EXTENSIONS}/ConstrainedWidgets/*)
    source_group("VTKExtensions\\ConstrainedWidgets" FILES ${VtkExtensions_ConstrainedWidgets_List})

    set(VtkExtensions_list
            ${VtkExtensions_Core_List}
            ${VtkExtensions_Points_List}
            ${VtkExtensions_Widgets_List}
            ${VtkExtensions_CallbackTools_List}
            ${VtkExtensions_CosmoTools_List}
            ${VtkExtensions_InteractionStyle_List}
            ${VtkExtensions_Views_List}
            ${VtkExtensions_Utility_List}
            ${VtkExtensions_ConstrainedWidgets_List}
            )

    # VTK UTILS MODULE
    SET(QVTK_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/VtkUtils)
    file(GLOB_RECURSE VtkUtils_list ${QVTK_UTILS}/*)
    source_group("VtkUtils" FILES ${VtkUtils_list})

    # TOOLS MODULE
    SET(QVTK_TOOLS ${CMAKE_CURRENT_SOURCE_DIR}/Tools)
    file(GLOB Basic_list ${QVTK_TOOLS}/*.h ${QVTK_TOOLS}/*.cpp ${QVTK_TOOLS}/*.hpp)
    source_group("Tools" FILES ${Basic_list})
    file(GLOB_RECURSE AnnotationTools_list ${QVTK_TOOLS}/AnnotationTools/*)
    source_group("Tools\\AnnotationTools" FILES ${AnnotationTools_list})
    file(GLOB_RECURSE CameraTools_list ${QVTK_TOOLS}/CameraTools/*)
    source_group("Tools\\CameraTools" FILES ${CameraTools_list})
    file(GLOB_RECURSE ColorTools_list ${QVTK_TOOLS}/ColorTools/*)
    source_group("Tools\\ColorTools" FILES ${ColorTools_list})
    file(GLOB_RECURSE Common_list ${QVTK_TOOLS}/Common/*)
    source_group("Tools\\Common" FILES ${Common_list})
    file(GLOB_RECURSE FilterTools_list ${QVTK_TOOLS}/FilterTools/*)
    source_group("Tools\\FilterTools" FILES ${FilterTools_list})
    file(GLOB_RECURSE MeasurementTools_list ${QVTK_TOOLS}/MeasurementTools/*)
    source_group("Tools\\MeasurementTools" FILES ${MeasurementTools_list})
    file(GLOB_RECURSE PickingTools_list ${QVTK_TOOLS}/PickingTools/*)
    source_group("Tools\\PickingTools" FILES ${PickingTools_list})
    file(GLOB_RECURSE TransformTools_list ${QVTK_TOOLS}/TransformTools/*)
    source_group("Tools\\TransformTools" FILES ${TransformTools_list})
    file(GLOB_RECURSE SelectionTools_list ${QVTK_TOOLS}/SelectionTools/*)
    source_group("Tools\\SelectionTools" FILES ${SelectionTools_list})

    set(Tools_list
            ${Basic_list}
            ${AnnotationTools_list}
            ${CameraTools_list}
            ${ColorTools_list}
            ${Common_list}
            ${FilterTools_list}
            ${MeasurementTools_list}
            ${PickingTools_list}
            ${TransformTools_list}
            ${SelectionTools_list})

    # CONVERTERS MODULE (Direct CV_db -> VTK conversion)
    SET(QVTK_CONVERTERS ${CMAKE_CURRENT_SOURCE_DIR}/Converters)
    file(GLOB_RECURSE Converters_list ${QVTK_CONVERTERS}/*)
    source_group("Converters" FILES ${Converters_list})

    # VTK RENDERING MODULE (Actor maps, event callbacks, rendering utilities)
    SET(QVTK_RENDERING ${CMAKE_CURRENT_SOURCE_DIR}/VtkRendering)
    file(GLOB_RECURSE VtkRendering_Core_list ${QVTK_RENDERING}/Core/*)
    source_group("VtkRendering\\Core" FILES ${VtkRendering_Core_list})
    file(GLOB_RECURSE VtkRendering_Rendering_list ${QVTK_RENDERING}/Rendering/*)
    source_group("VtkRendering\\Rendering" FILES ${VtkRendering_Rendering_list})
    file(GLOB_RECURSE VtkRendering_Context_list ${QVTK_RENDERING}/Context/*)
    source_group("VtkRendering\\Context" FILES ${VtkRendering_Context_list})
    set(VtkRendering_Fixes_list
        ${QVTK_RENDERING}/Fixes/vtkRenderWindowInteractorFix.h 
        ${QVTK_RENDERING}/Fixes/vtkRenderWindowInteractorFix.cpp)
    if(${VTK_VERSION} VERSION_GREATER_EQUAL 9.0 AND UNIX AND NOT APPLE)
        list(APPEND VtkRendering_Fixes_list 
            ${QVTK_RENDERING}/Fixes/vtkFixedXRenderWindowInteractor.h 
            ${QVTK_RENDERING}/Fixes/vtkFixedXRenderWindowInteractor.cpp)
    endif()
    if(APPLE)
        list(APPEND VtkRendering_Fixes_list ${QVTK_RENDERING}/Fixes/vtkRenderWindowInteractorFix.mm)
    endif()
    source_group("VtkRendering\\Fixes" FILES ${VtkRendering_Fixes_list})
    set(VtkRendering_list
        ${VtkRendering_Core_list}
        ${VtkRendering_Rendering_list}
        ${VtkRendering_Context_list}
        ${VtkRendering_Fixes_list})

    # VISUALIZATION MODULE (3D/2D visualizer and display tools)
    SET(QVTK_VISUALIZATION ${CMAKE_CURRENT_SOURCE_DIR}/Visualization)
    file(GLOB_RECURSE Visualization_list ${QVTK_VISUALIZATION}/*)
    source_group("Visualization" FILES ${Visualization_list})

    add_library(${PROJECT_NAME} SHARED
            ${header_list}
            ${ui_list_all}
            ${Converters_list}
            ${VtkRendering_list}
            ${VtkUtils_list}
            ${VtkExtensions_list}
            ${Tools_list}
            ${Visualization_list})

    # Enable AUTOMOC and AUTOUIC for this target as we now use QObject based classes
    set_target_properties(${PROJECT_NAME} PROPERTIES AUTOMOC ON AUTOUIC ON)

    # Enable automatic symbol export on Windows to avoid MOC issues with export macros
    if (WIN32)
        set_target_properties(${PROJECT_NAME} PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif ()

    if (MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/sdl->")
    endif ()

    target_include_directories(${PROJECT_NAME} PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/Visualization")

    if (CMAKE_VERSION VERSION_LESS 3.13)
        link_directories(${VTK_LIBRARIES_DIRS})
    else ()
        target_link_directories(${PROJECT_NAME} PUBLIC
                ${VTK_LIBRARIES_DIRS})
    endif ()

    # TBB
    if (MSVC)
        if (CVCORELIB_USE_TBB)
            include_directories(${TBB_INCLUDE_DIRS})
        endif ()
    endif ()

    target_link_libraries(${PROJECT_NAME} 3rdparty_vtk)

    target_compile_definitions(${PROJECT_NAME} PRIVATE ECV_VTK_ENGINE_LIBRARY_BUILD)
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_VTK_BACKEND)

    set_target_properties(${PROJECT_NAME} PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            )

    # INTERNAL LIB
    target_link_libraries(${PROJECT_NAME}
            CV_DB_LIB
            CV_IO_LIB)

    # Qt
    # Qt5/Qt6 compatibility - use unified Qt:: prefix
    # QVTKOpenGLNativeWidget inherits from QOpenGLWidget, which requires OpenGLWidgets in Qt6
    target_link_libraries(${PROJECT_NAME}
            Qt::Core
            Qt::Gui
            Qt::Widgets
            Qt::PrintSupport
            Qt::OpenGL)
    if(USE_QT6)
        target_link_libraries(${PROJECT_NAME} Qt::OpenGLWidgets)
    endif()

    if (NOT APPLE) # no need to copy targets on macos due to rpath
        # install (shared) lib to specified destinations
        InstallSharedLibrary(TARGET ${PROJECT_NAME})
    endif ()

    #import VTK dlls (if any, WIN32 only)
    include(cmake/ExportVTKDlls.cmake)
    export_VTK_dlls(${ACloudViewer_DEST_FOLDER})

endif ()
