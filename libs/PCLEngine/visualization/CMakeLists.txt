set(SUBSYS_NAME visualization)
set(SUBSYS_DESC "Point cloud visualization library")
set(SUBSYS_DEPS common io kdtree geometry search octree)

if(NOT VTK_FOUND)
    set(DEFAULT FALSE)
    set(REASON "VTK was not found.")
else()
    set(DEFAULT TRUE)
    set(REASON)
    if(${VTK_VERSION} VERSION_LESS 9.0)
      set(VTK_USE_FILE "${VTK_USE_FILE}" CACHE INTERNAL "VTK_USE_FILE")
      include("${VTK_USE_FILE}")
    endif()
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
endif(NOT VTK_FOUND)

if (ANDROID)
  set (build FALSE)
  message("VTK was found, but cannot be compiled for Android. Please use VES instead.")
endif ()

if (OPENGL_FOUND)
  if(OPENGL_INCLUDE_DIR)
    include_directories("${OPENGL_INCLUDE_DIR}")
  endif()
  if(OPENGL_DEFINITIONS)
    add_definitions("${OPENGL_DEFINITIONS}")
  endif()
endif()

if(build)
    set(srcs
        src/pcl_visualizer.cpp
        src/image_viewer.cpp
        src/vtk/pcl_image_canvas_source_2d.cpp
        src/vtk/pcl_context_item.cpp
        src/vtk/vtkRenderWindowInteractorFix.cpp
        )
    if("${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" VERSION_LESS "5.6")
        # A custom interactor is necessary on VTK 5.4 and below
        list(APPEND srcs src/interactor.cpp)
    endif("${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" VERSION_LESS "5.6")
    
    if("${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" VERSION_GREATER "5.6")
        # PCLPainter2D uses some functions not present in vtk 5.6
        # list(APPEND srcs
            # src/pcl_painter2D.cpp
            # src/pcl_plotter.cpp
            # )
    endif("${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" VERSION_GREATER "5.6")

    set(incs 
        "include/pcl/${SUBSYS_NAME}/image_viewer.h"
        )
    if("${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" VERSION_GREATER "5.6")
        # list(APPEND incs
            # "include/pcl/${SUBSYS_NAME}/pcl_plotter.h"
            # )
    endif("${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" VERSION_GREATER "5.6")

    set(impl_incs 
        "include/pcl/${SUBSYS_NAME}/impl/image_viewer.hpp"
        )
    if("${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" VERSION_GREATER "5.6")
        # list(APPEND impl_incs
            # "include/pcl/${SUBSYS_NAME}/impl/pcl_plotter.hpp"
            # )
    endif("${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" VERSION_GREATER "5.6")

    set(vtk_incs 
        "include/pcl/${SUBSYS_NAME}/vtk/pcl_image_canvas_source_2d.h"
        "include/pcl/${SUBSYS_NAME}/vtk/pcl_context_item.h"
        "include/pcl/${SUBSYS_NAME}/vtk/vtkRenderWindowInteractorFix.h"
        )

    # on apple, a workaround is used for the cocoa render window interactor
    if(APPLE)
      list(APPEND srcs
        include/pcl/visualization/vtk/vtkRenderWindowInteractorFix.mm)
    endif()

    set(LIB_NAME "pcl_${SUBSYS_NAME}")
    PCL_ADD_LIBRARY("${LIB_NAME}" "${SUBSYS_NAME}" ${srcs} ${incs} ${impl_incs} ${vtk_incs})

    target_include_directories("${LIB_NAME}" PUBLIC ${VTK_INCLUDE_DIRS})

    # apple workaround (continued)
    if(APPLE)
      target_link_libraries("${LIB_NAME}" "-framework Cocoa")
    endif()
     
    if (NOT USE_SYSTEM_VTK AND NOT USE_SYSTEM_PCL)
      target_link_libraries("${LIB_NAME}" 3rdparty_pcl 3rdparty_vtk)
    else()
      if("${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}" VERSION_LESS "5.6")
        target_link_libraries("${LIB_NAME}" pcl_common pcl_io pcl_kdtree vtkCommon vtkWidgets vtkHybrid vtkFiltering vtkRendering ${OPENGL_LIBRARIES})
      else()
        target_link_libraries("${LIB_NAME}" pcl_common pcl_io pcl_kdtree ${VTK_LIBRARIES} ${OPENGL_LIBRARIES})
      endif()
    endif()
  
    set(EXT_DEPS "")
    if(WITH_OPENNI)
      list(APPEND EXT_DEPS libopenni)
    endif()
    if(WITH_OPENNI2)
      list(APPEND EXT_DEPS libopenni2)
    endif()
    if(WITH_ENSENSO)
      list(APPEND EXT_DEPS ensenso)
    endif()
    if(WITH_DAVIDSDK)
      list(APPEND EXT_DEPS davidSDK)
    endif()
    if(WITH_DSSDK)
      list(APPEND EXT_DEPS dssdk)
    endif()
    if(WITH_RSSDK)
      list(APPEND EXT_DEPS rssdk)
    endif()
    PCL_MAKE_PKGCONFIG("${LIB_NAME}" "${SUBSYS_NAME}" "${SUBSYS_DESC}"
      "${SUBSYS_DEPS}" "${EXT_DEPS}" "" "" "")

    # Install include files
    PCL_ADD_INCLUDES("${SUBSYS_NAME}" "${SUBSYS_NAME}" ${incs})
    PCL_ADD_INCLUDES("${SUBSYS_NAME}" "${SUBSYS_NAME}/impl" ${impl_incs})
    PCL_ADD_INCLUDES("${SUBSYS_NAME}" "${SUBSYS_NAME}/vtk" ${vtk_incs})
	
endif(build)
