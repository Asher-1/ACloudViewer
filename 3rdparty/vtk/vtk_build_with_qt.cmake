include(ExternalProject)

if(WIN32)
    set(VTK_LIB_SUFFIX $<$<CONFIG:Debug>:d>)
else()
    set(VTK_LIB_SUFFIX "")
endif()

set(VTK_VERSION 9.3)

set(VTK_LIBRARIES
    vtkfmt-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkloguru-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkkissfft-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonCore-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonMath-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonMisc-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonSystem-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonDataModel-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonTransforms-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonExecutionModel-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonComputationalGeometry-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkParallelCore-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersCore-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersVerdict-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersFlowPaths-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersHybrid-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersGeneral-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersSources-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersModeling-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersExtraction-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersGeometry-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkpugixml-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkIOExport-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkIOXML-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkIOXMLParser-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtksys-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonColor-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkChartsCore-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkViewsQt-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkViewsCore-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkViewsInfovis-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkViewsContext2D-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkImagingCore-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkImagingSources-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkGUISupportQt-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkInteractionImage-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkInteractionStyle-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkInteractionWidgets-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkIOCore-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkIOChemistry-${VTK_VERSION}${VTK_LIB_SUFFIX} # dependent by vtkPDBReader
    vtkIOXML-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkIOPLY-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkIOImage-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkIOLSDyna-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkIOLegacy-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkIOGeometry-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingQt-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingLabel-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingLOD-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingVtkJS-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingCore-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingOpenGL2-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingGL2PSOpenGL2-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingFreeType-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingContext2D-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingAnnotation-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingHyperTreeGrid-${VTK_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingContextOpenGL2-${VTK_VERSION}${VTK_LIB_SUFFIX}
)


if(BUILD_VTK_FROM_SOURCE)

    foreach(item IN LISTS VTK_LIBRARIES)
        list(APPEND VTK_BUILD_BYPRODUCTS <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${item}${CMAKE_STATIC_LIBRARY_SUFFIX})
    endforeach()

    if (BUILD_WITH_CONDA)
        if (WIN32)
            SET(CONDA_LIB_DIR ${CONDA_PREFIX}/Library)
        else ()
            SET(CONDA_LIB_DIR ${CONDA_PREFIX}/lib)
        endif()
    endif()

    ExternalProject_Add(
        ext_vtk
        PREFIX vtk
        URL https://www.vtk.org/files/release/${VTK_VERSION}/VTK-${VTK_VERSION}.1.tar.gz
        URL_HASH SHA256=8354ec084ea0d2dc3d23dbe4243823c4bfc270382d0ce8d658939fd50061cab8
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/vtk"
        # do not update
        UPDATE_COMMAND ""
        CMAKE_ARGS
            ${ExternalProject_CMAKE_ARGS_hidden}
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_PREFIX_PATH=${CONDA_LIB_DIR}
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DVTK_BUILD_TESTING=OFF
            -DVTK_BUILD_EXAMPLES=OFF
            -DVTK_SMP_ENABLE_STDTHREAD=OFF # fix error LNK2019
            -DVTK_QT_VERSION:STRING=5
            -DVTK_GROUP_ENABLE_Qt=YES
            -DVTK_MODULE_ENABLE_VTK_IOPLY=YES
            -DVTK_GROUP_ENABLE_StandAlone=YES
            -DVTK_MODULE_ENABLE_VTK_ViewsQt=YES
            -DVTK_MODULE_ENABLE_VTK_ViewsCore=YES
            -DVTK_MODULE_ENABLE_VTK_RenderingQt=YES
            -DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES
            -DVTK_MODULE_ENABLE_VTK_GUISupportQtQuick=NO
            -DVTK_MODULE_ENABLE_VTK_GUISupportQtSQL=NO
        BUILD_BYPRODUCTS
            ${VTK_BUILD_BYPRODUCTS}
    )

    ExternalProject_Get_Property(ext_vtk INSTALL_DIR)
    set(VTK_LIB_DIR ${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR})
    set(VTK_INCLUDE_DIRS "${INSTALL_DIR}/include/vtk-${VTK_VERSION}/")

else() #### download prebuilt vtk

    foreach(item IN LISTS VTK_LIBRARIES)
        list(APPEND VTK_BUILD_BYPRODUCTS <SOURCE_DIR>/lib/${item}${CMAKE_STATIC_LIBRARY_SUFFIX})
    endforeach()

    if(LINUX_AARCH64)
        message(FATAL "No precompiled vtk for platform. Enable BUILD_VTK_FROM_SOURCE")
    elseif(APPLE_AARCH64)
        message(FATAL "No precompiled vtk for platform. Enable BUILD_VTK_FROM_SOURCE")
    elseif(APPLE)
        set(VTK_URL
            https://github.com/isl-org/open3d_downloads/releases/download/vtk/vtk_${VTK_VERSION}_macos_10.15.tar.gz
        )
        set(VTK_SHA256 a93579d1f135abb9e0ebfe774f46c22cfd7e88ee61cb1aba16ef83a5402ed918)
    elseif(UNIX)
        set(VTK_URL
            https://github.com/isl-org/open3d_downloads/releases/download/vtk/vtk_${VTK_VERSION}_linux_x86_64.tar.gz
        )
        set(VTK_SHA256 28e36654ed18aa9f668a0486a6c3d26a0ca6cf6a593dbd15be4736b40880a82b)
    elseif(WIN32)
        if (STATIC_WINDOWS_RUNTIME)
            set(VTK_URL
                https://github.com/isl-org/open3d_downloads/releases/download/vtk/vtk_${VTK_VERSION}_win_staticrt.tar.gz
            )
            set(VTK_SHA256 4a6e2d00652dc86a3bea774060eaad643424800fe2cf369d325ebaaa49693c95)
        else()
            set(VTK_URL
                https://github.com/isl-org/open3d_downloads/releases/download/vtk/vtk_${VTK_VERSION}_win.tar.gz
            )
            set(VTK_SHA256 6ee09115d23ec18d6d01d1e4c89fa236ec69406d8ba8cc1b8ec37c4123b93caa)
        endif()
    else()
        message(FATAL "Unsupported platform")
    endif()

    ExternalProject_Add(
        ext_vtk
        PREFIX vtk
        URL ${VTK_URL}
        URL_HASH SHA256=${VTK_SHA256}
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/vtk"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS
            ${VTK_BUILD_BYPRODUCTS}
    )

    ExternalProject_Get_Property(ext_vtk SOURCE_DIR)
    set(VTK_LIB_DIR "${SOURCE_DIR}/lib")
    set(VTK_INCLUDE_DIRS "${SOURCE_DIR}/include/vtk-${VTK_VERSION}/")

endif() # BUILD_VTK_FROM_SOURCE