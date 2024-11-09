include(ExternalProject)

# OS specific settings
if(WIN32)
    # Windows defaults to hidden symbol visibility, override that
    # TODO: It would be better to explictly export symbols.
    #       Then, we could use -fvisibility=hidden for Linux as well
    # SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if(MSVC)
        # Make sure we don't hit the 65535 object member limit with MSVC
        #
        # /bigobj allows object files with more than 65535 members
        # /Ob2 enables function inlining, because MSVC is particularly
        # verbose with inline members
        #
        # See: https://github.com/tensorflow/tensorflow/pull/10962
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj;/Ob2>")
    endif()
    if (STATIC_WINDOWS_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()

if (APPLE)
    set (CMAKE_OSX_DEPLOYMENT_TARGET "12.6" CACHE STRING
        "Minimum OS X deployment version" FORCE)
endif()

# copy of var definitions from find_dependencies.cmake

# CMake arguments for configuring ExternalProjects. Use the second _hidden
# version by default.
set(ExternalProject_CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_CUDA_COMPILER_LAUNCHER=${CMAKE_CUDA_COMPILER_LAUNCHER}
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
    # Always build 3rd party code in Release mode. Ignored by multi-config
    # generators (XCode, MSVC). MSVC needs matching config anyway.
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POLICY_DEFAULT_CMP0091:STRING=NEW
    -DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)
# Keep 3rd party symbols hidden from Open3D user code. Do not use if 3rd party
# libraries throw exceptions that escape Open3D.
set(ExternalProject_CMAKE_ARGS_hidden
    ${ExternalProject_CMAKE_ARGS}
    # Apply LANG_VISIBILITY_PRESET to static libraries and archives as well
    -DCMAKE_POLICY_DEFAULT_CMP0063:STRING=NEW
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_CUDA_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
)

if(WIN32)
    set(VTK_LIB_SUFFIX $<$<CONFIG:Debug>:d>)
else()
    set(VTK_LIB_SUFFIX "")
endif()

set(VTK_VERSION 9.3.1)
set(VTK_MAJOR_VERSION 9.3)

set(VTK_LIBRARIES
    vtkfmt-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkloguru-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkkissfft-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonCore-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonMath-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonMisc-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonSystem-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonDataModel-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonTransforms-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonExecutionModel-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonComputationalGeometry-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkParallelCore-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersCore-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersVerdict-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersFlowPaths-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersHybrid-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersGeneral-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersSources-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersModeling-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersExtraction-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkFiltersGeometry-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkpugixml-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkIOExport-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkIOXML-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkIOXMLParser-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtksys-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkCommonColor-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkChartsCore-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkViewsQt-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkViewsCore-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkViewsInfovis-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkViewsContext2D-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkImagingCore-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkImagingSources-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkGUISupportQt-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkInteractionImage-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkInteractionStyle-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkInteractionWidgets-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkIOCore-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkIOChemistry-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX} # dependent by vtkPDBReader
    vtkIOXML-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkIOPLY-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkIOImage-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkIOLSDyna-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkIOLegacy-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkIOGeometry-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingQt-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingLabel-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingLOD-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingVtkJS-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingCore-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingOpenGL2-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingGL2PSOpenGL2-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingFreeType-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingContext2D-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingAnnotation-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingHyperTreeGrid-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
    vtkRenderingContextOpenGL2-${VTK_MAJOR_VERSION}${VTK_LIB_SUFFIX}
)

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
    URL https://www.vtk.org/files/release/${VTK_MAJOR_VERSION}/VTK-${VTK_VERSION}.tar.gz
    URL_HASH SHA256=8354ec084ea0d2dc3d23dbe4243823c4bfc270382d0ce8d658939fd50061cab8
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/vtk"
    # do not update
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE OFF
    BUILD_ALWAYS 0
    INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
    CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS_hidden}
        -DBUILD_SHARED_LIBS=ON
        -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
        # Syncing GLIBCXX_USE_CXX11_ABI for MSVC causes problems, but directly
        # checking CXX_COMPILER_ID is not supported.
        $<IF:$<PLATFORM_ID:Windows>,"",-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI}>
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
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
        -DCMAKE_PREFIX_PATH=${CONDA_LIB_DIR}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    BUILD_BYPRODUCTS
        ${VTK_BUILD_BYPRODUCTS}
)

ExternalProject_Get_Property(ext_vtk INSTALL_DIR)
set(VTK_INCLUDE_DIRS "${INSTALL_DIR}/include/vtk-${VTK_MAJOR_VERSION}/")
set(VTK_LIBRARIES_DIRS ${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR})
set(VTK_DIR ${INSTALL_DIR}/lib/cmake/vtk-${VTK_MAJOR_VERSION})
set(VTK_CMAKE_FLAGS -DVTK_DIR=${INSTALL_DIR}/lib/cmake/vtk-${VTK_MAJOR_VERSION})
set(VTK_BINARY_DIR ${INSTALL_DIR}/bin)
if (WIN32)
    copy_shared_library(ext_vtk
                        LIB_DIR   ${VTK_BINARY_DIR}
                        LIBRARIES ${VTK_LIBRARIES})
endif()