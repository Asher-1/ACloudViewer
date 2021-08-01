set(EXAMPLE_BIN_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")
macro(EXAMPLE_CPP EXAMPLE_CPP_NAME)
    include_directories(${OPENGL_ENGINE_LIB_SOURCE_DIR})
    if (MSVC)
        include_directories(${OPENGL_ENGINE_LIB_SOURCE_DIR}/msvc)
    endif ()

    add_executable(${EXAMPLE_CPP_NAME} "${EXAMPLE_CPP_NAME}.cpp")
    cloudViewer_show_and_abort_on_warning(${EXAMPLE_CPP_NAME})
    cloudViewer_set_global_properties(${EXAMPLE_CPP_NAME})

    set(DEPENDENCIES "${ARGN}")
    foreach (DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${EXAMPLE_CPP_NAME} ${DEPENDENCY})
    endforeach ()
	
	target_link_libraries(${EXAMPLE_CPP_NAME}
			${INTERNAL_EIGEN3_TARGET}
			)

    target_link_libraries(${EXAMPLE_CPP_NAME}
            ECV_DB_LIB
            ECV_IO_LIB
            OPENGL_ENGINE_LIB)

    set_target_properties(${EXAMPLE_CPP_NAME} PROPERTIES
            FOLDER "examples/cpp/"
            RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_BIN_DIR}"
            )

    if (NOT BUILD_EXAMPLES)
        set_target_properties(${EXAMPLE_CPP_NAME} PROPERTIES
                EXCLUDE_FROM_ALL TRUE
                )
    endif ()

    #InstallSharedLibrary( TARGET ${EXAMPLE_CPP_NAME} )
endmacro(EXAMPLE_CPP)

EXAMPLE_CPP(CameraPoseTrajectory)
EXAMPLE_CPP(ColorMapOptimization)
EXAMPLE_CPP(DepthCapture)
EXAMPLE_CPP(EvaluatePCDMatch)
EXAMPLE_CPP(FileDialog 3rdparty_tinyfiledialogs)
EXAMPLE_CPP(FileSystem)
EXAMPLE_CPP(Flann)
EXAMPLE_CPP(Image)
EXAMPLE_CPP(IntegrateRGBD)
EXAMPLE_CPP(ISSKeypoints)
EXAMPLE_CPP(LineSet)
EXAMPLE_CPP(Polylines)
EXAMPLE_CPP(Primitives)
EXAMPLE_CPP(Facets)
EXAMPLE_CPP(Log)
EXAMPLE_CPP(Octree)
EXAMPLE_CPP(OdometryRGBD)
if (TARGET 3rdparty_openmp)
    EXAMPLE_CPP(OpenMP 3rdparty_openmp)
else ()
    EXAMPLE_CPP(OpenMP)
endif ()
EXAMPLE_CPP(PCDFileFormat)
EXAMPLE_CPP(PointCloud)
EXAMPLE_CPP(PoseGraph)
EXAMPLE_CPP(ProgramOptions)
EXAMPLE_CPP(RegistrationColoredICP)
EXAMPLE_CPP(RegistrationRANSAC)
EXAMPLE_CPP(RGBDOdometry)
EXAMPLE_CPP(SLAC)
EXAMPLE_CPP(SLACIntegrate)
EXAMPLE_CPP(TriangleMesh)
EXAMPLE_CPP(TIntegrateRGBD)
EXAMPLE_CPP(TOdometryRGBD)
EXAMPLE_CPP(HalfEdgeTriangleMesh)
if (TARGET 3rdparty_openmp)
    EXAMPLE_CPP(TrimMeshBasedOnPointCloud 3rdparty_openmp)
else ()
    EXAMPLE_CPP(TrimMeshBasedOnPointCloud)
endif ()
EXAMPLE_CPP(ViewDistances)
EXAMPLE_CPP(ViewPCDMatch)
EXAMPLE_CPP(Visualizer)
EXAMPLE_CPP(VoxelHashing)
EXAMPLE_CPP(Voxelization)

if (BUILD_GUI)
    EXAMPLE_CPP(Draw)
    EXAMPLE_CPP(OffscreenRendering)
    EXAMPLE_CPP(MultipleWindows 3rdparty_threads)
    EXAMPLE_CPP(TICPOdometry 3rdparty_threads)
    EXAMPLE_CPP(TICPReconstruction 3rdparty_threads)
    EXAMPLE_CPP(VoxelHashingGUI 3rdparty_threads)
endif ()

if (BUILD_WEBRTC)
    EXAMPLE_CPP(DrawWebRTC)
endif ()

if (BUILD_LIBREALSENSE)
    if (TARGET 3rdparty_openmp)
        EXAMPLE_CPP(RealSenseBagReader 3rdparty_librealsense 3rdparty_jsoncpp 3rdparty_openmp)
    else ()
        EXAMPLE_CPP(RealSenseBagReader 3rdparty_librealsense 3rdparty_jsoncpp)
    endif ()
    EXAMPLE_CPP(RealSenseRecorder 3rdparty_librealsense)
endif ()

if (BUILD_AZURE_KINECT)
    EXAMPLE_CPP(AzureKinectRecord 3rdparty_k4a)
    EXAMPLE_CPP(AzureKinectMKVReader 3rdparty_jsoncpp)
    EXAMPLE_CPP(AzureKinectViewer 3rdparty_k4a)
endif ()

include(ProcessorCount)
ProcessorCount(NPROC)

# build-examples-iteratively is used to conserve space on CI machine.
add_custom_target(build-examples-iteratively
        COMMAND ${CMAKE_COMMAND}
        -DEXAMPLE_TARGETS="${EXAMPLE_TARGETS}"
        -DCMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
        -DEXAMPLE_BIN_DIR="${EXAMPLE_BIN_DIR}"
        -DCMAKE_BUILD_TYPE="$<CONFIG>"
        -DNPROC="${NPROC}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/iterative_build_examples.cmake
        )