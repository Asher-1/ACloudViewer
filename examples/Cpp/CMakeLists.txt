set(EXAMPLE_BIN_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")
macro(cloudViewer_add_example EXAMPLE_CPP_NAME)
    include_directories(${OPENGL_ENGINE_LIB_SOURCE_DIR})

    add_executable(${EXAMPLE_CPP_NAME} "${EXAMPLE_CPP_NAME}.cpp")
	
	# Don't use cloudViewer_link_3rdparty_libraries(${EXAMPLE_CPP_NAME}).
    # Unlike benchmarks, examples are intended as standalone apps that link
    # the main CloudViewer library, while the benchmarks executable needs 3rd-party
    # libraries since it measures the internal componets of CloudViewer.
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
	
	list(APPEND EXAMPLE_TARGETS ${EXAMPLE_CPP_NAME})

endmacro(cloudViewer_add_example)

cloudViewer_add_example(CameraPoseTrajectory)
cloudViewer_add_example(ColorMapOptimization)
cloudViewer_add_example(DepthCapture)
cloudViewer_add_example(EvaluatePCDMatch)
cloudViewer_add_example(FileDialog 3rdparty_tinyfiledialogs)
cloudViewer_add_example(FileSystem)
cloudViewer_add_example(Flann)
cloudViewer_add_example(Image)
cloudViewer_add_example(IntegrateRGBD)
cloudViewer_add_example(ISSKeypoints)
cloudViewer_add_example(LineSet)
cloudViewer_add_example(Polylines)
cloudViewer_add_example(Primitives)
cloudViewer_add_example(Facets)
cloudViewer_add_example(Log)
cloudViewer_add_example(Octree)
cloudViewer_add_example(OdometryRGBD)
if (TARGET 3rdparty_openmp)
    cloudViewer_add_example(OpenMP 3rdparty_openmp)
else ()
    cloudViewer_add_example(OpenMP)
endif ()
cloudViewer_add_example(PCDFileFormat)
cloudViewer_add_example(PointCloud)
cloudViewer_add_example(PoseGraph)
cloudViewer_add_example(ProgramOptions)
cloudViewer_add_example(GeneralizedICP)
cloudViewer_add_example(RegistrationColoredICP)
cloudViewer_add_example(RegistrationRANSAC)
cloudViewer_add_example(RGBDOdometry)
cloudViewer_add_example(SLAC)
cloudViewer_add_example(SLACIntegrate)
cloudViewer_add_example(TriangleMesh)
cloudViewer_add_example(TIntegrateRGBD)
cloudViewer_add_example(TOdometryRGBD)
cloudViewer_add_example(HalfEdgeTriangleMesh)
if (TARGET 3rdparty_openmp)
    cloudViewer_add_example(TrimMeshBasedOnPointCloud 3rdparty_openmp)
else ()
    cloudViewer_add_example(TrimMeshBasedOnPointCloud)
endif ()
cloudViewer_add_example(ViewDistances)
cloudViewer_add_example(ViewPCDMatch)
cloudViewer_add_example(Visualizer)
cloudViewer_add_example(VoxelHashing)
cloudViewer_add_example(Voxelization)

if (BUILD_GUI)
    cloudViewer_add_example(Draw)
    cloudViewer_add_example(OffscreenRendering)
    cloudViewer_add_example(MultipleWindows 3rdparty_threads)
    cloudViewer_add_example(TICPOdometry 3rdparty_threads)
    cloudViewer_add_example(TICPReconstruction 3rdparty_threads)
    cloudViewer_add_example(VoxelHashingGUI 3rdparty_threads)
endif ()

if (BUILD_WEBRTC)
    cloudViewer_add_example(DrawWebRTC)
endif ()

if (BUILD_LIBREALSENSE)
    if (TARGET 3rdparty_openmp)
        cloudViewer_add_example(RealSenseBagReader 3rdparty_librealsense 3rdparty_jsoncpp 3rdparty_openmp)
    else ()
        cloudViewer_add_example(RealSenseBagReader 3rdparty_librealsense 3rdparty_jsoncpp)
    endif ()
    cloudViewer_add_example(RealSenseRecorder 3rdparty_librealsense)
endif ()

if (BUILD_AZURE_KINECT)
    cloudViewer_add_example(AzureKinectRecord 3rdparty_k4a)
    cloudViewer_add_example(AzureKinectMKVReader 3rdparty_jsoncpp)
    cloudViewer_add_example(AzureKinectViewer 3rdparty_k4a)
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