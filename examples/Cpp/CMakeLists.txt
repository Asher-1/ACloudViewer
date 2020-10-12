macro(EXAMPLE_CPP EXAMPLE_CPP_NAME)
	include_directories( ${OPENGL_ENGINE_LIB_SOURCE_DIR} )
	if( MSVC )
	   include_directories( ${OPENGL_ENGINE_LIB_SOURCE_DIR}/msvc )
	endif()
	
	file( GLOB header_list *.h  )
	
    add_executable(${EXAMPLE_CPP_NAME} "${EXAMPLE_CPP_NAME}.cpp" ${header_list})
    ShowAndAbortOnWarning(${EXAMPLE_CPP_NAME})

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${EXAMPLE_CPP_NAME} ${DEPENDENCY})
    endforeach()
	
	target_link_libraries( ${EXAMPLE_CPP_NAME} 
							ECV_DB_LIB
							ECV_IO_LIB
							OPENGL_ENGINE_LIB)

    set_target_properties(${EXAMPLE_CPP_NAME} PROPERTIES
            FOLDER "examples/Cpp/"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")
endmacro(EXAMPLE_CPP)

EXAMPLE_CPP(CameraPoseTrajectory)
EXAMPLE_CPP(ColorMapOptimization)
EXAMPLE_CPP(DepthCapture)
EXAMPLE_CPP(EvaluateFeatureMatch)
EXAMPLE_CPP(EvaluatePCDMatch)
EXAMPLE_CPP(FileDialog  tinyfiledialogs)
EXAMPLE_CPP(FileSystem)
EXAMPLE_CPP(Flann)
EXAMPLE_CPP(Image)
EXAMPLE_CPP(IntegrateRGBD)
EXAMPLE_CPP(LineSet      )
EXAMPLE_CPP(Polylines      )
EXAMPLE_CPP(Primitives      )
EXAMPLE_CPP(Facets      )
EXAMPLE_CPP(Log          )
EXAMPLE_CPP(OdometryRGBD)
if (WITH_OPENMP)
    EXAMPLE_CPP(OpenMP)
endif (WITH_OPENMP)
EXAMPLE_CPP(PCDFileFormat)
EXAMPLE_CPP(PointCloud)
EXAMPLE_CPP(PoseGraph)
EXAMPLE_CPP(ProgramOptions)
if (BUILD_LIBREALSENSE)
    EXAMPLE_CPP(RealSense realsense)
endif (BUILD_LIBREALSENSE)

if (BUILD_AZURE_KINECT)
    EXAMPLE_CPP(AzureKinectRecord )
    EXAMPLE_CPP(AzureKinectMKVReader )
    EXAMPLE_CPP(AzureKinectViewer)
endif (BUILD_AZURE_KINECT)

EXAMPLE_CPP(RegistrationRANSAC)
EXAMPLE_CPP(RGBDOdometry             )
EXAMPLE_CPP(TriangleMesh             )
EXAMPLE_CPP(TrimMeshBasedOnPointCloud)
EXAMPLE_CPP(ViewDistances            )
EXAMPLE_CPP(ViewPCDMatch             )
EXAMPLE_CPP(Visualizer  )
EXAMPLE_CPP(Voxelization)
