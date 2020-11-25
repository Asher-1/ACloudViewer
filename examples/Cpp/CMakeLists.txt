include(ProcessorCount)
ProcessorCount(NPROC)

set(EXAMPLE_BIN_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")
macro(EXAMPLE_CPP EXAMPLE_CPP_NAME)
    include_directories( ${OPENGL_ENGINE_LIB_SOURCE_DIR} )
    if( MSVC )
       include_directories( ${OPENGL_ENGINE_LIB_SOURCE_DIR}/msvc )
    endif()
	
    add_executable(${EXAMPLE_CPP_NAME} "${EXAMPLE_CPP_NAME}.cpp")
    ShowAndAbortOnWarning(${EXAMPLE_CPP_NAME})
    cloudViewer_set_global_properties(${EXAMPLE_CPP_NAME})

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${EXAMPLE_CPP_NAME} ${DEPENDENCY})
    endforeach()
	
    target_link_libraries( ${EXAMPLE_CPP_NAME}
                            ECV_DB_LIB
                            ECV_IO_LIB
                            OPENGL_ENGINE_LIB)

    set_target_properties(${EXAMPLE_CPP_NAME} PROPERTIES
        FOLDER "examples/cpp/"
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_BIN_DIR}"
    )
	
    if(NOT BUILD_EXAMPLES)
        set_target_properties(${EXAMPLE_CPP_NAME} PROPERTIES
            EXCLUDE_FROM_ALL TRUE
        )
    endif()

    #InstallSharedLibrary( TARGET ${EXAMPLE_CPP_NAME} )
endmacro(EXAMPLE_CPP)

EXAMPLE_CPP(CameraPoseTrajectory                                    )
EXAMPLE_CPP(ColorMapOptimization                                    )
EXAMPLE_CPP(DepthCapture                                            )
EXAMPLE_CPP(EvaluateFeatureMatch    ${OPENMP_TARGET} ${FLANN_TARGET})
EXAMPLE_CPP(EvaluatePCDMatch                                        )
EXAMPLE_CPP(FileDialog                     ${TINYFILEDIALOGS_TARGET})
EXAMPLE_CPP(FileSystem                                              )
EXAMPLE_CPP(Flann                                    ${FLANN_TARGET})
EXAMPLE_CPP(Image                                                   )
EXAMPLE_CPP(IntegrateRGBD                                           )
EXAMPLE_CPP(ISSKeypoints                                            )
EXAMPLE_CPP(LineSet                                                 )
EXAMPLE_CPP(Polylines                                               )
EXAMPLE_CPP(Primitives                                              )
EXAMPLE_CPP(Facets                                                  )
EXAMPLE_CPP(Log                                                     )
EXAMPLE_CPP(OdometryRGBD                                            )
EXAMPLE_CPP(PCDFileFormat                                           )
EXAMPLE_CPP(PointCloud                                              )
EXAMPLE_CPP(PoseGraph                                               )
EXAMPLE_CPP(ProgramOptions                                          )
EXAMPLE_CPP(RegistrationRANSAC                                      )
EXAMPLE_CPP(RGBDOdometry                                            )
EXAMPLE_CPP(TriangleMesh                                            )
EXAMPLE_CPP(TrimMeshBasedOnPointCloud               ${OPENMP_TARGET})
EXAMPLE_CPP(ViewDistances                                           )
EXAMPLE_CPP(ViewPCDMatch                                            )
EXAMPLE_CPP(Visualizer                                              )
EXAMPLE_CPP(Voxelization                                            )

if (WITH_OPENMP)
    EXAMPLE_CPP(OpenMP 					${OPENMP_TARGET})
endif ()

if (BUILD_LIBREALSENSE)
    EXAMPLE_CPP(RealSense 				${LIBREALSENSE_TARGET})
endif ()

if(BUILD_AZURE_KINECT)
    EXAMPLE_CPP(AzureKinectRecord     	${K4A_TARGET})
    EXAMPLE_CPP(AzureKinectMKVReader  	${JSONCPP_TARGET})
    EXAMPLE_CPP(AzureKinectViewer     	${K4A_TARGET})
endif()

# build-examples-iteratively is used to conserve space on CI machine.
add_custom_target(build-examples-iteratively
    COMMAND ${CMAKE_COMMAND}
    -DEXAMPLE_TARGETS="${EXAMPLE_TARGETS}"
    -DCMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
    -DEXAMPLE_BIN_DIR="${EXAMPLE_BIN_DIR}"
    -DCMAKE_BUILD_TYPE="$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>$<$<CONFIG:RelWithDebInfo>:RelWithDebInfo>$<$<CONFIG:MinSizeRel>:MinSizeRel>"
    -DNPROC="${NPROC}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/iterative_build_examples.cmake
)
