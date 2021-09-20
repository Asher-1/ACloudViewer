project(ErowCloudViewer VERSION ${PROJECT_VERSION})

OPTION( UPDATE_TRANSLATION_FILES "Generate or update ts files" OFF )

include( CMakePolicies NO_POLICY_SCOPE )

if (MSVC)
    if (CVCORELIB_USE_TBB)
        include_directories( ${TBB_INCLUDE_DIRS})
    endif()
endif()

if (BUILD_PYTHON_MODULE)
    include_directories( ${ECV_PYTHON_LIB_SOURCE_DIR} )
endif()

include_directories( ${ErowCloudViewerPlugins_SOURCE_DIR} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/db_tree )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/ui_templates )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

file( GLOB header_list *.h  )
file( GLOB source_list *.cpp )

file( GLOB ui_list ui_templates/*.ui )

file( GLOB qrc_list *.qrc Resources/themes/*.qrc)

file( GLOB txt_list TODO.txt ${ErowCloudViewerProjects_SOURCE_DIR}/CHANGELOG.txt)

add_subdirectory( db_tree )
add_subdirectory( pluginManager )

# for
add_subdirectory( QSimpleUpdater )

if (BUILD_RECONSTRUCTION)
    add_subdirectory(reconstruction)
endif()

add_subdirectory( extern/QCustomPlot EXCLUDE_FROM_ALL )

qt5_wrap_ui( generated_ui_list ${ui_list} )
qt5_add_resources( generated_qrc_list ${qrc_list} )

if( MSVC )
    # App icon with MSVC
    set( rc_list Resources/images/icon/ecv_icon.rc )

    #to get rid of the (system) console
    add_executable( ${PROJECT_NAME} WIN32 ${header_list} ${source_list} ${generated_ui_list} ${generated_qrc_list} ${rc_list} ${txt_list})

    # generate or update translation files(ts)
    if(UPDATE_TRANSLATION_FILES)
        file(GLOB_RECURSE SOURCE_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../*.h ${CMAKE_CURRENT_SOURCE_DIR}/../*.ui)
        find_package(Qt5Core REQUIRED)
        find_package(Qt5LinguistTools REQUIRED)
        set( TRANSLATION_FILES ${CMAKE_CURRENT_SOURCE_DIR}/translations/ErowCloudViewer_zh.ts)
        foreach( TRANSLATION_FILE ${TRANSLATION_FILES} )
            message("TRANSLATION_FILE : ${TRANSLATION_FILE}")
            qt5_update_translation(QMs ${PROJECT_NAME} ${SOURCE_LIST} ${TRANSLATION_FILE})
            message("QMs: ${QMs}")
        endforeach()
    endif()
elseif( APPLE )
    add_executable( ${PROJECT_NAME} MACOSX_BUNDLE ${header_list} ${source_list} ${generated_ui_list} ${generated_qrc_list} ${txt_list} )
else()
    add_executable( ${PROJECT_NAME} ${header_list} ${source_list} ${generated_ui_list} ${generated_qrc_list} ${rc_list} ${txt_list} )
endif()

if (WIN32)
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/ErowCloudViewer.exe ${CMAKE_BINARY_DIR}/eCV/$<CONFIG>/ErowCloudViewer.exe
            VERBATIM
    )
else()
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/bin/ErowCloudViewer ${CMAKE_BINARY_DIR}/eCV/$<CONFIG>/ErowCloudViewer
            VERBATIM
    )
endif()


DeployQt(
    TARGET
		${PROJECT_NAME}
    DEPLOY_PATH
		${EROWCLOUDVIEWER_DEST_FOLDER}
)

target_link_libraries( ${PROJECT_NAME}
    CVAppCommon
    QCustomPlot
    Qt5::Network
    Qt5::PrintSupport
)

if( CVCORELIB_USE_QT_CONCURRENT )
    # We want *deployqt to handle this dependency, so instead of just copying the file
    # link to the target.
    target_link_libraries( ${PROJECT_NAME}
                           Qt5::Concurrent
    )
endif()

# contrib. libraries support
if( APPLE )
    target_link_contrib( ${PROJECT_NAME} ${EROWCLOUDVIEWER_MAC_FRAMEWORK_DIR} )
elseif( UNIX )
    target_link_contrib( ${PROJECT_NAME} lib/erowcloudviewer )
else()
    target_link_contrib( ${PROJECT_NAME} ${EROWCLOUDVIEWER_DEST_FOLDER} )
endif()

if (BUILD_RECONSTRUCTION)
    if (${BUILD_GUI})
        target_compile_definitions( ${PROJECT_NAME} PRIVATE GUI_ENABLED)
    endif()
    if (${USE_SIMD})
        target_compile_definitions( ${PROJECT_NAME} PRIVATE SIMD_ENABLED)
    endif()
    if (${WITH_OPENMP})
        target_compile_definitions( ${PROJECT_NAME} PRIVATE OPENMP_ENABLED)
    endif()
    if (${BUILD_CUDA_MODULE})
        target_compile_definitions( ${PROJECT_NAME} PRIVATE CUDA_ENABLED)
    endif()
    if (${CVCORELIB_USE_CGAL})
        target_compile_definitions( ${PROJECT_NAME} PRIVATE CGAL_ENABLED)
    endif()
    if (${BUILD_UNIT_TESTS})
        target_compile_definitions( ${PROJECT_NAME} PRIVATE TESTS_ENABLED)
    endif()
    target_compile_definitions( ${PROJECT_NAME} PRIVATE OPENGL_ENABLED)

	if(MSVC)
        # Some fixes for the Glog library.
        add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
        add_definitions("-DGOOGLE_GLOG_DLL_DECL=")
		# fix gflags library bugs on windows
		add_definitions(-DGFLAGS_DLL_DECLARE_FLAG= -DGFLAGS_DLL_DEFINE_FLAG= -DGFLAGS_IS_A_DLL=0)
		target_link_libraries(${PROJECT_NAME} shlwapi.lib)
    endif()

    include_directories(${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/include)
    include_directories(${COLMAP_SOURCE_DIR}/lib)
    include_directories(${COLMAP_SOURCE_DIR}/src)
    
    target_link_libraries(${PROJECT_NAME} colmap)

    # Add custom preprocessor definitions
    target_compile_definitions( ${PROJECT_NAME} PRIVATE USE_COLMAP_MODULE )
endif()

# fix that You must build your code with position independent code if Qt was built with -reduce-relocations
cloudViewer_set_targets_independent(${PROJECT_NAME})

if (BUILD_PYTHON_MODULE)
    target_link_libraries( ${PROJECT_NAME} ECV_PYTHON_LIB )
    # Add custom preprocessor definitions
    target_compile_definitions( ${PROJECT_NAME} PRIVATE USE_PYTHON_MODULE )
endif()	

if (USE_PCL_BACKEND)
    target_link_directories( ${PROJECT_NAME}
            PUBLIC ${QPCL_ENGINE_LIB_SOURCE_DIR}
    )
    include_directories( ${QPCL_ENGINE_LIB_SOURCE_DIR})
    target_link_libraries( ${PROJECT_NAME} QPCL_ENGINE_LIB )
endif()

# Add custom preprocessor definitions
if( WIN32 )
    set_property( TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS)
    if (MSVC)
        set_target_properties( ${PROJECT_NAME} PROPERTIES LINK_FLAGS " /MANIFEST:NO" )
    endif()
endif()

# App icon with Code::Blocks/MinGW
if( WIN32 )
    if( MINGW )
        if (NOT CMAKE_RC_COMPILER)
            set( CMAKE_RC_COMPILER windres )
        endif()
        add_custom_command( TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_RC_COMPILER} -i ${CMAKE_CURRENT_SOURCE_DIR}/Resources/images/icon/ecv_icon.rc --input-format=rc -o ${CMAKE_CURRENT_BINARY_DIR}/ecv_icon.res -O coff )
    endif()
endif()

# install program
cloudViewer_install_ext( TARGETS ${PROJECT_NAME} ${EROWCLOUDVIEWER_DEST_FOLDER} "" )

# Auxiliary files
set( auxFiles  ${ErowCloudViewerProjects_SOURCE_DIR}/CHANGELOG.txt bin_other/license.txt bin_other/global_shift_list_template.txt )

if( WIN32 )
    # Additional auxiliary file(s)
    file( GLOB batch_list bin_other/*.bat )
    list( APPEND auxFiles ${batch_list} )
endif()

# Install auxiliary files
if( WIN32 OR APPLE )
    foreach( filename ${auxFiles} )
        cloudViewer_install_ext( FILES ${filename} ${EROWCLOUDVIEWER_DEST_FOLDER} "" )
    endforeach()
else( UNIX )
    foreach( filename ${auxFiles} )
        cloudViewer_install_ext( FILES ${filename} share/erowcloudviewer "" )
    endforeach()
endif()

# Translation
add_subdirectory( translations )

# Put together our Application bundle on macOS
if( APPLE )    
    add_subdirectory( Mac )
endif()

# Install plugins and shaders in the correct folder for each platform
if( APPLE )	
    InstallPlugins(
            DEST_PATH "${EROWCLOUDVIEWER_MAC_PLUGIN_DIR}"
            SHADER_DEST_PATH "${EROWCLOUDVIEWER_MAC_BASE_DIR}/Contents/Shaders"
    )
elseif( UNIX )
    InstallPlugins(
            DEST_PATH "${CMAKE_INSTALL_LIBDIR}/erowcloudviewer/plugins"
            SHADER_DEST_PATH share/erowcloudviewer/shaders
    )
else()
    InstallPlugins(
            DEST_PATH "${EROWCLOUDVIEWER_DEST_FOLDER}"
            DEST_FOLDER "plugins"
            SHADER_DEST_PATH "${EROWCLOUDVIEWER_DEST_FOLDER}"
            SHADER_DEST_FOLDER "shaders"
    )
endif()
