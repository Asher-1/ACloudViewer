project(${MAIN_APP_NAME} VERSION ${PROJECT_VERSION})

OPTION(UPDATE_TRANSLATION_FILES "Generate or update ts files" OFF)

include(CMakePolicies NO_POLICY_SCOPE)

if (MSVC)
    if (CVCORELIB_USE_TBB)
        include_directories(${TBB_INCLUDE_DIRS})
    endif ()
endif ()

if (BUILD_PYTHON_MODULE)
    include_directories(${ECV_PYTHON_LIB_SOURCE_DIR})
endif ()

include_directories(${ACloudViewerPlugins_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/db_tree)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ui_templates)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB header_list *.h)
file(GLOB source_list *.cpp)

file(GLOB ui_list ui_templates/*.ui)

file(GLOB qrc_list *.qrc Resources/themes/*.qrc)

file(GLOB txt_list TODO.txt ${COLOUDVIEWER_PROJECT_PATH}/CHANGELOG.txt)

add_subdirectory(db_tree)
add_subdirectory(pluginManager)

# for
add_subdirectory(QSimpleUpdater)

if (BUILD_RECONSTRUCTION)
    add_subdirectory(reconstruction)
endif ()

add_subdirectory(extern/QCustomPlot EXCLUDE_FROM_ALL)

qt5_wrap_ui(generated_ui_list ${ui_list})
qt5_add_resources(generated_qrc_list ${qrc_list})

if (MSVC)
    # App icon with MSVC
    set(rc_list Resources/images/icon/ecv_icon.rc)

    #to get rid of the (system) console
    add_executable(${PROJECT_NAME} WIN32 ${header_list} ${source_list} ${generated_ui_list} ${generated_qrc_list} ${rc_list} ${txt_list})

    # generate or update translation files(ts)
    if (UPDATE_TRANSLATION_FILES)
        file(GLOB_RECURSE SOURCE_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../*.h ${CMAKE_CURRENT_SOURCE_DIR}/../*.ui)
        find_package(Qt5Core REQUIRED)
        find_package(Qt5LinguistTools REQUIRED)
        set(TRANSLATION_FILES ${CMAKE_CURRENT_SOURCE_DIR}/translations/ACloudViewer_zh.ts)
        foreach (TRANSLATION_FILE ${TRANSLATION_FILES})
            message("TRANSLATION_FILE : ${TRANSLATION_FILE}")
            qt5_update_translation(QMs ${PROJECT_NAME} ${SOURCE_LIST} ${TRANSLATION_FILE})
            message("QMs: ${QMs}")
        endforeach ()
    endif ()
elseif (APPLE)
    add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${header_list} ${source_list} ${generated_ui_list} ${generated_qrc_list} ${txt_list})
else ()
    add_executable(${PROJECT_NAME} ${header_list} ${source_list} ${generated_ui_list} ${generated_qrc_list} ${rc_list} ${txt_list})
endif ()

# Put together our Application bundle on macOS
if (APPLE)
    add_subdirectory(Mac)
elseif(UNIX)
    set_target_properties(${TARGET_NAME} PROPERTIES
                            INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib/")
endif ()

# Translation Note: must be after add_subdirectory(Mac) on apple
add_subdirectory(translations)

if (BUILD_RECONSTRUCTION)
    if (${BUILD_GUI})
        target_compile_definitions(${PROJECT_NAME} PRIVATE GUI_ENABLED)
    endif ()
    if (${USE_SIMD})
        target_compile_definitions(${PROJECT_NAME} PRIVATE SIMD_ENABLED)
    endif ()
    if (${WITH_OPENMP})
        target_compile_definitions(${PROJECT_NAME} PRIVATE OPENMP_ENABLED)
    endif ()
    if (${BUILD_CUDA_MODULE})
        target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_ENABLED)
    endif ()
    if (${CVCORELIB_USE_CGAL})
        target_compile_definitions(${PROJECT_NAME} PRIVATE CGAL_ENABLED)
    endif ()
    if (${BUILD_UNIT_TESTS})
        target_compile_definitions(${PROJECT_NAME} PRIVATE TESTS_ENABLED)
    endif ()
    target_compile_definitions(${PROJECT_NAME} PRIVATE OPENGL_ENABLED)

    if (MSVC)
        # Some fixes for the Glog library.
        add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
        add_definitions("-DGOOGLE_GLOG_DLL_DECL=")
        # fix gflags library bugs on windows
        add_definitions(-DGFLAGS_DLL_DECLARE_FLAG= -DGFLAGS_DLL_DEFINE_FLAG= -DGFLAGS_IS_A_DLL=0)
        target_link_libraries(${PROJECT_NAME} shlwapi.lib)
    endif ()

    include_directories(${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/include)
    include_directories(${COLMAP_SOURCE_DIR}/lib)
    include_directories(${COLMAP_SOURCE_DIR}/src)

    target_link_libraries(${PROJECT_NAME} colmap)

    # Add custom preprocessor definitions
    target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_RECONSTRUCTION)
endif ()

target_link_libraries(${PROJECT_NAME}
        CVAppCommon
        QCustomPlot
        Qt5::Network
        Qt5::WebSockets
        Qt5::PrintSupport
        )

if (CVCORELIB_USE_QT_CONCURRENT)
    # We want *deployqt to handle this dependency, so instead of just copying the file
    # link to the target.
    target_link_libraries(${PROJECT_NAME}
            Qt5::Concurrent
            )
endif ()

# contrib. libraries support
if (APPLE)
    target_link_contrib(${PROJECT_NAME} ${ACloudViewer_MAC_FRAMEWORK_DIR})
elseif (UNIX)
    target_link_contrib(${PROJECT_NAME} lib)
else ()
    target_link_contrib(${PROJECT_NAME} ${ACloudViewer_DEST_FOLDER})
endif ()

# fix that You must build your code with position independent code if Qt was built with -reduce-relocations
cloudViewer_set_targets_independent(${PROJECT_NAME})

if (BUILD_PYTHON_MODULE)
    target_link_libraries(${PROJECT_NAME} ECV_PYTHON_LIB)
    # Add custom preprocessor definitions
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_PYTHON_MODULE)
endif ()

if (USE_PCL_BACKEND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DVTK_VERSION=\"${VTK_VERSION}\")
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DPCL_VERSION=\"${PCL_VERSION}\")
    target_link_directories(${PROJECT_NAME}
            PUBLIC ${QPCL_ENGINE_LIB_SOURCE_DIR}
            )
    include_directories(${QPCL_ENGINE_LIB_SOURCE_DIR})
    target_link_libraries(${PROJECT_NAME} QPCL_ENGINE_LIB)
endif ()

# Add custom preprocessor definitions
if (WIN32)
    set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS)
    if (MSVC)
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS " /MANIFEST:NO")
    endif ()
endif ()

if (WIN32)
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/${PROJECT_NAME}.exe ${CMAKE_BINARY_DIR}/eCV/$<CONFIG>/${PROJECT_NAME}.exe
            VERBATIM
    )
elseif (APPLE)
    set(RESOURCE_DIR_NAME "Contents/Resources")
    file(GLOB RESOURCE_FILES "Mac/*.icns")
    set(CLOUDVIEWER_TRANSLATION_DESTINATION "${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}.app/Contents/translations")
    set(CLOUDVIEWER_RESOURCE_DESTINATION "${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}.app/${RESOURCE_DIR_NAME}")
    set(CLOUDVIEWER_PLUGINS_DESTINATION "${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}.app/Contents/cvPlugins")
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            # copy the resource files into the bundle
            COMMAND ${CMAKE_COMMAND} -E remove_directory "${CLOUDVIEWER_RESOURCE_DESTINATION}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CLOUDVIEWER_RESOURCE_DESTINATION}"
            COMMAND ${CMAKE_COMMAND} -E copy ${RESOURCE_FILES} "${CLOUDVIEWER_RESOURCE_DESTINATION}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CLOUDVIEWER_TRANSLATION_DESTINATION}"
            COMMAND sh -c "cp ${CMAKE_BINARY_DIR}/eCV/translations/*.qm ${CLOUDVIEWER_TRANSLATION_DESTINATION}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/bin/cvPlugins "${CLOUDVIEWER_PLUGINS_DESTINATION}"
            VERBATIM
    )

    # for macdeployqt
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}.app ${CMAKE_BINARY_DIR}/eCV/${PROJECT_NAME}.app
            VERBATIM
    )
else ()
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME} ${CMAKE_BINARY_DIR}/eCV/$<CONFIG>/${PROJECT_NAME}
            VERBATIM
    )
endif ()


if (WIN32)
    DeployQt(
            TARGET
            ${PROJECT_NAME}
            DEPLOY_PATH
            ${ACloudViewer_DEST_FOLDER}
    )
elseif (APPLE)
    DeployQt(
            TARGET
            ${PROJECT_NAME}
            DEPLOY_PATH
            ${PROJECT_NAME}
    )
endif ()

# App icon with Code::Blocks/MinGW
if (WIN32)
    if (MINGW)
        if (NOT CMAKE_RC_COMPILER)
            set(CMAKE_RC_COMPILER windres)
        endif ()
        add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_RC_COMPILER} -i ${CMAKE_CURRENT_SOURCE_DIR}/Resources/images/icon/ecv_icon.rc --input-format=rc -o ${CMAKE_CURRENT_BINARY_DIR}/ecv_icon.res -O coff)
    endif ()
endif ()

if (NOT APPLE)
    # Install program.
    # On macOs, the DeployQt step will install the bundle that contains the executable with
    # library paths properly set, reinstalling the executable here would break the bundle.
    message(STATUS "INSTALL_DESTINATIONS: " ${INSTALL_DESTINATIONS})
    message(STATUS "ACloudViewer_DEST_FOLDER: " ${ACloudViewer_DEST_FOLDER})
    cloudViewer_install_ext(TARGETS ${PROJECT_NAME} ${INSTALL_DESTINATIONS} "")
endif ()

# Auxiliary files
set(auxFiles ${COLOUDVIEWER_PROJECT_PATH}/CHANGELOG.txt bin_other/license.txt bin_other/global_shift_list_template.txt)

if (WIN32)
    # Additional auxiliary file(s)
    file(GLOB batch_list bin_other/*.bat)
    list(APPEND auxFiles ${batch_list})
endif ()

# Install auxiliary files
if (WIN32)
    foreach (filename ${auxFiles})
        cloudViewer_install_ext(FILES ${filename} ${ACloudViewer_DEST_FOLDER} "")
    endforeach ()
elseif (APPLE)
    foreach (filename ${auxFiles})
        cloudViewer_install_ext(FILES ${filename} ${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME} "")
        if (${BUILD_GUI})
            cloudViewer_install_ext(FILES ${filename} ${CMAKE_INSTALL_PREFIX}/bin/CloudViewer "")
        endif ()
        if (${BUILD_RECONSTRUCTION})
            cloudViewer_install_ext(FILES ${filename} ${CMAKE_INSTALL_PREFIX}/bin/colmap "")
        endif()
    endforeach ()
endif ()

# Install plugins and shaders in the correct folder for each platform
if (APPLE) # no need to install due to macdeployqt
    # InstallPlugins(
    #         DEST_PATH "${ACloudViewer_MAC_PLUGIN_DIR}"
    #         SHADER_DEST_PATH "${ACloudViewer_MAC_BASE_DIR}/Contents/Shaders"
    # )
elseif (UNIX)
    InstallPlugins(
            DEST_PATH "plugins"
            SHADER_DEST_PATH shaders
    )
else () # no need ?
    InstallPlugins(
            DEST_PATH "${ACloudViewer_DEST_FOLDER}"
            DEST_FOLDER "plugins"
            SHADER_DEST_PATH "${ACloudViewer_DEST_FOLDER}"
            SHADER_DEST_FOLDER "shaders"
    )
endif ()
