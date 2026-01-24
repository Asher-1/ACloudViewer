cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

project(${MAIN_APP_NAME} VERSION ${PROJECT_VERSION})

OPTION(UPDATE_TRANSLATION_FILES "Generate or update ts files" OFF)

include(CMakePolicies NO_POLICY_SCOPE)

if (MSVC)
    if (CVCORELIB_USE_TBB)
        include_directories(${TBB_INCLUDE_DIRS})
    endif ()
endif ()

if (BUILD_PYTHON_MODULE)
    include_directories(${CV_PYTHON_LIB_SOURCE_DIR})
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

# Use generated CHANGELOG.txt from build directory
# CHANGELOG_TXT is set in parent CMakeLists.txt
if(DEFINED CHANGELOG_TXT)
    set(CHANGELOG_FILE "${CHANGELOG_TXT}")
else()
    set(CHANGELOG_FILE "${COLOUDVIEWER_PROJECT_PATH}/CHANGELOG.md")
endif()
file(GLOB txt_list TODO.txt ${CHANGELOG_FILE})

add_subdirectory(db_tree)
add_subdirectory(pluginManager)

# for
add_subdirectory(QSimpleUpdater)

if (BUILD_RECONSTRUCTION)
    add_subdirectory(reconstruction)
endif ()

add_subdirectory(extern/QCustomPlot EXCLUDE_FROM_ALL)

# Qt5/Qt6 compatibility
if(USE_QT6)
    qt6_wrap_ui(generated_ui_list ${ui_list})
    qt6_add_resources(generated_qrc_list ${qrc_list})
else()
    qt5_wrap_ui(generated_ui_list ${ui_list})
    qt5_add_resources(generated_qrc_list ${qrc_list})
endif()

if (MSVC)
    # App icon with MSVC
    set(rc_list Resources/images/icon/ecv_icon.rc)

    #to get rid of the (system) console
    add_executable(${PROJECT_NAME} WIN32 ${header_list} ${source_list} ${generated_ui_list} ${generated_qrc_list} ${rc_list} ${txt_list})

    # generate or update translation files(ts)
    if (UPDATE_TRANSLATION_FILES)
        file(GLOB_RECURSE SOURCE_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../*.h ${CMAKE_CURRENT_SOURCE_DIR}/../*.ui)
        # Qt5/Qt6 compatibility
        # Note: Core is already found in CMakeExternalLibs.cmake
        # LinguistTools needs to be found separately (it's a tool package)
        if(USE_QT6)
            find_package(Qt6 COMPONENTS LinguistTools REQUIRED)
            set(TRANSLATION_FILES ${CMAKE_CURRENT_SOURCE_DIR}/translations/ACloudViewer_zh.ts)
            foreach (TRANSLATION_FILE ${TRANSLATION_FILES})
                message("TRANSLATION_FILE : ${TRANSLATION_FILE}")
                qt6_update_translation(QMs ${PROJECT_NAME} ${SOURCE_LIST} ${TRANSLATION_FILE})
                message("QMs: ${QMs}")
            endforeach ()
        else()
            find_package(Qt5LinguistTools REQUIRED)
            set(TRANSLATION_FILES ${CMAKE_CURRENT_SOURCE_DIR}/translations/ACloudViewer_zh.ts)
            foreach (TRANSLATION_FILE ${TRANSLATION_FILES})
                message("TRANSLATION_FILE : ${TRANSLATION_FILE}")
                qt5_update_translation(QMs ${PROJECT_NAME} ${SOURCE_LIST} ${TRANSLATION_FILE})
                message("QMs: ${QMs}")
            endforeach ()
        endif()
    endif ()
elseif (APPLE)
    add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${header_list} ${source_list} ${generated_ui_list} ${generated_qrc_list} ${txt_list})
else ()
    add_executable(${PROJECT_NAME} ${header_list} ${source_list} ${generated_ui_list} ${generated_qrc_list} ${rc_list} ${txt_list})
endif ()

# Put together our Application bundle on macOS
if (APPLE)
    add_subdirectory(Mac)
endif ()

# Translation Note: must be after add_subdirectory(Mac) on apple
add_subdirectory(translations)

if (BUILD_RECONSTRUCTION)
    if (${BUILD_GUI})
        target_compile_definitions(${PROJECT_NAME} PRIVATE GUI_ENABLED)
    endif ()
    if (${DOWNLOAD_ENABLED})
        target_compile_definitions(${PROJECT_NAME} PRIVATE COLMAP_DOWNLOAD_ENABLED)
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
    if (${PBA_ENABLED})
        target_compile_definitions(${PROJECT_NAME} PRIVATE PBA_ENABLED)
    endif ()
    target_compile_definitions(${PROJECT_NAME} PRIVATE OPENGL_ENABLED)

    if (MSVC)
        # Some fixes for the Glog library.
        add_compile_definitions(GLOG_NO_ABBREVIATED_SEVERITIES)
        add_compile_definitions(GOOGLE_GLOG_DLL_DECL=)
        # fix gflags library bugs on windows
        add_compile_definitions(GFLAGS_DLL_DECLARE_FLAG= GFLAGS_DLL_DEFINE_FLAG= GFLAGS_IS_A_DLL=0)
        target_link_libraries(${PROJECT_NAME} shlwapi.lib)
    endif ()

    include_directories(${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}/include)
    include_directories(${COLMAP_SOURCE_DIR}/lib)
    include_directories(${COLMAP_SOURCE_DIR}/src)

    target_link_libraries(${PROJECT_NAME} ${COLMAP_LIB_NAME})

    # Add custom preprocessor definitions
    target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_RECONSTRUCTION)
endif ()

# Qt5/Qt6 compatibility - use unified Qt:: prefix
target_link_libraries(${PROJECT_NAME}
        CVAppCommon
        QCustomPlot
        Qt::Network
        Qt::WebSockets
        Qt::PrintSupport
        )

if (CVCORELIB_USE_QT_CONCURRENT)
    target_link_libraries(${PROJECT_NAME}
            Qt::Concurrent
            )
endif()

# running gui app from the build folder needs tbb.dll to be in the same folder.
if (NOT USE_SYSTEM_TBB)
    if (TARGET tbb)
        if (WIN32)
            add_custom_command(
                TARGET ${PROJECT_NAME}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:tbb> "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/"
            )
        else()
            add_custom_command(
                TARGET ${PROJECT_NAME}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:tbb> "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/"
            )
        endif()
    endif()
endif()

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
    target_link_libraries(${PROJECT_NAME} CV_PYTHON_LIB)
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
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/${CloudViewer_INSTALL_BIN_DIR}/$<CONFIG>/${PROJECT_NAME}.exe ${CMAKE_BINARY_DIR}/app/$<CONFIG>/${PROJECT_NAME}.exe
            VERBATIM
    )
elseif (APPLE)
    set(RESOURCE_DIR_NAME "Contents/Resources")
    file(GLOB RESOURCE_FILES "Mac/*.icns")
    set(BINARY_APP_PATH ${CMAKE_BINARY_DIR}/${CloudViewer_INSTALL_BIN_DIR}/${PROJECT_NAME}.app)
    set(CLOUDVIEWER_TRANSLATION_DESTINATION "${BINARY_APP_PATH}/Contents/translations")
    set(CLOUDVIEWER_RESOURCE_DESTINATION "${BINARY_APP_PATH}/${RESOURCE_DIR_NAME}")
    set(CLOUDVIEWER_PLUGINS_DESTINATION "${BINARY_APP_PATH}/Contents/cvPlugins")
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            # copy the resource files into the bundle
            COMMAND ${CMAKE_COMMAND} -E remove_directory "${CLOUDVIEWER_RESOURCE_DESTINATION}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CLOUDVIEWER_RESOURCE_DESTINATION}"
            COMMAND ${CMAKE_COMMAND} -E copy ${RESOURCE_FILES} "${CLOUDVIEWER_RESOURCE_DESTINATION}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CLOUDVIEWER_TRANSLATION_DESTINATION}"
            COMMAND sh -c "cp ${CMAKE_BINARY_DIR}/app/translations/*.qm ${CLOUDVIEWER_TRANSLATION_DESTINATION}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/${CloudViewer_INSTALL_BIN_DIR}/cvPlugins "${CLOUDVIEWER_PLUGINS_DESTINATION}"
             # for macdeployqt
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${BINARY_APP_PATH} ${CMAKE_BINARY_DIR}/app/${PROJECT_NAME}.app
            VERBATIM
    )
else ()
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/${CloudViewer_INSTALL_BIN_DIR}/${PROJECT_NAME} ${CMAKE_BINARY_DIR}/app/$<CONFIG>/${PROJECT_NAME}
            VERBATIM
    )
endif ()

if (WIN32 OR APPLE)
    DeployQt(
            TARGET
            ${PROJECT_NAME}
            DEPLOY_PATH
            ${ACloudViewer_DEST_FOLDER}
    )
elseif (UNIX)
    # Install program.
    # On macOs and win32, the DeployQt step will install the bundle that contains the executable with
    # library paths properly set, reinstalling the executable here would break the bundle.
    message(STATUS "INSTALL_DESTINATIONS: " ${INSTALL_DESTINATIONS})
    message(STATUS "ACloudViewer_DEST_FOLDER: " ${ACloudViewer_DEST_FOLDER})
    cloudViewer_install_ext(TARGETS ${PROJECT_NAME} ${INSTALL_DESTINATIONS}/${CloudViewer_INSTALL_BIN_DIR}/${PROJECT_NAME} "")
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

# Auxiliary files
# Use generated CHANGELOG.txt from build directory
if(DEFINED CHANGELOG_TXT)
    set(CHANGELOG_FILE "${CHANGELOG_TXT}")
else()
    set(CHANGELOG_FILE "${COLOUDVIEWER_PROJECT_PATH}/CHANGELOG.md")
endif()
set(auxFiles ${CHANGELOG_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/bin_other/license.txt ${CMAKE_CURRENT_SOURCE_DIR}/bin_other/global_shift_list_template.txt)

if (WIN32)
    # Additional auxiliary file(s)
    file(GLOB batch_list ${CMAKE_CURRENT_SOURCE_DIR}/bin_other/*.bat)
    list(APPEND auxFiles ${batch_list})
endif ()

# Install auxiliary files
if (WIN32)
    foreach (filename ${auxFiles})
        cloudViewer_install_ext(FILES ${filename} ${ACloudViewer_DEST_FOLDER} "")
    endforeach ()
elseif (APPLE)
    set(APPLE_INSTALL_BIN_PATH ${CMAKE_INSTALL_PREFIX}/${CloudViewer_INSTALL_BIN_DIR})
    foreach (filename ${auxFiles})
        cloudViewer_install_ext(FILES ${filename} ${APPLE_INSTALL_BIN_PATH}/${MAIN_APP_NAME} "")
        if (${BUILD_GUI})
            cloudViewer_install_ext(FILES ${filename} ${APPLE_INSTALL_BIN_PATH}/${CLOUDVIEWER_APP_NAME} "")
        endif ()
        if (${BUILD_RECONSTRUCTION})
            cloudViewer_install_ext(FILES ${filename} ${APPLE_INSTALL_BIN_PATH}/${COLMAP_APP_NAME} "")
        endif()
    endforeach ()
endif ()

# Install plugins and shaders in the correct folder for each platform
if (APPLE)
    # no need to install due to macdeployqt already done
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
