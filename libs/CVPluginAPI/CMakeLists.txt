cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

project( CVPluginAPI )

add_library( ${PROJECT_NAME} SHARED )

# Qt5/Qt6 compatibility
# Note: Gui is already found in CMakeExternalLibs.cmake

add_subdirectory( include )
add_subdirectory( src )
add_subdirectory( ui )

if (BUILD_UNIT_TESTS)
    add_subdirectory( tests )
endif()

# We only need the includes from the stub
get_target_property( STUB_INCLUDES CVPluginStub INCLUDE_DIRECTORIES )

target_include_directories( ${PROJECT_NAME} PUBLIC "${STUB_INCLUDES}" )

# Header-only dependency: ecvAssetIntegrity::PinnedDigest() resolves
# release-asset digests from the registry published with AICore. No library
# linkage needed.
target_include_directories( ${PROJECT_NAME} PRIVATE
	${CMAKE_SOURCE_DIR}/core/AICore/include )

# Plain signature: this target and the shared PluginZlibMinizip module
# link old-style throughout (CMake rejects mixing the two forms).
target_link_libraries( ${PROJECT_NAME}
	CV_DB_LIB
	CV_IO_LIB
	Qt::Network
)

# ecvAICoreMeshViewerDialog: the QOpenGLWidget orbit preview shared by the
# AICore mesh plugins (qTrellis, qInstantMesh). Qt5 hosts QOpenGLWidget in
# Qt::OpenGL; Qt6 moved it to the OpenGLWidgets module. Plain signature:
# the rest of this target links old-style and CMake requires one style.
target_link_libraries( ${PROJECT_NAME} Qt::OpenGL)
if(TARGET Qt::OpenGLWidgets)
	target_link_libraries( ${PROJECT_NAME} Qt::OpenGLWidgets)
endif()

# zlib + minizip for cross-platform zip extraction (ecvTestDataRepository)
include( ${CMAKE_SOURCE_DIR}/plugins/cmake/PluginZlibMinizip.cmake )
cloudviewer_plugin_link_zlib_minizip(${PROJECT_NAME}
	PLUGIN_NAME CVPluginAPI)

# _USE_MATH_DEFINES: the ecvAICoreMeshViewer orbit math uses M_PI. MSVC's
# <cmath> only exposes M_PI with this macro defined; glibc exposes it
# unconditionally (same contract the qInstantMesh target carried before the
# viewer migrated here from qTrellis/qInstantMesh).
target_compile_definitions( ${PROJECT_NAME} PRIVATE
	CVPLUGIN_API_LIBRARY_BUILD
	_USE_MATH_DEFINES )

set_target_properties( ${PROJECT_NAME} PROPERTIES
	AUTOMOC ON
	AUTOUIC ON # FIXME Remove after everything has moved to targets and we can set it globally
	CXX_VISIBILITY_PRESET hidden
)

if (NOT APPLE) # no need to copy targets on macos due to rpath
	# install (shared) lib to specified destinations
	InstallSharedLibrary(TARGET ${PROJECT_NAME})
endif ()