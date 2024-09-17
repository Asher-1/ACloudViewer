# ------------------------------------------------------------------------------
# Qt
# ------------------------------------------------------------------------------
## we will use cmake automoc / autouic / autorcc feature
# init qt

set( CMAKE_AUTOMOC ON )
set( CMAKE_AUTORCC ON )

# FIXME Eventually turn this on when we've completed the move to targets
#set( CMAKE_AUTOUIC ON )

set( QT5_ROOT_PATH CACHE PATH "Qt5 root directory (i.e. where the 'bin' folder lies)" )
if ( QT5_ROOT_PATH )
	list( APPEND CMAKE_PREFIX_PATH ${QT5_ROOT_PATH} )
endif()

# find qt5 components
# find_package(Qt5 COMPONENTS OpenGL Widgets Core Svg Gui PrintSupport Concurrent REQUIRED)
find_package( Qt5
    COMPONENTS
        Core
        Gui
        Svg
        OpenGL
        Widgets
        Network
        WebSockets
        Concurrent
        PrintSupport
    REQUIRED
)

# in the case no Qt5Config.cmake file could be found, cmake will explicitly ask the user for the QT5_DIR containing it!
# thus no need to keep additional variables and checks

# Starting with the QtCore lib, find the bin and root directories
get_target_property( Qt5_LIB_LOCATION Qt5::Core LOCATION_${CMAKE_BUILD_TYPE} )
get_filename_component( Qt5_LIB_LOCATION ${Qt5_LIB_LOCATION} DIRECTORY )
if ( WIN32 )
    get_target_property( QMAKE_LOCATION Qt5::qmake IMPORTED_LOCATION )
    get_filename_component( Qt5_BIN_DIR ${QMAKE_LOCATION} DIRECTORY )
    get_filename_component( QT5_ROOT_PATH "${Qt5_BIN_DIR}/.." ABSOLUTE )
endif()

# Qt5 was built with -reduce-relocations.
# fix that can not be used when making a PIE object; recompile with -fPIC issue
if(Qt5_POSITION_INDEPENDENT_CODE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    if(BUILD_CUDA_MODULE AND NOT WIN32)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC")
    endif()
endif()

# fix nvcc fatal : Unknown option 'fPIC'
if ( BUILD_CUDA_MODULE AND NOT WIN32)
    # Warning: convert the fpic option in Qt5::Core over to INTERFACE_POSITION_INDEPENDENT_CODE
    get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
    string(REPLACE "-fPIC" "" new_core_options ${core_options})

    set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
    set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")
endif()


# turn on QStringBuilder for more efficient string construction
#	see https://doc.qt.io/qt-5/qstring.html#more-efficient-string-construction
add_definitions( -DQT_USE_QSTRINGBUILDER )
				
# ------------------------------------------------------------------------------
# OpenGL
# ------------------------------------------------------------------------------
if ( WIN32 )
	# Where to find OpenGL libraries
	set(WINDOWS_OPENGL_LIBS "C:\\Program Files (x86)\\Windows Kits\\8.0\\Lib\\win8\\um\\x64" CACHE PATH "WindowsSDK libraries" )
	list( APPEND CMAKE_PREFIX_PATH ${WINDOWS_OPENGL_LIBS} )
endif()

# Intel's Threading Building Blocks (TBB)
if (COMPILE_CV_CORE_LIB_WITH_TBB)
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_TBB")
	include_directories( ${TBB_INCLUDE_DIRS} )
endif()


get_target_property( QMAKE_LOCATION Qt5::qmake IMPORTED_LOCATION )
get_filename_component( Qt5_BIN_DIR ${QMAKE_LOCATION} DIRECTORY )
get_filename_component( QT5_ROOT_PATH "${Qt5_BIN_DIR}/.." ABSOLUTE )
get_filename_component( QT5_PLUGINS_PATH "${Qt5_BIN_DIR}/../plugins" ABSOLUTE )

message(STATUS "Qt5_BIN_DIR: " ${Qt5_BIN_DIR})
message(STATUS "QT5_ROOT_PATH: " ${QT5_ROOT_PATH})
message(STATUS "QT5_PLUGINS_PATH: " ${QT5_PLUGINS_PATH})
set(QT5_PLUGINS_PATH_LIST   "${QT5_PLUGINS_PATH}/platforms" "${QT5_PLUGINS_PATH}/iconengines" 
                            "${QT5_PLUGINS_PATH}/imageformats" "${QT5_PLUGINS_PATH}/xcbglintegrations")
message(STATUS "QT5_PLUGINS_PATH_LIST: " ${QT5_PLUGINS_PATH_LIST})