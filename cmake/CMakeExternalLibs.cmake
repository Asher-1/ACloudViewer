# ------------------------------------------------------------------------------
# Qt
# ------------------------------------------------------------------------------
## we will use cmake automoc / autouic / autorcc feature
# init qt

set( CMAKE_AUTOMOC ON )
set( CMAKE_AUTORCC ON )

# FIXME Eventually turn this on when we've completed the move to targets
#set( CMAKE_AUTOUIC ON )

# Qt version selection: USE_QT6 option (similar to PCL 1.15+)
if(USE_QT6)
    set(QT_VERSION_MAJOR 6)
    message(STATUS "Building with Qt6")
    
    # Qt6 root path
    set( QT_ROOT_PATH CACHE PATH "Qt6 root directory (i.e. where the 'bin' folder lies)" )
    if ( QT_ROOT_PATH )
        list( APPEND CMAKE_PREFIX_PATH ${QT_ROOT_PATH} )
    endif()
    
    # find qt6 components
    find_package( Qt6
        COMPONENTS
            Core
            Gui
            Svg
            OpenGL
            OpenGLWidgets
            Widgets
            Network
            WebSockets
            Concurrent
            PrintSupport
        REQUIRED
    )
    
    # Set Qt6 as the active Qt version
    set(QT_PREFIX Qt6)
    set(QT_MAJOR_VERSION 6)
    
    # Get Qt6 paths - use unified Qt:: prefix after aliases are created
    # Note: We need to get QMAKE_LOCATION before creating aliases, so use Qt6::qmake here
    get_target_property( QMAKE_LOCATION Qt6::qmake IMPORTED_LOCATION )
    if(QMAKE_LOCATION)
        get_filename_component( QT_BIN_DIR ${QMAKE_LOCATION} DIRECTORY )
        get_filename_component( QT_ROOT_PATH "${QT_BIN_DIR}/.." ABSOLUTE )
        get_filename_component( QT_PLUGINS_PATH "${QT_BIN_DIR}/../plugins" ABSOLUTE )
    endif()
    
    # Qt6 was built with -reduce-relocations.
    if(Qt6_POSITION_INDEPENDENT_CODE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        if(BUILD_CUDA_MODULE AND NOT WIN32)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}--compiler-options=-fPIC")
        endif()
    endif()
    
    # fix nvcc fatal : Unknown option 'fPIC'
    # Note: Use Qt6::Core here since we need to modify properties before creating aliases
    if ( BUILD_CUDA_MODULE AND NOT WIN32)
        get_property(core_options TARGET Qt6::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
        string(REPLACE "-fPIC" "" new_core_options ${core_options})
        set_property(TARGET Qt6::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
        set_property(TARGET Qt6::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")
    endif()
    
    message(STATUS "Qt6_BIN_DIR: " ${QT_BIN_DIR})
    message(STATUS "QT6_ROOT_PATH: " ${QT_ROOT_PATH})
    message(STATUS "QT6_PLUGINS_PATH: " ${QT_PLUGINS_PATH})
    
    # Backward compatibility: also set Qt5 variables pointing to Qt6
    set(QT5_ROOT_PATH ${QT_ROOT_PATH})
    set(QT5_PLUGINS_PATH ${QT_PLUGINS_PATH})
    set(Qt5_BIN_DIR ${QT_BIN_DIR})
    
    # Create unified Qt:: aliases pointing to Qt6:: targets
    # This allows code to use Qt::Core instead of Qt6::Core or Qt5::Core
    # Also create Qt5:: aliases for backward compatibility
    set(QT_COMPONENTS Core Gui Widgets OpenGL Svg Network WebSockets Concurrent PrintSupport)
    foreach(COMPONENT ${QT_COMPONENTS})
        # Create unified Qt:: alias
        if(NOT TARGET Qt::${COMPONENT})
            add_library(Qt::${COMPONENT} INTERFACE IMPORTED)
            set_target_properties(Qt::${COMPONENT} PROPERTIES
                INTERFACE_LINK_LIBRARIES Qt6::${COMPONENT}
            )
        endif()
        # Create Qt5:: alias for backward compatibility
        if(NOT TARGET Qt5::${COMPONENT})
            add_library(Qt5::${COMPONENT} INTERFACE IMPORTED)
            set_target_properties(Qt5::${COMPONENT} PROPERTIES
                INTERFACE_LINK_LIBRARIES Qt6::${COMPONENT}
            )
        endif()
    endforeach()
    
    # Qt6-specific components (not available in Qt5)
    if(NOT TARGET Qt::OpenGLWidgets)
        add_library(Qt::OpenGLWidgets INTERFACE IMPORTED)
        set_target_properties(Qt::OpenGLWidgets PROPERTIES
            INTERFACE_LINK_LIBRARIES Qt6::OpenGLWidgets
        )
    endif()
    
    # Create unified Qt::qmake alias
    if(NOT TARGET Qt::qmake)
        add_executable(Qt::qmake IMPORTED)
        set_target_properties(Qt::qmake PROPERTIES
            IMPORTED_LOCATION ${QMAKE_LOCATION}
        )
    endif()
    # Create Qt5::qmake alias for backward compatibility
    if(NOT TARGET Qt5::qmake)
        add_executable(Qt5::qmake IMPORTED)
        set_target_properties(Qt5::qmake PROPERTIES
            IMPORTED_LOCATION ${QMAKE_LOCATION}
        )
    endif()
    
else()
    set(QT_VERSION_MAJOR 5)
    message(STATUS "Building with Qt5")
    
    # Qt5 root path
    set( QT5_ROOT_PATH CACHE PATH "Qt5 root directory (i.e. where the 'bin' folder lies)" )
    if ( QT5_ROOT_PATH )
        list( APPEND CMAKE_PREFIX_PATH ${QT5_ROOT_PATH} )
    endif()
    
    # find qt5 components
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
    
    # Set Qt5 as the active Qt version
    set(QT_PREFIX Qt5)
    set(QT_MAJOR_VERSION 5)
    
    # Starting with the QtCore lib, find the bin and root directories
    # Note: Use Qt5::Core here since we need to get properties before creating aliases
    get_target_property( Qt5_LIB_LOCATION Qt5::Core LOCATION_${CMAKE_BUILD_TYPE} )
    get_filename_component( Qt5_LIB_LOCATION ${Qt5_LIB_LOCATION} DIRECTORY )
    if ( WIN32 )
        get_target_property( QMAKE_LOCATION Qt5::qmake IMPORTED_LOCATION )
        get_filename_component( Qt5_BIN_DIR ${QMAKE_LOCATION} DIRECTORY )
        get_filename_component( QT5_ROOT_PATH "${Qt5_BIN_DIR}/.." ABSOLUTE )
    endif()
    
    # Qt5 was built with -reduce-relocations.
    if(Qt5_POSITION_INDEPENDENT_CODE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        if(BUILD_CUDA_MODULE AND NOT WIN32)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}--compiler-options=-fPIC")
        endif()
    endif()
    
    # fix nvcc fatal : Unknown option 'fPIC'
    # Note: Use Qt5::Core here since we need to modify properties before creating aliases
    if ( BUILD_CUDA_MODULE AND NOT WIN32)
        get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
        string(REPLACE "-fPIC" "" new_core_options ${core_options})
        set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
        set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")
    endif()
    
    # Get Qt5 paths - use Qt5::qmake here since we need it before creating aliases
    get_target_property( QMAKE_LOCATION Qt5::qmake IMPORTED_LOCATION )
    get_filename_component( Qt5_BIN_DIR ${QMAKE_LOCATION} DIRECTORY )
    get_filename_component( QT5_ROOT_PATH "${Qt5_BIN_DIR}/.." ABSOLUTE )
    get_filename_component( QT5_PLUGINS_PATH "${Qt5_BIN_DIR}/../plugins" ABSOLUTE )
    
    message(STATUS "Qt5_BIN_DIR: " ${Qt5_BIN_DIR})
    message(STATUS "QT5_ROOT_PATH: " ${QT5_ROOT_PATH})
    message(STATUS "QT5_PLUGINS_PATH: " ${QT5_PLUGINS_PATH})
    
    # For forward compatibility, also set unified variables
    set(QT_ROOT_PATH ${QT5_ROOT_PATH})
    set(QT_PLUGINS_PATH ${QT5_PLUGINS_PATH})
    set(QT_BIN_DIR ${Qt5_BIN_DIR})
    
    # Create unified Qt:: aliases pointing to Qt5:: targets
    # This allows code to use Qt::Core instead of Qt5::Core or Qt6::Core
    set(QT_COMPONENTS Core Gui Widgets OpenGL Svg Network WebSockets Concurrent PrintSupport)
    foreach(COMPONENT ${QT_COMPONENTS})
        # Create unified Qt:: alias
        if(NOT TARGET Qt::${COMPONENT})
            add_library(Qt::${COMPONENT} INTERFACE IMPORTED)
            set_target_properties(Qt::${COMPONENT} PROPERTIES
                INTERFACE_LINK_LIBRARIES Qt5::${COMPONENT}
            )
        endif()
    endforeach()
    
    # Create unified Qt::qmake alias
    if(NOT TARGET Qt::qmake)
        add_executable(Qt::qmake IMPORTED)
        set_target_properties(Qt::qmake PROPERTIES
            IMPORTED_LOCATION ${QMAKE_LOCATION}
        )
    endif()
endif()

# turn on QStringBuilder for more efficient string construction
#	see https://doc.qt.io/qt-5/qstring.html#more-efficient-string-construction
add_compile_definitions(QT_USE_QSTRINGBUILDER)

# Set unified plugin path list (works for both Qt5 and Qt6)
if(USE_QT6)
    set(QT_PLUGINS_PATH_LIST "${QT_PLUGINS_PATH}/platforms")
    if (WIN32)
        list(APPEND QT_PLUGINS_PATH_LIST "${QT_PLUGINS_PATH}/styles")
    endif()
    if (UNIX AND NOT APPLE)
        list(APPEND QT_PLUGINS_PATH_LIST "${QT_PLUGINS_PATH}/xcbglintegrations")
        list(APPEND QT_PLUGINS_PATH_LIST "${QT_PLUGINS_PATH}/platformthemes")
    endif()
    list(APPEND QT_PLUGINS_PATH_LIST "${QT_PLUGINS_PATH}/iconengines")
    list(APPEND QT_PLUGINS_PATH_LIST "${QT_PLUGINS_PATH}/imageformats")
    
    # Backward compatibility: also set Qt5 variables
    set(QT5_PLUGINS_PATH_LIST ${QT_PLUGINS_PATH_LIST})
else()
    set(QT_PLUGINS_PATH_LIST "${QT5_PLUGINS_PATH}/platforms")
    if (WIN32)
        list(APPEND QT_PLUGINS_PATH_LIST "${QT5_PLUGINS_PATH}/styles")
    endif()
    if (UNIX AND NOT APPLE)
        list(APPEND QT_PLUGINS_PATH_LIST "${QT5_PLUGINS_PATH}/xcbglintegrations")
        list(APPEND QT_PLUGINS_PATH_LIST "${QT5_PLUGINS_PATH}/platformthemes")
    endif()
    list(APPEND QT_PLUGINS_PATH_LIST "${QT5_PLUGINS_PATH}/iconengines")
    list(APPEND QT_PLUGINS_PATH_LIST "${QT5_PLUGINS_PATH}/imageformats")
endif()

message(STATUS "QT_PLUGINS_PATH_LIST: " ${QT_PLUGINS_PATH_LIST})

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
