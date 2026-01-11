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
    
    # Qt6 root path (user can set this to override auto-detection)
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
    
    # Get Qt6 paths - use qmake location as primary source (works on all platforms)
    get_target_property( QT_QMAKE_LOCATION Qt6::qmake IMPORTED_LOCATION )
    get_target_property( _qt_core_location Qt6::Core LOCATION )
    
    # Get bin directory from qmake location
    if(QT_QMAKE_LOCATION)
        get_filename_component( QT_BIN_DIR ${QT_QMAKE_LOCATION} DIRECTORY )
    endif()
    
    # Get lib directory from Qt6::Core location
    if(_qt_core_location)
        get_filename_component( QT_LIB_DIR "${_qt_core_location}" DIRECTORY )
    endif()
    
    # Determine QT_ROOT_PATH and QT_PLUGINS_PATH based on installation type and platform
    # 
    # Platform-specific Qt6 installation layouts:
    # 
    # Linux (apt-installed, Ubuntu 22.04+):
    #   - lib: /usr/lib/x86_64-linux-gnu/libQt6Core.so
    #   - plugins: /usr/lib/x86_64-linux-gnu/qt6/plugins
    #   - qmake: /usr/bin/qmake6
    #
    # Linux (aqtinstall, Ubuntu 20.04):
    #   - lib: /opt/qt6/6.5.3/gcc_64/lib/libQt6Core.so
    #   - plugins: /opt/qt6/6.5.3/gcc_64/plugins
    #   - qmake: /opt/qt6/6.5.3/gcc_64/bin/qmake
    #
    # Windows:
    #   - lib: C:/Qt/6.x.x/msvc2019_64/lib/Qt6Core.lib
    #   - plugins: C:/Qt/6.x.x/msvc2019_64/plugins
    #   - qmake: C:/Qt/6.x.x/msvc2019_64/bin/qmake.exe
    #
    # macOS (Homebrew):
    #   - lib: /usr/local/opt/qt@6/lib/QtCore.framework or libQt6Core.dylib
    #   - plugins: /usr/local/opt/qt@6/share/qt/plugins or /usr/local/opt/qt@6/plugins
    #   - qmake: /usr/local/opt/qt@6/bin/qmake
    #
    # macOS (Qt installer):
    #   - lib: ~/Qt/6.x.x/macos/lib/libQt6Core.dylib
    #   - plugins: ~/Qt/6.x.x/macos/plugins
    #   - qmake: ~/Qt/6.x.x/macos/bin/qmake
    
    set(QT_PLUGINS_PATH "")
    set(QT_ROOT_PATH "")
    
    if(WIN32)
        # Windows: plugins are at <root>/plugins, root is parent of bin
        if(QT_BIN_DIR)
            get_filename_component(QT_ROOT_PATH "${QT_BIN_DIR}/.." ABSOLUTE)
            set(QT_PLUGINS_PATH "${QT_ROOT_PATH}/plugins")
        endif()
    elseif(APPLE)
        # macOS: try multiple common locations
        if(QT_BIN_DIR)
            get_filename_component(QT_ROOT_PATH "${QT_BIN_DIR}/.." ABSOLUTE)
            # Check for standard Qt installer layout
            if(EXISTS "${QT_ROOT_PATH}/plugins")
                set(QT_PLUGINS_PATH "${QT_ROOT_PATH}/plugins")
            # Check for Homebrew layout (share/qt/plugins)
            elseif(EXISTS "${QT_ROOT_PATH}/share/qt/plugins")
                set(QT_PLUGINS_PATH "${QT_ROOT_PATH}/share/qt/plugins")
            endif()
        endif()
    else()
        # Linux: check various installation layouts
        if(QT_LIB_DIR)
            # Check for system Qt6 (apt-installed): plugins at <lib_dir>/qt6/plugins
            if(EXISTS "${QT_LIB_DIR}/qt6/plugins")
                set(QT_PLUGINS_PATH "${QT_LIB_DIR}/qt6/plugins")
                # For system Qt6, root is typically /usr
                get_filename_component(QT_ROOT_PATH "${QT_LIB_DIR}/../.." ABSOLUTE)
            # Check for standalone Qt6 (aqtinstall): plugins at <root>/plugins
            elseif(EXISTS "${QT_LIB_DIR}/../plugins")
                get_filename_component(QT_PLUGINS_PATH "${QT_LIB_DIR}/../plugins" ABSOLUTE)
                get_filename_component(QT_ROOT_PATH "${QT_LIB_DIR}/.." ABSOLUTE)
            endif()
        endif()
        
        # Fallback: try common Linux system paths
        if(NOT QT_PLUGINS_PATH)
            if(EXISTS "/usr/lib/x86_64-linux-gnu/qt6/plugins")
                set(QT_PLUGINS_PATH "/usr/lib/x86_64-linux-gnu/qt6/plugins")
                set(QT_ROOT_PATH "/usr")
            elseif(EXISTS "/usr/lib/aarch64-linux-gnu/qt6/plugins")
                set(QT_PLUGINS_PATH "/usr/lib/aarch64-linux-gnu/qt6/plugins")
                set(QT_ROOT_PATH "/usr")
            elseif(EXISTS "/usr/lib/qt6/plugins")
                set(QT_PLUGINS_PATH "/usr/lib/qt6/plugins")
                set(QT_ROOT_PATH "/usr")
            elseif(EXISTS "/usr/lib64/qt6/plugins")
                set(QT_PLUGINS_PATH "/usr/lib64/qt6/plugins")
                set(QT_ROOT_PATH "/usr")
            endif()
        endif()
        
        # Final fallback: derive from qmake location
        if(NOT QT_PLUGINS_PATH AND QT_BIN_DIR)
            get_filename_component(QT_ROOT_PATH "${QT_BIN_DIR}/.." ABSOLUTE)
            set(QT_PLUGINS_PATH "${QT_ROOT_PATH}/plugins")
        endif()
    endif()
    
    # Verify plugins path exists
    if(NOT EXISTS "${QT_PLUGINS_PATH}")
        message(WARNING "Qt6 plugins path does not exist: ${QT_PLUGINS_PATH}")
    endif()
    
    # Qt6 was built with -reduce-relocations.
    if(Qt6_POSITION_INDEPENDENT_CODE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        if(BUILD_CUDA_MODULE AND NOT WIN32)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}--compiler-options=-fPIC")
        endif()
    endif()
    
    # fix nvcc fatal : Unknown option 'fPIC'
    if ( BUILD_CUDA_MODULE AND NOT WIN32)
        get_property(core_options TARGET Qt6::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
        string(REPLACE "-fPIC" "" new_core_options ${core_options})
        set_property(TARGET Qt6::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
        set_property(TARGET Qt6::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")
    endif()
    
    message(STATUS "QT_BIN_DIR: " ${QT_BIN_DIR})
    message(STATUS "QT_ROOT_PATH: " ${QT_ROOT_PATH})
    message(STATUS "QT_LIB_DIR: " ${QT_LIB_DIR})
    message(STATUS "QT_PLUGINS_PATH: " ${QT_PLUGINS_PATH})
    
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
            IMPORTED_LOCATION ${QT_QMAKE_LOCATION}
        )
    endif()
    # Create Qt5::qmake alias for backward compatibility
    if(NOT TARGET Qt5::qmake)
        add_executable(Qt5::qmake IMPORTED)
        set_target_properties(Qt5::qmake PROPERTIES
            IMPORTED_LOCATION ${QT_QMAKE_LOCATION}
        )
    endif()
    
else()
    set(QT_VERSION_MAJOR 5)
    message(STATUS "Building with Qt5")
    
    # Qt5 root path (user can set this to override auto-detection)
    set( QT_ROOT_PATH CACHE PATH "Qt5 root directory (i.e. where the 'bin' folder lies)" )
    if ( QT_ROOT_PATH )
        list( APPEND CMAKE_PREFIX_PATH ${QT_ROOT_PATH} )
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
    
    # Get Qt5 paths from Qt5::Core library location
    get_target_property( QT_QMAKE_LOCATION Qt5::qmake IMPORTED_LOCATION )
    get_target_property( _qt_core_location Qt5::Core LOCATION_${CMAKE_BUILD_TYPE} )
    
    if(_qt_core_location)
        get_filename_component( QT_LIB_DIR "${_qt_core_location}" DIRECTORY )
    endif()
    
    if(QT_QMAKE_LOCATION)
        get_filename_component( QT_BIN_DIR ${QT_QMAKE_LOCATION} DIRECTORY )
        get_filename_component( QT_ROOT_PATH "${QT_BIN_DIR}/.." ABSOLUTE )
        get_filename_component( QT_PLUGINS_PATH "${QT_ROOT_PATH}/plugins" ABSOLUTE )
    endif()
    
    # Qt5 was built with -reduce-relocations.
    if(Qt5_POSITION_INDEPENDENT_CODE)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        if(BUILD_CUDA_MODULE AND NOT WIN32)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}--compiler-options=-fPIC")
        endif()
    endif()
    
    # fix nvcc fatal : Unknown option 'fPIC'
    if ( BUILD_CUDA_MODULE AND NOT WIN32)
        get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
        string(REPLACE "-fPIC" "" new_core_options ${core_options})
        set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
        set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")
    endif()
    
    message(STATUS "QT_BIN_DIR: " ${QT_BIN_DIR})
    message(STATUS "QT_ROOT_PATH: " ${QT_ROOT_PATH})
    message(STATUS "QT_LIB_DIR: " ${QT_LIB_DIR})
    message(STATUS "QT_PLUGINS_PATH: " ${QT_PLUGINS_PATH})
    
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
            IMPORTED_LOCATION ${QT_QMAKE_LOCATION}
        )
    endif()
endif()

# turn on QStringBuilder for more efficient string construction
#	see https://doc.qt.io/qt-5/qstring.html#more-efficient-string-construction
add_compile_definitions(QT_USE_QSTRINGBUILDER)

# Set unified plugin path list (works for both Qt5 and Qt6)
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
