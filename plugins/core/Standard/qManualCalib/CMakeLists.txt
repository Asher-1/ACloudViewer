cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

option( PLUGIN_STANDARD_QMANUAL_CALIB "Install qManualCalib (Manual Calibration Tools) plugin" OFF )

if( PLUGIN_STANDARD_QMANUAL_CALIB )
	project( QMANUAL_CALIB_PLUGIN )

	option(MCALIB_WITH_FFMPEG_SUPPORT
		"Enable FFmpeg H.264/HEVC decode for online camera bags (requires FFmpeg)"
		ON)

	AddPlugin( NAME ${PROJECT_NAME} )

	add_subdirectory( calib_io )
	add_subdirectory( bev_stitch )
	add_subdirectory( include )
	add_subdirectory( src )

	target_include_directories( ${PROJECT_NAME} PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${CMAKE_CURRENT_SOURCE_DIR}/calib_io/include
		${CMAKE_CURRENT_SOURCE_DIR}/bev_stitch/include
	)

	# C++17 structured bindings and parallel algorithms
	target_compile_features( ${PROJECT_NAME} PUBLIC cxx_std_17 )

	# Cross-platform compiler settings
	if(MSVC)
		# M_PI, min/max macros, UTF-8 source literals (Chinese UI strings)
		target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES NOMINMAX)
		target_compile_options(${PROJECT_NAME} PRIVATE /utf-8 /bigobj)
	elseif(APPLE)
		target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
	else()
		target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
		if(UNIX)
			# std::ifstream wide-path fallback not needed; keep filesystem off plugins
		endif()
	endif()

	# Threading + Qt Concurrent (slider async load, background bag index)
	target_link_libraries(${PROJECT_NAME}
		3rdparty_threads
		Qt::Concurrent
		Qt::Core
		MCALIB_CALIB_IO
		MCALIB_BEV_STITCH
	)

	# Eigen3 (use project's 3rdparty target)
	target_link_libraries(${PROJECT_NAME} 3rdparty_eigen3)

	# OpenCV (use project's 3rdparty target, requires BUILD_OPENCV=ON)
	if(BUILD_OPENCV)
		add_dependencies(${PROJECT_NAME} 3rdparty_opencv)
		target_link_libraries(${PROJECT_NAME} 3rdparty_opencv)
		target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
	else()
		message(WARNING "qManualCalib requires BUILD_OPENCV=ON, please enable it.")
		set(BUILD_OPENCV ON)
	endif()

	# LZ4/BZ2 and bag/config I/O live in calib_io/ — see calib_io/CMakeLists.txt
	# BEV stitch lives in bev_stitch/ — see bev_stitch/CMakeLists.txt

	if(MCALIB_FFMPEG_FOUND)
		mcalib_export_ffmpeg_runtime(${ACloudViewer_DEST_FOLDER})
	endif()

	option(MCALIB_BUILD_TESTS "Build qManualCalib test executables" OFF)
	option(MCALIB_BUILD_TOOLS "Build qManualCalib CLI tools" OFF)
	option(MCALIB_BUILD_EXPORT_BEV_CLI
		"Deprecated alias: enables MCALIB_BUILD_TOOLS" OFF)
	# Backward compatibility with the previous export-only option.
	if(MCALIB_BUILD_EXPORT_BEV_CLI)
		set(MCALIB_BUILD_TOOLS ON)
	endif()

	if(MCALIB_BUILD_TESTS)
		add_executable(test_bag_reader
			${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bag_reader.cpp
		)
		target_compile_features(test_bag_reader PUBLIC cxx_std_17)
		target_compile_definitions(test_bag_reader PRIVATE
			MCALIB_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
		target_link_libraries(test_bag_reader PRIVATE
			MCALIB_CALIB_IO
			CVCoreLib 3rdparty_threads 3rdparty_opencv)
		if(NOT MSVC)
			target_compile_options(test_bag_reader PRIVATE -fPIC)
		endif()
	endif()

	if(MCALIB_BUILD_TOOLS)
		add_subdirectory(tools)
	endif()
endif()
