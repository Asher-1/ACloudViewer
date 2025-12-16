option( PLUGIN_STANDARD_G3POINT "Install example plugin" OFF )

if ( PLUGIN_STANDARD_G3POINT )
	project( QG3POINT_PLUGIN )

	AddPlugin( NAME ${PROJECT_NAME} )

	target_sources(${PROJECT_NAME}
		PRIVATE
		ui/WolmanCustomPlot.ui
	)

	set(QG3POINT_PLUGIN_VERSION "0.6")

	add_subdirectory( include )
	add_subdirectory( src )
	add_subdirectory( ui )

	# target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)  # for mlpack

	# QCUSTOMPLOT
	# Qt5/Qt6 compatibility - use unified Qt:: prefix
	target_link_libraries(${PROJECT_NAME}
		QCustomPlot
		Qt::PrintSupport
	)

	target_link_libraries( ${PROJECT_NAME} 3rdparty_eigen3 3rdparty_nanoflann)

	#================
	# git commit hash
	# Get the current working branch
	execute_process(
		COMMAND git rev-parse --abbrev-ref HEAD
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		OUTPUT_VARIABLE GIT_BRANCH_G3POINT
		OUTPUT_STRIP_TRAILING_WHITESPACE)

	# Get the latest commit hash
	execute_process(
		COMMAND git rev-parse --short HEAD
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		OUTPUT_VARIABLE GIT_COMMIT_HASH_G3POINT
		OUTPUT_STRIP_TRAILING_WHITESPACE)

	# Get the latest commit hash
	execute_process(
		COMMAND git describe
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		OUTPUT_VARIABLE GIT_TAG_G3POINT
		OUTPUT_STRIP_TRAILING_WHITESPACE)

	# Get the current working branch
	execute_process(
		COMMAND git rev-parse --abbrev-ref HEAD
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		OUTPUT_VARIABLE GIT_BRANCH_CC
		OUTPUT_STRIP_TRAILING_WHITESPACE)

	# Get the latest commit hash
	execute_process(
		COMMAND git rev-parse --short HEAD
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		OUTPUT_VARIABLE GIT_COMMIT_HASH_CC
		OUTPUT_STRIP_TRAILING_WHITESPACE)

	message(${PROJECT_NAME} " " GIT_BRANCH_G3POINT " " ${GIT_BRANCH_G3POINT})
	message(${PROJECT_NAME} " " GIT_COMMIT_HASH_G3POINT " " ${GIT_COMMIT_HASH_G3POINT})
	message(${PROJECT_NAME} " " GIT_TAG_G3POINT " " ${GIT_TAG_G3POINT})
	message(${PROJECT_NAME} " " GIT_BRANCH_CC " " ${GIT_BRANCH_CC})
	message(${PROJECT_NAME} " " GIT_COMMIT_HASH_CC " " ${GIT_COMMIT_HASH_CC})
	message(${PROJECT_NAME} " " QG3POINT_VERSION " " ${QG3POINT_PLUGIN_VERSION})

	target_compile_definitions(${PROJECT_NAME} PRIVATE
		GIT_BRANCH_CC="${GIT_BRANCH_CC}"
		GIT_COMMMIT_HASH_CC="${GIT_COMMIT_HASH_CC}"
		PUBLIC GIT_BRANCH_G3POINT="${GIT_BRANCH_G3POINT}"
		GIT_COMMMIT_HASH_G3POINT="${GIT_COMMIT_HASH_G3POINT}"
		GIT_TAG_G3POINT="${GIT_TAG_G3POINT}"
		G3POINT_VERSION="${QG3POINT_PLUGIN_VERSION}"
		)

endif()
