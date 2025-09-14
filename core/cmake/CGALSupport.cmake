# ------------------------------------------------------------------------------
# CGAL+CMake support for ACloudViewer
# ------------------------------------------------------------------------------

set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
set(CGAL_DATA_DIR "unused")
if (BUILD_WITH_CONDA)
	if (WIN32)
		set(CGAL_DIR "${CONDA_PREFIX}/Library/lib/cmake/CGAL")
	else()
		set(CGAL_DIR "${CONDA_PREFIX}/lib/cmake/CGAL")
		set(GMPXX_INCLUDE_DIR "${CONDA_PREFIX}/include")
		set(GMP_INCLUDE_DIR "${CONDA_PREFIX}/include")
		set(MPFR_INCLUDE_DIR "${CONDA_PREFIX}/include")
	endif()
endif()
find_package( CGAL QUIET COMPONENTS Core ) # implies findGMP

if (CGAL_FOUND)
	if(${CGAL_MAJOR_VERSION} LESS 4)
		message(SEND_ERROR "CC Lib requires at least CGAL 4.3")
	endif()
	if(${CGAL_MAJOR_VERSION} EQUAL 4 AND CGAL_MINOR_VERSION LESS 3)
		message(SEND_ERROR "CC Lib requires at least CGAL 4.3")
	endif()

	# Take care of GMP and MPFR DLLs on Windows!
	if( WIN32 )
		# We need to get rid of CGAL CXX flags
		set(CGAL_DONT_OVERRIDE_CMAKE_FLAGS ON CACHE INTERNAL "override CGAL flags" FORCE)

		include( ${CGAL_USE_FILE} )
		include_directories(${CGAL_INCLUDE_DIR})

		# message(${GMP_LIBRARIES})
		list(GET GMP_LIBRARIES 0 FIRST_GMP_LIB_FILE)
		get_filename_component(GMP_LIB_FOLDER ${FIRST_GMP_LIB_FILE} DIRECTORY)
		message(STATUS "GMP_LIB_FOLDER: ${GMP_LIB_FOLDER}")

		file( GLOB GMP_DLL_FILES ${GMP_LIB_FOLDER}/*.dll )
		foreach( dest ${INSTALL_DESTINATIONS} )
			cloudViewer_install_files( "${GMP_DLL_FILES}" ${dest} ) # Mind the quotes!
		endforeach()
	endif()
else()
	message(SEND_ERROR "Could not find CGAL")
endif()
