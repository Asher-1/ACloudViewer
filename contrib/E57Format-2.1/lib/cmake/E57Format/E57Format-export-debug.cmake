#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "E57Format" for configuration "Debug"
set_property(TARGET E57Format APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(E57Format PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/E57Format-d.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS E57Format )
list(APPEND _IMPORT_CHECK_FILES_FOR_E57Format "${_IMPORT_PREFIX}/lib/E57Format-d.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
