message(STATUS "SOURCE_DIR: ${SOURCE_DIR}; 
                DESTINATION_DIR: ${DESTINATION_DIR}; 
                LIB_FILTERS: ${LIB_FILTERS};
                LIBRARY_SUFFIX: ${LIBRARY_SUFFIX}")
set(TARGET_LIB_LIST "")
foreach(LIB ${LIB_FILTERS})
  file(GLOB LIB_FILES "${SOURCE_DIR}/*${LIB}${LIBRARY_SUFFIX}")
  if(NOT LIB_FILES STREQUAL "")
    list(APPEND TARGET_LIB_LIST ${LIB_FILES})
  endif()
endforeach()

message(STATUS "Start copy libraries: ${TARGET_LIB_LIST}")
foreach(FILE ${TARGET_LIB_LIST})
  if (EXISTS "${FILE}")
    file(COPY ${FILE} DESTINATION ${DESTINATION_DIR})
  endif()
endforeach()