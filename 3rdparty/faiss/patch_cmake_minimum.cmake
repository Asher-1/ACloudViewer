if (NOT DEFINED SOURCE_DIR OR NOT EXISTS "${SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "faiss compatibility patch: SOURCE_DIR/CMakeLists.txt is required")
endif ()

file(READ "${SOURCE_DIR}/CMakeLists.txt" _faiss_cmakelists)
string(REPLACE "cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)"
               "cmake_minimum_required(VERSION 3.16 FATAL_ERROR)"
               _faiss_patched_cmakelists "${_faiss_cmakelists}")
if (_faiss_patched_cmakelists STREQUAL _faiss_cmakelists AND
    NOT _faiss_cmakelists MATCHES "cmake_minimum_required\\(VERSION 3.16")
    message(FATAL_ERROR "faiss compatibility patch: expected v1.14.1 CMake minimum was not found")
endif ()
file(WRITE "${SOURCE_DIR}/CMakeLists.txt" "${_faiss_patched_cmakelists}")
