set(SOURCE_DIR "${INPUT_DIR}")
message(STATUS "SOURCE_DIR = ${SOURCE_DIR}")
file(GLOB_RECURSE cmake_files
    "${SOURCE_DIR}/CMakeLists.txt"
    "${SOURCE_DIR}/*.cmake"
)
foreach(f ${cmake_files})
    message(STATUS "Fixing: ${f}")
    file(READ "${f}" content)
    string(REGEX REPLACE "cmake_minimum_required\\([ \\t]*VERSION[ \\t]*[0-9.]+([ \\t]*\\.\\.\\.[0-9.]+)?[ \\t]*\\)[ \t\r\n]*" "" content "${content}")
    file(WRITE "${f}" "cmake_minimum_required(VERSION 3.10)\n${content}")
endforeach()