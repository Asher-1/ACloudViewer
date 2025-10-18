# This file defines the variables
# ${PROJECT_NAME}_BUILD_TAG

# calculate the tag
set(T_ ${CMAKE_SYSTEM_PROCESSOR})
if (CMAKE_CL_64)
    set(T_ ${T_}_64)
endif (CMAKE_CL_64)
string(TOLOWER ${CMAKE_SYSTEM_NAME} T1_)
set(T_ ${T_}-${T1_})
if (MSVC90)
    set(T1_ "-vc90")
elseif (MSVC10)
    set(T1_ "-vc100")
elseif (MSVC80)
    set(T1_ "-vc80")
elseif (MSVC71)
    set(T1_ "-vc711")
elseif (MSVC70)
    set(T1_ "-vc7")
elseif (MINGW)
    set(T1_ "-mgw")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
        OUTPUT_VARIABLE T2_
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX REPLACE "([0-9])\\.([0-9])(\\.[0-9])?" "\\1\\2" T2_ ${T2_})
    set(T1_ ${T1_}${T2_})
elseif (CMAKE_COMPILER_IS_GNUCXX)
    set(T1_ "-gcc")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
        OUTPUT_VARIABLE T2_
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX REPLACE "([0-9])\\.([0-9])(\\.[0-9])?" "\\1\\2" T2_ ${T2_})
    set(T1_ ${T1_}${T2_})
else()
    set(T1_)
endif()
set(T_ ${T_}${T1_})
set(${PROJECT_NAME}_BUILD_TAG ${T_})
