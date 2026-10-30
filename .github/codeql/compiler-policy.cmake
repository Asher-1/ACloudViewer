# 仅用于显式启用的 CodeQL 扫描，不改变生产构建的优化策略。
include_guard(GLOBAL)

if(CMAKE_VERSION VERSION_LESS 3.19)
    message(FATAL_ERROR "CodeQL compiler policy requires CMake 3.19 or newer")
endif()
if(NOT CLOUDVIEWER_CODEQL_OPTIMIZATION MATCHES "^O(1|3)$")
    message(FATAL_ERROR "CodeQL optimization must be O1 or O3")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "CodeQL compiler policy requires a Linux build")
endif()
foreach(language C CXX)
    if(CMAKE_${language}_COMPILER_ID AND
       NOT CMAKE_${language}_COMPILER_ID MATCHES "^(GNU|Clang)$")
        message(FATAL_ERROR "CodeQL compiler policy requires GNU or Clang")
    endif()
endforeach()

# 主工程会自动探测 ccache；扫描必须真实编译，不能重放缓存对象跳过提取。
set(CMAKE_C_COMPILER_LAUNCHER "")
set(CMAKE_CXX_COMPILER_LAUNCHER "")

# O(目录数 + 目标数)，只保存当前目录的列表；不修改源码、链接或导出属性。
function(cloudviewer_codeql_apply_options directory)
    # Draco 在 project() 之后才设置默认 Release，验证也必须等配置结束。
    if(directory STREQUAL CMAKE_SOURCE_DIR AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        message(FATAL_ERROR "CodeQL compiler policy requires a Release build")
    endif()
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        get_target_property(type "${target}" TYPE)
        if(type MATCHES "^(EXECUTABLE|STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY)$")
            # 配置末尾追加目标选项，抵抗 Embree 等上游重置 Release flags。
            # 保留 NDEBUG、ISA、PIC、OpenMP 和数值选项，只选择代码生成优化级别。
            target_compile_options("${target}" PRIVATE
                "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:-$CACHE{CLOUDVIEWER_CODEQL_OPTIMIZATION}>")
            set_target_properties("${target}" PROPERTIES
                C_COMPILER_LAUNCHER "" CXX_COMPILER_LAUNCHER "")
        endif()
    endforeach()
    get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        cloudviewer_codeql_apply_options("${child}")
    endforeach()
endfunction()

# project() 入口早于上游 flags 重置和目标创建，必须延迟到根目录配置结束。
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
    CALL cloudviewer_codeql_apply_options "${CMAKE_SOURCE_DIR}")
