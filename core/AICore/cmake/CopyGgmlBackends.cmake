# Copy ggml runtime libraries to an application output directory.
# Invoked as a POST_BUILD / packaging script so globs run after ExternalProject.
#
# Modes (pass via -D):
#   COPY_CORE_SHARED=ON — copy libggml + libggml-base without duplicating symlink
#                         targets (configure_file/wheel/archive flattening fix)
#   COPY_BACKEND_MODULES=ON — copy CPU module(s) + optional GPU modules
#   REQUESTED_FILES_PIPE — legacy explicit backend module list (| separated)
#   (default) — legacy glob of all libggml* in GGML_BACKEND_DIR
#
# COPY_BACKEND_MODULES variables:
#   CPU_ALL_VARIANTS   ON: glob libggml-cpu-* from GGML_BACKEND_DIR
#   GPU_FILES_PIPE     optional | separated GPU backend filenames
#
# Common variables:
#   GGML_BACKEND_DIR  source directory
#   DEST_DIR          destination directory
#   LIB_PREFIX        e.g. "lib"
#   LIB_SUFFIX        e.g. ".so" or ".dylib"

cmake_minimum_required(VERSION 3.16)

function(_ggml_list_contains list value out_var)
    set(_found FALSE)
    foreach(_item IN LISTS ${list})
        if("${_item}" STREQUAL "${value}")
            set(_found TRUE)
            break()
        endif()
    endforeach()
    set(${out_var} "${_found}" PARENT_SCOPE)
endfunction()

function(ggml_copy_core_shared_libs src_dir dest_dir lib_prefix lib_suffix)
    file(MAKE_DIRECTORY "${dest_dir}")
    string(REPLACE "." "\\." _lib_suffix_regex "${lib_suffix}")

    file(GLOB _stale
        "${dest_dir}/${lib_prefix}ggml${lib_suffix}*"
        "${dest_dir}/${lib_prefix}ggml-base${lib_suffix}*")
    if(_stale)
        file(REMOVE ${_stale})
    endif()

    set(_versioned_files "")
    foreach(_base IN ITEMS ggml ggml-base)
        file(GLOB _matches
            "${src_dir}/${lib_prefix}${_base}${lib_suffix}*")
        foreach(_path IN LISTS _matches)
            if(NOT EXISTS "${_path}")
                continue()
            endif()
            get_filename_component(_name "${_path}" NAME)
            if(_name MATCHES "${_lib_suffix_regex}\\.[0-9]+\\.[0-9]+\\.[0-9]+$")
                if(IS_SYMLINK "${_path}")
                    get_filename_component(_path "${_path}" REALPATH)
                    get_filename_component(_name "${_path}" NAME)
                endif()
                _ggml_list_contains(_versioned_files "${_path}" _already)
                if(_already)
                    continue()
                endif()
                list(APPEND _versioned_files "${_path}")
            endif()
        endforeach()
    endforeach()

    if(NOT _versioned_files)
        message(FATAL_ERROR
            "CopyGgmlBackends: no versioned ggml core libs in ${src_dir}")
    endif()

    foreach(_path IN LISTS _versioned_files)
        get_filename_component(_name "${_path}" NAME)
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_path}" "${dest_dir}/${_name}")
        message(STATUS "CopyGgmlBackends: ${_name}")

        if(_name MATCHES "^${lib_prefix}(ggml-base|ggml)${_lib_suffix_regex}\\.([0-9]+)\\.[0-9]+\\.[0-9]+$")
            set(_soname_major "${CMAKE_MATCH_2}")
            set(_base_name "${CMAKE_MATCH_1}")
            set(_soname "${lib_prefix}${_base_name}${lib_suffix}.${_soname_major}")
            set(_link_name "${lib_prefix}${_base_name}${lib_suffix}")
            foreach(_alias IN ITEMS "${_soname}" "${_link_name}")
                if(EXISTS "${dest_dir}/${_alias}")
                    file(REMOVE "${dest_dir}/${_alias}")
                endif()
            endforeach()
            execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                "${_name}" "${dest_dir}/${_soname}")
            execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                "${_soname}" "${dest_dir}/${_link_name}")
            message(STATUS "CopyGgmlBackends: ${_soname} -> ${_name}")
            message(STATUS "CopyGgmlBackends: ${_link_name} -> ${_soname}")
        endif()
    endforeach()
endfunction()

function(ggml_remove_stale_backend_modules dest_dir lib_prefix lib_suffix)
    file(GLOB _stale_cpu
        "${dest_dir}/${lib_prefix}ggml-cpu*${lib_suffix}")
    if(_stale_cpu)
        file(REMOVE ${_stale_cpu})
    endif()
    foreach(_backend IN ITEMS blas cuda metal vulkan opencl sycl rpc)
        set(_path "${dest_dir}/${lib_prefix}ggml-${_backend}${lib_suffix}")
        if(EXISTS "${_path}")
            file(REMOVE "${_path}")
        endif()
    endforeach()
endfunction()

function(ggml_copy_backend_modules src_dir dest_dir lib_prefix lib_suffix
         cpu_all_variants gpu_files_pipe)
    file(MAKE_DIRECTORY "${dest_dir}")
    ggml_remove_stale_backend_modules(
        "${dest_dir}" "${lib_prefix}" "${lib_suffix}")

    if(cpu_all_variants)
        file(GLOB _cpu_mods
            "${src_dir}/${lib_prefix}ggml-cpu*${lib_suffix}")
        if(NOT _cpu_mods)
            message(FATAL_ERROR
                "CopyGgmlBackends: no CPU backend modules found in ${src_dir}")
        endif()
        foreach(_path IN LISTS _cpu_mods)
            if(IS_SYMLINK "${_path}")
                continue()
            endif()
            get_filename_component(_name "${_path}" NAME)
            execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_path}" "${dest_dir}/${_name}")
            message(STATUS "CopyGgmlBackends: ${_name}")
        endforeach()
    else()
        set(_cpu_name "${lib_prefix}ggml-cpu${lib_suffix}")
        set(_cpu_path "${src_dir}/${_cpu_name}")
        if(NOT EXISTS "${_cpu_path}")
            message(FATAL_ERROR
                "CopyGgmlBackends: required CPU backend missing: ${_cpu_path}")
        endif()
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_cpu_path}" "${dest_dir}/${_cpu_name}")
        message(STATUS "CopyGgmlBackends: ${_cpu_name}")
    endif()

    if(gpu_files_pipe)
        string(REPLACE "|" ";" _gpu_files "${gpu_files_pipe}")
        list(REMOVE_DUPLICATES _gpu_files)
        foreach(_name IN LISTS _gpu_files)
            if(_name STREQUAL "")
                continue()
            endif()
            set(_source "${src_dir}/${_name}")
            if(NOT EXISTS "${_source}")
                message(FATAL_ERROR
                    "CopyGgmlBackends: configured GPU backend is missing: ${_source}")
            endif()
            execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_source}" "${dest_dir}/${_name}")
            message(STATUS "CopyGgmlBackends: ${_name}")
        endforeach()
    endif()
endfunction()

if(NOT EXISTS "${GGML_BACKEND_DIR}")
    message(FATAL_ERROR
        "CopyGgmlBackends: source dir does not exist: ${GGML_BACKEND_DIR}")
endif()

if(COPY_CORE_SHARED)
    ggml_copy_core_shared_libs(
        "${GGML_BACKEND_DIR}" "${DEST_DIR}" "${LIB_PREFIX}" "${LIB_SUFFIX}")
    return()
endif()

if(COPY_BACKEND_MODULES)
    if(CPU_ALL_VARIANTS)
        set(_cpu_all ON)
    else()
        set(_cpu_all OFF)
    endif()
    ggml_copy_backend_modules(
        "${GGML_BACKEND_DIR}" "${DEST_DIR}" "${LIB_PREFIX}" "${LIB_SUFFIX}"
        "${_cpu_all}" "${GPU_FILES_PIPE}")
    return()
endif()

if(CPU_ALL_VARIANTS)
    file(GLOB _cpu_backends
        "${GGML_BACKEND_DIR}/${LIB_PREFIX}ggml-cpu*${LIB_SUFFIX}")
    if(NOT _cpu_backends)
        message(FATAL_ERROR
            "CopyGgmlBackends: no CPU backend modules found in ${GGML_BACKEND_DIR}")
    endif()
elseif(REQUIRED_FILE AND NOT EXISTS "${GGML_BACKEND_DIR}/${REQUIRED_FILE}")
    message(FATAL_ERROR
        "CopyGgmlBackends: required runtime file is missing: "
        "${GGML_BACKEND_DIR}/${REQUIRED_FILE}")
endif()

if(REQUESTED_FILES_PIPE)
    string(REPLACE "|" ";" _requested_files "${REQUESTED_FILES_PIPE}")
    file(MAKE_DIRECTORY "${DEST_DIR}")
    ggml_remove_stale_backend_modules("${DEST_DIR}" "${LIB_PREFIX}" "${LIB_SUFFIX}")
    foreach(_name IN LISTS _requested_files)
        set(_source "${GGML_BACKEND_DIR}/${_name}")
        if(NOT EXISTS "${_source}")
            message(FATAL_ERROR
                "CopyGgmlBackends: configured backend is missing: ${_source}")
        endif()
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_source}" "${DEST_DIR}/${_name}")
        message(STATUS "CopyGgmlBackends: ${_name}")
    endforeach()
    return()
endif()

file(GLOB _ggml_libs "${GGML_BACKEND_DIR}/${LIB_PREFIX}ggml*${LIB_SUFFIX}*")
if(NOT _ggml_libs)
    message(FATAL_ERROR
        "CopyGgmlBackends: no ggml libs found in ${GGML_BACKEND_DIR}")
endif()

file(MAKE_DIRECTORY "${DEST_DIR}")

set(_seen_realpaths "")
foreach(_lib ${_ggml_libs})
    if(IS_SYMLINK "${_lib}")
        continue()
    endif()
    get_filename_component(_real "${_lib}" REALPATH)
    _ggml_list_contains(_seen_realpaths "${_real}" _already)
    if(_already)
        continue()
    endif()
    list(APPEND _seen_realpaths "${_real}")
    get_filename_component(_name "${_lib}" NAME)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_lib}" "${DEST_DIR}/${_name}"
    )
    message(STATUS "CopyGgmlBackends: ${_name}")
endforeach()
