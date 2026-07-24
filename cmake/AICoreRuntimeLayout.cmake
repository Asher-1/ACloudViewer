# AICoreRuntimeLayout.cmake — ggml dynamic-backend packaging paths (internal)
#
# ggml.cmake exports these after ExternalProject configure. They describe where
# libggml*.so / ggml-cpu.so / ggml-vulkan.so live on disk — not generic CloudViewer
# libs. Wheel/install scripts copy from these dirs because AICore loads backends
# at runtime (AICORE_BACKEND_DL).
#
# GGML_MODULE_SUFFIX is NOT CMAKE_SHARED_LIBRARY_SUFFIX:
#   Linux   — both are ".so" (same)
#   Windows — both are ".dll" (same)
#   macOS   — SHARED libs use ".dylib" (CMAKE_SHARED_LIBRARY_SUFFIX), but ggml
#             backend MODULE libs are built as ".so" because ggml-backend-reg
#             dlopen() searches ".so" on all UNIX. Using CMAKE_SHARED_LIBRARY_SUFFIX
#             on macOS would miss libggml-cpu.so / libggml-metal.so at copy/install.
#
# When AICore is OFF, ggml.cmake may not run; this fallback matches ggml.cmake
# (_GGML_MODULE_SUFFIX: .dll on WIN32, else .so). After ggml configure, cache
# GGML_MODULE_SUFFIX from ggml.cmake takes precedence.

if(NOT DEFINED GGML_MODULE_SUFFIX)
    if(WIN32)
        set(GGML_MODULE_SUFFIX ".dll")
    else()
        set(GGML_MODULE_SUFFIX ".so")
    endif()
endif()
