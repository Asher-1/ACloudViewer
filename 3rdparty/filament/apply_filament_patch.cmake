# Keep the pinned Filament sources compatible with current C++ toolchains.
if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "Filament patch requires SOURCE_DIR")
endif()

set(_memalign "${SOURCE_DIR}/libs/utils/include/utils/memalign.h")
set(_sampler "${SOURCE_DIR}/libs/image/src/ImageSampler.cpp")
set(_zutil "${SOURCE_DIR}/third_party/libz/zutil.h")
file(READ "${_memalign}" _memalign_source)
file(READ "${_sampler}" _sampler_source)
file(READ "${_zutil}" _zutil_source)
set(_memalign_patched "${_memalign_source}")

# <stddef.h> need not declare std::ptrdiff_t. Do not rely on transitive
# inclusion from <type_traits>, which changed in the macOS 15 libc++ headers.
if(NOT _memalign_source MATCHES "#include <cstddef>")
    string(FIND "${_memalign_source}" "#include <type_traits>" _anchor)
    if(_anchor EQUAL -1)
        message(FATAL_ERROR "Filament memalign.h patch anchor is missing")
    endif()
    string(REPLACE "#include <type_traits>"
                   "#include <cstddef>\n#include <type_traits>"
                   _memalign_patched "${_memalign_source}")
endif()

# Modern Apple systems provide fdopen. The classic Mac fallback would replace
# its SDK declaration with NULL; zutil.h already has a separate __APPLE__ branch.
set(_zutil_old "#if defined(MACOS) || defined(TARGET_OS_MAC)")
set(_zutil_new "#if (defined(MACOS) || defined(TARGET_OS_MAC)) && !defined(__APPLE__)")
string(FIND "${_zutil_source}" "${_zutil_old}" _zutil_anchor)
string(FIND "${_zutil_source}" "${_zutil_new}" _zutil_fixed)
if(_zutil_anchor EQUAL -1 AND _zutil_fixed EQUAL -1)
    message(FATAL_ERROR "Filament zutil.h patch anchor is missing")
endif()
string(REPLACE "${_zutil_old}" "${_zutil_new}" _zutil_patched "${_zutil_source}")

# Validate all inputs and anchors before writing any of the patched sources.
if(NOT _memalign_patched STREQUAL _memalign_source)
    file(WRITE "${_memalign}" "${_memalign_patched}")
endif()
if(NOT _zutil_patched STREQUAL _zutil_source)
    file(WRITE "${_zutil}" "${_zutil_patched}")
endif()

# Preserve the existing ImageSampler macro fix without BSD/GNU sed differences.
string(REPLACE "M_PIf" "M_PI_f" _sampler_patched "${_sampler_source}")
if(NOT _sampler_patched STREQUAL _sampler_source)
    file(WRITE "${_sampler}" "${_sampler_patched}")
endif()
