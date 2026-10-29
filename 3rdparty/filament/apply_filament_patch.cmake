# Keep the pinned Filament sources compatible with current C++ toolchains.
if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "Filament patch requires SOURCE_DIR")
endif()

set(_memalign "${SOURCE_DIR}/libs/utils/include/utils/memalign.h")
set(_sampler "${SOURCE_DIR}/libs/image/src/ImageSampler.cpp")
file(READ "${_memalign}" _memalign_source)
file(READ "${_sampler}" _sampler_source)

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
    file(WRITE "${_memalign}" "${_memalign_patched}")
endif()

# Preserve the existing ImageSampler macro fix without BSD/GNU sed differences.
string(REPLACE "M_PIf" "M_PI_f" _sampler_patched "${_sampler_source}")
if(NOT _sampler_patched STREQUAL _sampler_source)
    file(WRITE "${_sampler}" "${_sampler_patched}")
endif()
