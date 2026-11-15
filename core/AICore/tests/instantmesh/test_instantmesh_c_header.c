// SPDX-License-Identifier: MIT
// Compile as C11 to verify public headers do not leak Qt/STL/C++ types.
#include "aicore/instantmesh_capi.h"

_Static_assert(sizeof(((aicore_instantmesh_model_paths*)0)->struct_size) == 4,
               "struct_size must be uint32_t");
_Static_assert(sizeof(((aicore_instantmesh_generate_params*)0)->abi_version) == 4,
               "abi_version must be uint32_t");

int main(void) {
    aicore_instantmesh_model_paths paths = AICORE_INSTANTMESH_MODEL_PATHS_INIT;
    aicore_instantmesh_generate_params params = AICORE_INSTANTMESH_GENERATE_PARAMS_INIT;
    return paths.struct_size != sizeof(paths) ||
           params.struct_size != sizeof(params) ||
           paths.abi_version != AICORE_INSTANTMESH_ABI_VERSION ||
           params.abi_version != (uint32_t)aicore_instantmesh_abi_version();
}
