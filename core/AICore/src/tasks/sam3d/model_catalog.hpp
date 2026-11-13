// Internal declarations behind the aicore_sam3d_model_* C ABI (see
// sam3d_capi.h for the public surface and the catalog ownership contract).
#pragma once

#include <string>

#include "aicore/sam3d_capi.h"

namespace aicore {

const char* sam3d_model_download_base();
const char* sam3d_model_license_note();
int sam3d_model_count();
const aicore_sam3d_model_entry* sam3d_model_at(int index);
const aicore_sam3d_model_entry* sam3d_model_by_filename(const char* filename);
std::string sam3d_model_cache_dir();

}  // namespace aicore
