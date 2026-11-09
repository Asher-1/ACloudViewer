#pragma once

#include <string>

namespace aicore {

// Delegates to the shared CVCore filesystem asset-root policy (UTF-8).
std::string locate_data_root();

// UTF-8 {locate_data_root()}/extract/{sub_dir}, normalized with forward
// slashes.
std::string extract_model_dir(const char* sub_dir);

}  // namespace aicore
