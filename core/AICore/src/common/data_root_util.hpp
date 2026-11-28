#pragma once

#include <string>

namespace aicore {

// Delegates to the shared CVCore filesystem asset-root policy (UTF-8).
std::string locate_data_root();

// User home directory through the same shared CVCore policy (UTF-8).
// Single sanctioned home-resolution interface for task modules that need
// user-layout fallbacks; tasks must not read HOME themselves.
std::string home_dir();

// UTF-8 {locate_data_root()}/extract/{sub_dir}, normalized with forward
// slashes.
std::string extract_model_dir(const char* sub_dir);

}  // namespace aicore
