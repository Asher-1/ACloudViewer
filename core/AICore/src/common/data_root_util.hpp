#pragma once

#include <string>

namespace aicore {

// Mirrors cloudViewer::data::LocateDataRoot() in libs/cloudViewer/data/Dataset.cpp:
//   CLOUDVIEWER_DATA_ROOT, else GetHomeDirectory() + "/cloudViewer_data"
std::string locate_data_root();

// {locate_data_root()}/extract/{sub_dir}
std::string extract_model_dir(const char* sub_dir);

}  // namespace aicore
