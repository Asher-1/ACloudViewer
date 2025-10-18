// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace cloudViewer {

int CleanDatabase(const std::string& database_path,
                  const std::string& clean_type);

int CreateDatabase(const std::string& database_path);

int MergeDatabase(const std::string& database_path1,
                  const std::string& database_path2,
                  const std::string& merged_database_path);

}  // namespace cloudViewer
