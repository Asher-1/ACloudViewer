// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace cloudViewer {

int GraphicalUserInterface(const std::string& database_path = "",
                           const std::string& image_path = "",
                           const std::string& import_path = "");

int GenerateProject(const std::string& output_path,
                    const std::string& quality = "high");

}  // namespace cloudViewer
