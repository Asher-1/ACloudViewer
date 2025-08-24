// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Do not include this in public facing header.

#pragma once

#include <string>

namespace cloudViewer {
namespace utility {

/// \brief Function to extract files compressed in `.zip` format.
/// \param file_path Path to file. Example: "/path/to/file/file.zip"
/// \param extract_dir Directory path where the file will be extracted to.
void ExtractFromZIP(const std::string& file_path,
                    const std::string& extract_dir);

}  // namespace utility
}  // namespace cloudViewer
