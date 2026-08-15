// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <fstream>
#include <string>

class QString;

namespace mcalib {

/// Convert a Qt path to UTF-8 std::string (safe on Windows/macOS/Linux).
std::string pathFromQString(const QString& path);

/// Open a file for binary reading using UTF-8 paths on all platforms.
bool openInputFile(std::ifstream& stream, const std::string& utf8_path);

/// Open a file for text/binary writing using UTF-8 paths on all platforms.
bool openOutputFile(std::ofstream& stream, const std::string& utf8_path);

}  // namespace mcalib
