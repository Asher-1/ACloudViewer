// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <filesystem>
#include <memory>

#include "scene/database.h"

namespace colmap {

// Can be used to construct temporary in-memory database.
constexpr inline char kInMemorySqliteDatabasePath[] = ":memory:";

std::shared_ptr<Database> OpenSqliteDatabase(const std::filesystem::path& path);

}  // namespace colmap
