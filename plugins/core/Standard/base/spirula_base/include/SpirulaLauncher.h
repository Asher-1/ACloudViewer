// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>

namespace spirula {

//! Launches the official Spirula Studio GUI detached from this process.
//! The plugin does not own the training process: outputs cross back as files
//! (run directory / splat.ply / mesh.ply), which is what keeps upstream sync
//! down to a tag bump.
//! Returns false and fills errorOut when the binary cannot be started.
bool launchGui(const QString& exePath, QString* errorOut = nullptr);

}  // namespace spirula
