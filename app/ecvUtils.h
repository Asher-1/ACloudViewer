// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

class QString;

namespace ecvUtils {
//! Display a warning or error for locked verts
void DisplayLockedVerticesWarning(const QString &meshName, bool displayAsError);
}  // namespace ecvUtils
