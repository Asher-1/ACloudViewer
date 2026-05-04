// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

namespace Visualization {

/// Registers ecvViewManager's typed per-view signal relay (ecvGLView /
/// ecvDisplayTools PMF connections). Safe to call more than once.
void registerViewManagerTypedRelay();

}  // namespace Visualization
