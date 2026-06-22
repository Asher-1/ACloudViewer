// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

namespace Visualization {

/// Registers ecvViewManager's typed per-view signal relay (vtkGLView /
/// ecvDisplayTools PMF connections). Safe to call more than once.
void registerViewManagerTypedRelay();

/// Bridges ecvDisplayTools result signals (picking, camera state, entity
/// selection) to ecvViewManager so that app-level consumers receive them.
/// Must be called after ecvDisplayTools is constructed. Idempotent.
void installDisplayToolsRelay();

}  // namespace Visualization
