// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>

namespace spirula {

//! Locates the Spirula Studio executable at runtime.
//!
//! Probe order (first hit wins; result cached for the session):
//!   1. <app dir>/spirula-cuda  — only when `nvidia-smi` answers; a local
//!      CUDA variant (not part of the distribution, see 3rdparty/spirula)
//!   2. <app dir>/spirula       — the release Vulkan variant
//!   3. the QSPIRULA_BUILD_PATH compile-time fallback (ExternalProject output)
//!   4. PATH (QStandardPaths::findExecutable)
QString locateExecutable();

}  // namespace spirula
