// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/* AICore capability-module contract (extension guide).
 *
 * Each inference capability is a self-contained unit:
 *
 *   include/aicore/<cap>_capi.h   Stable extern-C surface: aicore_<cap>_*
 *   include/aicore/<cap>_*.h      Optional helpers (Qt, GGUF keys, …)
 *   src/<cap>/                    C++ engine in namespace aicore::<cap>
 *   tests/<cap>/                  White-box tests (AICore_BUILD_TESTS)
 *
 * Naming rules (all modules follow the same pattern):
 *   C API:     aicore_<cap>_<verb>   e.g. aicore_depth_load,
 * aicore_gaussian_run Context:   aicore_<cap>_ctx C++ NS:    aicore::<cap>
 *   Export:    AICORE_CAPI (C) / AICORE_CXX_API (C++ classes)
 *
 * Adding a module:
 *   1. Implement under src/<cap>/ in namespace aicore::<cap>.
 *   2. Add include/aicore/<cap>_capi.h with AICORE_CAPI entry points.
 *   3. Register sources in core/AICore/CMakeLists.txt.
 *   4. #include the new capi from aicore/aicore.h (optional umbrella).
 *
 * Consumers link libAICore and include only headers under include/aicore/.
 */

#include "aicore/backend_capi.h"
#include "aicore/depth_capi.h"
#include "aicore/depth_image.h"
#include "aicore/export.h"
#include "aicore/gaussian_capi.h"
