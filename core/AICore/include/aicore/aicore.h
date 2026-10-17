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
 *   include/aicore/<cap>_*.h      Optional public helpers (for example Qt)
 *   src/<cap>/                    C++ engine in namespace aicore::<cap>
 *   tests/<cap>/                  ABI tests; optional private white-box tests
 *
 * Naming rules (all modules follow the same pattern):
 *   C API:     aicore_<cap>_<verb>   e.g. aicore_depth_load_opts,
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

#include "aicore/aliked_capi.h"
#include "aicore/backend_capi.h"
#include "aicore/deeplsd_capi.h"
#include "aicore/depth_capi.h"
#include "aicore/depth_image.h"
#include "aicore/export.h"
#include "aicore/facedetect_capi.h"
#include "aicore/gaussian_capi.h"
#include "aicore/gkd_capi.h"
#include "aicore/image_view.h"
// NOTE: inference_log.h is intentionally NOT umbrella-included: it depends
// on <CVLog.h> (libs-layer logging), which lean capi test targets and other
// non-CVLog consumers do not have on their include path. Include it
// directly where CVLog is available (plugins, app-side code).
#include "aicore/lightglue_capi.h"
#include "aicore/lingbot_capi.h"
#include "aicore/loma_capi.h"
#include "aicore/model_catalog_capi.h"
#include "aicore/pipeline_timing.h"
#include "aicore/reid_capi.h"
#include "aicore/rfdetr_capi.h"
#include "aicore/rmbg_capi.h"
#include "aicore/runtime_capi.h"
#include "aicore/runtime_raii.h"
#include "aicore/sam3_capi.h"
#include "aicore/sam3d_capi.h"
#include "aicore/trellis_capi.h"
#include "aicore/yolo_capi.h"
