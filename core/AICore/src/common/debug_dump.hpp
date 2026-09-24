// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Sanctioned read-side environment access for debug-only dump hooks.
//
// AICore logic must be driven by explicit options/APIs, never by the process
// environment (enforced by tests/check_no_env_getenv.sh). This file is the
// single whitelisted read-side exception: it exposes the debug dump paths as
// plain accessors so task modules stay free of any environment mechanism
// (interface-only rule). The variables below enable binary tensor dumps used
// to debug the YOLO SAVPE visual-prompt path; they are read at call time and
// never influence any result value.
//
// Task modules call these accessors — the raw C env API stays in this
// file's implementation.

#pragma once

namespace aicore {
namespace debug {

/** Path from AICORE_SAVPE_DUMP (nullptr when unset): base path for the YOLO
 *  SAVPE readback dumps ("<path>.bin", "<path>_x.bin", "<path>_fpn0.bin",
 *  "<path>_cv20.bin", ...). */
const char* savpe_dump_path();

/** Path from AICORE_SAVPE_DUMP_MASK (nullptr when unset): raw visual-prompt
 *  mask buffer dump written right after rasterization. */
const char* savpe_mask_dump_path();

/** True when AICORE_SAVPE_DEBUG is set: enables the [savpe-dbg] rasterization
 *  rect diagnostics on stderr (off by default; the run is never affected). */
bool savpe_debug_enabled();

}  // namespace debug
}  // namespace aicore
