// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RenderOptions.h"

#include "util/logging.h"

namespace cloudViewer {
using namespace colmap;

bool RenderOptions::Check() const {
    CHECK_OPTION_GE(min_track_len, 0);
    CHECK_OPTION_GE(max_error, 0);
    CHECK_OPTION_GT(refresh_rate, 0);
    CHECK_OPTION(projection_type == ProjectionType::PERSPECTIVE ||
                 projection_type == ProjectionType::ORTHOGRAPHIC);
    return true;
}

}  // namespace cloudViewer
