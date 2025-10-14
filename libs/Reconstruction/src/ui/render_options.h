// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_UI_RENDER_OPTIONS_H_
#define COLMAP_SRC_UI_RENDER_OPTIONS_H_

#include <iostream>

namespace colmap {

struct RenderOptions {
    enum ProjectionType {
        PERSPECTIVE,
        ORTHOGRAPHIC,
    };

    // Minimum track length for a point to be rendered.
    int min_track_len = 3;

    // Maximum error for a point to be rendered.
    double max_error = 2;

    // The rate of registered images at which to refresh.
    int refresh_rate = 1;

    // Whether to automatically adjust the refresh rate. The bigger the
    // reconstruction gets, the less frequently the scene is rendered.
    bool adapt_refresh_rate = true;

    // Whether to visualize image connections.
    bool image_connections = false;

    // The projection type of the renderer.
    int projection_type = ProjectionType::PERSPECTIVE;

    bool Check() const;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_RENDER_OPTIONS_H_
