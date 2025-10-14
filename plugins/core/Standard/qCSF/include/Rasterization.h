// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "Cloth.h"
#include "wlPointCloud.h"

#define SQUARE_DIST(x1, y1, x2, y2) \
    (((x1) - (x2)) * ((x1) - (x2)) + ((y1) - (y2)) * ((y1) - (y2)))

class Rasterization {
public:
    // for a cloth particle, if no corresponding lidar point are found.
    // the heightval are set as its neighbor's
    double static findHeightValByNeighbor(Particle *p, Cloth &cloth);
    double static findHeightValByScanline(Particle *p, Cloth &cloth);

    // 对点云进行最临近搜索，寻找周围最近的N个点  避免求交运算
    static bool RasterTerrain(Cloth &cloth,
                              const wl::PointCloud &pc,
                              std::vector<double> &heightVal,
                              unsigned KNN = 1);
};
