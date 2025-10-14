// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_MVS_DEPTH_MAP_H_
#define COLMAP_SRC_MVS_DEPTH_MAP_H_

#include <string>
#include <vector>

#include "mvs/mat.h"
#include "util/bitmap.h"

namespace colmap {
namespace mvs {

class DepthMap : public Mat<float> {
public:
    DepthMap();
    DepthMap(const size_t width,
             const size_t height,
             const float depth_min,
             const float depth_max);
    DepthMap(const Mat<float>& mat,
             const float depth_min,
             const float depth_max);

    inline float GetDepthMin() const;
    inline float GetDepthMax() const;

    inline float Get(const size_t row, const size_t col) const;

    void Rescale(const float factor);
    void Downsize(const size_t max_width, const size_t max_height);

    Bitmap ToBitmap(const float min_percentile,
                    const float max_percentile) const;

private:
    float depth_min_ = -1.0f;
    float depth_max_ = -1.0f;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

float DepthMap::GetDepthMin() const { return depth_min_; }

float DepthMap::GetDepthMax() const { return depth_max_; }

float DepthMap::Get(const size_t row, const size_t col) const {
    return data_.at(row * width_ + col);
}

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_DEPTH_MAP_H_
