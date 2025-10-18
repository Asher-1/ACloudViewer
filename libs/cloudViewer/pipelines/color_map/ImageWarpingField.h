// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// #include "qGL.h"
#include <IJsonConvertible.h>

namespace cloudViewer {
namespace pipelines {
namespace color_map {

class ImageWarpingField : public cloudViewer::utility::IJsonConvertible {
public:
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    ImageWarpingField();
    ImageWarpingField(int width, int height, int number_of_vertical_anchors);
    void InitializeWarpingFields(int width,
                                 int height,
                                 int number_of_vertical_anchors);
    Eigen::Vector2d QueryFlow(int i, int j) const;
    Eigen::Vector2d GetImageWarpingField(double u, double v) const;

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    Eigen::VectorXd flow_;
    int anchor_w_;
    int anchor_h_;
    double anchor_step_;
};

}  // namespace color_map
}  // namespace pipelines
}  // namespace cloudViewer

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(
        cloudViewer::pipelines::color_map::ImageWarpingField);
