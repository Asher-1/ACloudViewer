// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <IJsonConvertible.h>

namespace cloudViewer {
namespace visualization {
class ViewParameters : public cloudViewer::utility::IJsonConvertible {
public:
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    typedef Eigen::Matrix<double, 17, 4, Eigen::RowMajor> Matrix17x4d;
    typedef Eigen::Matrix<double, 17, 1> Vector17d;
    typedef Eigen::aligned_allocator<Matrix17x4d> Matrix17x4d_allocator;

public:
    ViewParameters()
        : field_of_view_(0),
          zoom_(0),
          lookat_(0, 0, 0),
          up_(0, 0, 0),
          front_(0, 0, 0),
          boundingbox_min_(0, 0, 0),
          boundingbox_max_(0, 0, 0) {}
    ~ViewParameters() override {}

public:
    Vector17d ConvertToVector17d();
    void ConvertFromVector17d(const Vector17d &v);
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    double field_of_view_;
    double zoom_;
    Eigen::Vector3d lookat_;
    Eigen::Vector3d up_;
    Eigen::Vector3d front_;
    Eigen::Vector3d boundingbox_min_;
    Eigen::Vector3d boundingbox_max_;
};

}  // namespace visualization
}  // namespace cloudViewer
