// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "eCV_db.h"

#include <memory>
#include <vector>

#include "camera/PinholeCameraIntrinsic.h"

namespace cloudViewer {
namespace camera {

/// \class PinholeCameraParameters
///
/// \brief Contains both intrinsic and extrinsic pinhole camera parameters.
class ECV_DB_LIB_API PinholeCameraParameters :
	public cloudViewer::utility::IJsonConvertible {
public:
    // Must comment it due to unreferenced symbols when linked
    // CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    /// \brief Default Constructor.
    PinholeCameraParameters();
    virtual ~PinholeCameraParameters() override;

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    /// PinholeCameraIntrinsic object.
    PinholeCameraIntrinsic intrinsic_;
    /// Camera extrinsic parameters.
    Eigen::Matrix4d_u extrinsic_;

    std::string texture_file_;
};
}  // namespace camera
}  // namespace cloudViewer
