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

#include "camera/PinholeCameraParameters.h"

namespace cloudViewer {
namespace camera {

/// \class PinholeCameraTrajectory
///
/// Contains a list of PinholeCameraParameters, useful to storing trajectories.
class ECV_DB_LIB_API PinholeCameraTrajectory :
	public cloudViewer::utility::IJsonConvertible {
public:
    /// \brief Default Constructor.
    PinholeCameraTrajectory();
    virtual ~PinholeCameraTrajectory() override;

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    /// List of PinholeCameraParameters objects.
    std::vector<PinholeCameraParameters> parameters_;
};

}  // namespace camera
}  // namespace cloudViewer
