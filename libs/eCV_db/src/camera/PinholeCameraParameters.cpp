// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "camera/PinholeCameraParameters.h"

#include <Logging.h>
#include <json/json.h>

namespace cloudViewer {
namespace camera {
using namespace cloudViewer;

PinholeCameraParameters::PinholeCameraParameters() {}
PinholeCameraParameters::~PinholeCameraParameters() {}

bool PinholeCameraParameters::ConvertToJsonValue(Json::Value &value) const {
    Json::Value trajectory_array;
    value["class_name"] = "PinholeCameraParameters";
    value["version_major"] = 1;
    value["version_minor"] = 0;
    if (EigenMatrix4dToJsonArray(extrinsic_, value["extrinsic"]) == false) {
        return false;
    }
    if (intrinsic_.ConvertToJsonValue(value["intrinsic"]) == false) {
        return false;
    }
    return true;
}

bool PinholeCameraParameters::ConvertFromJsonValue(const Json::Value &value) {
    if (value.isObject() == false) {
        utility::LogWarning(
                "PinholeCameraParameters read JSON failed: unsupported json "
                "format.");
        return false;
    }
    if (value.get("class_name", "").asString() != "PinholeCameraParameters" ||
        value.get("version_major", 1).asInt() != 1 ||
        value.get("version_minor", 0).asInt() != 0) {
        utility::LogWarning(
                "PinholeCameraParameters read JSON failed: unsupported json "
                "format.");
        return false;
    }
    if (intrinsic_.ConvertFromJsonValue(value["intrinsic"]) == false) {
        return false;
    }
    if (EigenMatrix4dFromJsonArray(extrinsic_, value["extrinsic"]) == false) {
        return false;
    }
    return true;
}
}  // namespace camera
}  // namespace cloudViewer
