// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <IJsonConvertible.h>

#include <string>
#include <unordered_map>

#include "io/sensor/RGBDSensorConfig.h"

struct _k4a_device_configuration_t;  // Alias of k4a_device_configuration_t

namespace cloudViewer {
namespace io {

// Alternative implementation of _k4a_device_configuration_t with string values

/// \class AzureKinectSensorConfig
///
/// AzureKinect sensor configuration.
class AzureKinectSensorConfig : public RGBDSensorConfig {
public:
    /// Default constructor, default configs will be used
    AzureKinectSensorConfig();
    /// Initialize config with a map
    AzureKinectSensorConfig(
            const std::unordered_map<std::string, std::string> &config);
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    void ConvertFromNativeConfig(const _k4a_device_configuration_t &k4a_config);
    _k4a_device_configuration_t ConvertToNativeConfig() const;

public:
    // To avoid including k4a or json header, configs is stored in a map
    std::unordered_map<std::string, std::string> config_;

protected:
    static bool IsValidConfig(
            const std::unordered_map<std::string, std::string> &config,
            bool verbose = true);
};

}  // namespace io
}  // namespace cloudViewer
