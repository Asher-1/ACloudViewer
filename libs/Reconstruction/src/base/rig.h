// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <iosfwd>
#include <map>
#include <optional>

#include "geometry/rigid3.h"
#include <vector>

#include "base/pose.h"
#include "util/types.h"

namespace colmap {

// Persistent rig calibration. The generic sensor API matches current COLMAP;
// camera-named methods remain source-compatible adapters for the legacy SfM
// pipeline.
class Rig {
public:
    rig_t RigId() const;
    void SetRigId(rig_t rig_id);

    void AddRefSensor(const sensor_t& sensor_id);
    void AddSensor(const sensor_t& sensor_id,
                   const std::optional<Eigen::Vector4d>& sensor_from_rig_qvec,
                   const std::optional<Eigen::Vector3d>& sensor_from_rig_tvec);
    bool HasSensor(const sensor_t& sensor_id) const;
    size_t NumSensors() const;
    bool IsRefSensor(const sensor_t& sensor_id) const;
    const sensor_t& RefSensorId() const;
    // Upstream-parity (COLMAP 4.x sensor/rig.h): non-reference sensors with
    // their (optional) sensor-from-rig transformations.
    std::map<sensor_t, std::optional<Rigid3d>>& NonRefSensors();
    const std::map<sensor_t, std::optional<Rigid3d>>& NonRefSensors() const;
    std::vector<sensor_t> SensorIds() const;
    bool HasSensorFromRig(const sensor_t& sensor_id) const;

    // Upstream-parity rig accessor (COLMAP 4.x): SensorFromRig returns the
    // identity for the reference sensor.
    Rigid3d SensorFromRig(const sensor_t& sensor_id) const {
        if (!HasSensorFromRig(sensor_id)) {
            return Rigid3d();
        }
        return Rigid3d(Eigen::Quaterniond(SensorFromRigQvec(sensor_id)),
                       SensorFromRigTvec(sensor_id));
    }
    const Eigen::Vector4d& SensorFromRigQvec(const sensor_t& sensor_id) const;
    const Eigen::Vector3d& SensorFromRigTvec(const sensor_t& sensor_id) const;

    void AddRefCamera(camera_t camera_id);
    void AddCamera(camera_t camera_id,
                   const Eigen::Vector4d& cam_from_rig_qvec,
                   const Eigen::Vector3d& cam_from_rig_tvec);
    bool HasCamera(camera_t camera_id) const;
    size_t NumCameras() const;
    camera_t RefCameraId() const;
    std::vector<camera_t> CameraIds() const;

    const Eigen::Vector4d& CamFromRigQvec(camera_t camera_id) const;
    Eigen::Vector4d& CamFromRigQvec(camera_t camera_id);
    const Eigen::Vector3d& CamFromRigTvec(camera_t camera_id) const;
    Eigen::Vector3d& CamFromRigTvec(camera_t camera_id);

    bool ReadText(std::istream* stream);
    void WriteText(std::ostream* stream) const;
    bool ReadBinary(std::istream* stream);
    void WriteBinary(std::ostream* stream) const;

private:
    struct SensorPose {
        Eigen::Vector4d qvec = ComposeIdentityQuaternion();
        Eigen::Vector3d tvec = Eigen::Vector3d::Zero();
    };

    rig_t rig_id_ = kInvalidRigId;
    sensor_t ref_sensor_id_;
    std::map<sensor_t, std::optional<SensorPose>> sensors_;
};

}  // namespace colmap
