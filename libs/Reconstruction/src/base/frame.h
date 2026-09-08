// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <iosfwd>
#include <set>
#include <vector>

#include "base/pose.h"
#include "base/rig.h"
#include "geometry/rigid3.h"
#include "util/logging.h"
#include "util/types.h"

namespace colmap {

// One simultaneous capture of a Rig. Frame owns generic sensor data and an
// optional rig-from-world pose. Image membership remains a source-compatible
// camera-data adapter for the legacy SfM pipeline.
class Frame {
public:
    frame_t FrameId() const;
    void SetFrameId(frame_t frame_id);
    bool HasRigId() const { return rig_id_ != kInvalidRigId; }
    rig_t RigId() const;
    void SetRigId(rig_t rig_id);

    void AddDataId(const data_t& data_id);
    bool HasDataId(const data_t& data_id) const;
    const std::set<data_t>& DataIds() const;
    // Upstream-parity accessor (COLMAP 4.x scene/frame.h).
    size_t NumDataIds() const { return data_ids_.size(); }

    void AddImageId(image_t image_id);
    bool HasImageId(image_t image_id) const;
    const std::set<image_t>& ImageIds() const;

    bool HasPose() const;

    // Upstream-parity (COLMAP 4.x scene/frame.h): the owning rig and the
    // composition of sensor_from_rig and rig_from_world for a sensor.
    inline class Rig* RigPtr() const;
    inline void SetRigPtr(class Rig* rig);
    inline void ResetRigPtr();
    inline bool HasRigPtr() const;
    inline Rigid3d SensorFromWorld(sensor_t sensor_id) const;
    // Upstream-parity overload (COLMAP 4.x geometry/rigid3d based API).
    void SetRigFromWorld(const Rigid3d& cam_from_world) {
        const Eigen::Quaterniond& q = cam_from_world.rotation();
        SetRigFromWorld(Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()),
                        cam_from_world.translation());
    }
    Rigid3d RigFromWorld() const {
        const Eigen::Vector4d& q = RigFromWorldQvec();
        return Rigid3d(Eigen::Quaterniond(q(0), q(1), q(2), q(3)),
                       RigFromWorldTvec());
    }

    // Upstream COLMAP dbb41680 scene/frame.h: set the pose of a camera
    // sensor from its cam-from-world pose, inverting the rig extrinsics.
    void SetCamFromWorld(camera_t camera_id, const Rigid3d& cam_from_world) {
        THROW_CHECK_NOTNULL(rig_ptr_);
        const sensor_t sensor_id(SensorType::CAMERA, camera_id);
        if (rig_ptr_->IsRefSensor(sensor_id)) {
            SetRigFromWorld(cam_from_world);
        } else {
            SetRigFromWorld(Inverse(rig_ptr_->SensorFromRig(sensor_id)) *
                            cam_from_world);
        }
    }

    // Upstream COLMAP dbb41680 scene/frame.h parity: nullopt when the frame
    // has no pose yet.
    std::optional<Rigid3d> MaybeRigFromWorld() const {
        if (!HasPose()) {
            return std::nullopt;
        }
        return RigFromWorld();
    }

    void SetRigFromWorld(const Eigen::Vector4d& qvec,
                         const Eigen::Vector3d& tvec);
    void ResetPose();
    const Eigen::Vector4d& RigFromWorldQvec() const;
    const Eigen::Vector3d& RigFromWorldTvec() const;

    // Upstream-parity equality (COLMAP 4.x scene/frame.h): the pose is only
    // compared when both frames have one; this fork tracks it with the
    // has_pose_ flag and qvec/tvec storage instead of an optional Rigid3d.
    bool operator==(const Frame& other) const {
        return frame_id_ == other.frame_id_ && rig_id_ == other.rig_id_ &&
               data_ids_ == other.data_ids_ && has_pose_ == other.has_pose_ &&
               (!has_pose_ ||
                (rig_from_world_qvec_ == other.rig_from_world_qvec_ &&
                 rig_from_world_tvec_ == other.rig_from_world_tvec_));
    }
    bool operator!=(const Frame& other) const { return !(*this == other); }

    bool ReadText(std::istream* stream);
    void WriteText(std::ostream* stream) const;
    bool ReadBinary(std::istream* stream);
    void WriteBinary(std::ostream* stream) const;

private:
    frame_t frame_id_ = kInvalidFrameId;
    rig_t rig_id_ = kInvalidRigId;
    std::set<data_t> data_ids_;
    std::set<image_t> image_ids_;
    class Rig* rig_ptr_ = nullptr;
    bool has_pose_ = false;
    Eigen::Vector4d rig_from_world_qvec_ = ComposeIdentityQuaternion();
    Eigen::Vector3d rig_from_world_tvec_ = Eigen::Vector3d::Zero();
};

inline class Rig* Frame::RigPtr() const { return rig_ptr_; }
inline void Frame::SetRigPtr(class Rig* rig) { rig_ptr_ = rig; }
inline void Frame::ResetRigPtr() { rig_ptr_ = nullptr; }
inline bool Frame::HasRigPtr() const { return rig_ptr_ != nullptr; }
inline Rigid3d Frame::SensorFromWorld(sensor_t sensor_id) const {
    THROW_CHECK_NOTNULL(rig_ptr_);
    if (rig_ptr_->IsRefSensor(sensor_id)) {
        return RigFromWorld();
    }
    return rig_ptr_->SensorFromRig(sensor_id) * RigFromWorld();
}
}  // namespace colmap
