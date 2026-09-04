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
#include "util/types.h"

namespace colmap {

// One simultaneous capture of a Rig. Frame owns generic sensor data and an
// optional rig-from-world pose. Image membership remains a source-compatible
// camera-data adapter for the legacy SfM pipeline.
class Frame {
public:
    frame_t FrameId() const;
    void SetFrameId(frame_t frame_id);
    rig_t RigId() const;
    void SetRigId(rig_t rig_id);

    void AddDataId(const data_t& data_id);
    bool HasDataId(const data_t& data_id) const;
    const std::set<data_t>& DataIds() const;

    void AddImageId(image_t image_id);
    bool HasImageId(image_t image_id) const;
    const std::set<image_t>& ImageIds() const;

    bool HasPose() const;
    void SetRigFromWorld(const Eigen::Vector4d& qvec,
                         const Eigen::Vector3d& tvec);
    void ResetPose();
    const Eigen::Vector4d& RigFromWorldQvec() const;
    const Eigen::Vector3d& RigFromWorldTvec() const;

    bool ReadText(std::istream* stream);
    void WriteText(std::ostream* stream) const;
    bool ReadBinary(std::istream* stream);
    void WriteBinary(std::ostream* stream) const;

private:
    frame_t frame_id_ = kInvalidFrameId;
    rig_t rig_id_ = kInvalidRigId;
    std::set<data_t> data_ids_;
    std::set<image_t> image_ids_;
    bool has_pose_ = false;
    Eigen::Vector4d rig_from_world_qvec_ = ComposeIdentityQuaternion();
    Eigen::Vector3d rig_from_world_tvec_ = Eigen::Vector3d::Zero();
};

}  // namespace colmap
