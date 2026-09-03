// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "base/camera_rig.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

#include "util/endian.h"
#include "util/misc.h"

namespace colmap {

namespace {

constexpr uint32_t kCameraRigBinaryMagic = 0x43525631;  // CRV1
constexpr uint32_t kCameraRigFormatVersion = 1;
constexpr uint64_t kMaxSerializedRigElements = 1000000;

bool IsValidRelativePose(const Eigen::Vector4d& qvec,
                         const Eigen::Vector3d& tvec) {
  return qvec.allFinite() && tvec.allFinite() &&
         qvec.squaredNorm() > std::numeric_limits<double>::epsilon();
}

bool IsValidSnapshots(const std::vector<std::vector<image_t>>& snapshots,
                      const size_t num_cameras) {
  std::unordered_set<image_t> image_ids;
  for (const auto& snapshot : snapshots) {
    if (snapshot.empty() || snapshot.size() > num_cameras) {
      return false;
    }
    for (const image_t image_id : snapshot) {
      if (!image_ids.insert(image_id).second) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

CameraRig::CameraRig() {}

size_t CameraRig::NumCameras() const { return rig_cameras_.size(); }

size_t CameraRig::NumSnapshots() const { return snapshots_.size(); }

bool CameraRig::HasCamera(const camera_t camera_id) const {
  return rig_cameras_.count(camera_id);
}

camera_t CameraRig::RefCameraId() const { return ref_camera_id_; }

void CameraRig::SetRefCameraId(const camera_t camera_id) {
  CHECK(HasCamera(camera_id));
  ref_camera_id_ = camera_id;
}

std::vector<camera_t> CameraRig::GetCameraIds() const {
  std::vector<camera_t> rig_camera_ids;
  rig_camera_ids.reserve(NumCameras());
  for (const auto& rig_camera : rig_cameras_) {
    rig_camera_ids.push_back(rig_camera.first);
  }
  return rig_camera_ids;
}

const std::vector<std::vector<image_t>>& CameraRig::Snapshots() const {
  return snapshots_;
}

void CameraRig::WriteText(std::ostream* stream) const {
  CHECK_NOTNULL(stream);
  CHECK(stream->good());
  stream->precision(17);
  *stream << "# ACloudViewer CameraRig v1\n";
  *stream << "RIG " << kCameraRigFormatVersion << " " << ref_camera_id_
          << " " << NumCameras() << " " << NumSnapshots() << "\n";

  std::vector<camera_t> camera_ids = GetCameraIds();
  std::sort(camera_ids.begin(), camera_ids.end());
  for (const camera_t camera_id : camera_ids) {
    const Eigen::Vector4d& qvec = RelativeQvec(camera_id);
    const Eigen::Vector3d& tvec = RelativeTvec(camera_id);
    *stream << "CAMERA " << camera_id << " " << qvec(0) << " " << qvec(1)
            << " " << qvec(2) << " " << qvec(3) << " " << tvec(0) << " "
            << tvec(1) << " " << tvec(2) << "\n";
  }
  for (const auto& snapshot : snapshots_) {
    *stream << "SNAPSHOT " << snapshot.size();
    for (const image_t image_id : snapshot) {
      *stream << " " << image_id;
    }
    *stream << "\n";
  }
}

bool CameraRig::ReadText(std::istream* stream) {
  if (stream == nullptr || !stream->good()) {
    return false;
  }

  std::string tag;
  while (*stream >> tag && tag[0] == '#') {
    stream->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  if (!stream->good() || tag != "RIG") {
    return false;
  }

  uint32_t version = 0;
  camera_t ref_camera_id = kInvalidCameraId;
  uint64_t num_cameras = 0;
  uint64_t num_snapshots = 0;
  if (!(*stream >> version >> ref_camera_id >> num_cameras >> num_snapshots) ||
      version != kCameraRigFormatVersion || num_cameras == 0 ||
      num_cameras > kMaxSerializedRigElements ||
      num_snapshots > kMaxSerializedRigElements) {
    return false;
  }

  CameraRig parsed;
  for (uint64_t i = 0; i < num_cameras; ++i) {
    Eigen::Vector4d qvec;
    Eigen::Vector3d tvec;
    camera_t camera_id = kInvalidCameraId;
    if (!(*stream >> tag >> camera_id >> qvec(0) >> qvec(1) >> qvec(2) >>
          qvec(3) >> tvec(0) >> tvec(1) >> tvec(2)) ||
        tag != "CAMERA" || parsed.HasCamera(camera_id) ||
        !IsValidRelativePose(qvec, tvec)) {
      return false;
    }
    parsed.rig_cameras_.emplace(camera_id, RigCamera{qvec, tvec});
  }
  if (!parsed.HasCamera(ref_camera_id)) {
    return false;
  }
  parsed.ref_camera_id_ = ref_camera_id;

  for (uint64_t i = 0; i < num_snapshots; ++i) {
    uint64_t num_images = 0;
    if (!(*stream >> tag >> num_images) || tag != "SNAPSHOT" ||
        num_images == 0 || num_images > num_cameras) {
      return false;
    }
    std::vector<image_t> snapshot(num_images);
    for (image_t& image_id : snapshot) {
      if (!(*stream >> image_id)) {
        return false;
      }
    }
    parsed.snapshots_.push_back(std::move(snapshot));
  }
  if (!IsValidSnapshots(parsed.snapshots_, parsed.NumCameras())) {
    return false;
  }
  *this = std::move(parsed);
  return true;
}

void CameraRig::WriteBinary(std::ostream* stream) const {
  CHECK_NOTNULL(stream);
  CHECK(stream->good());
  WriteBinaryLittleEndian<uint32_t>(stream, kCameraRigBinaryMagic);
  WriteBinaryLittleEndian<uint32_t>(stream, kCameraRigFormatVersion);
  WriteBinaryLittleEndian<camera_t>(stream, ref_camera_id_);
  WriteBinaryLittleEndian<uint64_t>(stream, NumCameras());
  WriteBinaryLittleEndian<uint64_t>(stream, NumSnapshots());

  std::vector<camera_t> camera_ids = GetCameraIds();
  std::sort(camera_ids.begin(), camera_ids.end());
  for (const camera_t camera_id : camera_ids) {
    WriteBinaryLittleEndian<camera_t>(stream, camera_id);
    const Eigen::Vector4d& qvec = RelativeQvec(camera_id);
    const Eigen::Vector3d& tvec = RelativeTvec(camera_id);
    for (int i = 0; i < qvec.size(); ++i) {
      WriteBinaryLittleEndian<double>(stream, qvec(i));
    }
    for (int i = 0; i < tvec.size(); ++i) {
      WriteBinaryLittleEndian<double>(stream, tvec(i));
    }
  }
  for (const auto& snapshot : snapshots_) {
    WriteBinaryLittleEndian<uint64_t>(stream, snapshot.size());
    for (const image_t image_id : snapshot) {
      WriteBinaryLittleEndian<image_t>(stream, image_id);
    }
  }
}

bool CameraRig::ReadBinary(std::istream* stream) {
  if (stream == nullptr || !stream->good()) {
    return false;
  }
  const uint32_t magic = ReadBinaryLittleEndian<uint32_t>(stream);
  const uint32_t version = ReadBinaryLittleEndian<uint32_t>(stream);
  const camera_t ref_camera_id = ReadBinaryLittleEndian<camera_t>(stream);
  const uint64_t num_cameras = ReadBinaryLittleEndian<uint64_t>(stream);
  const uint64_t num_snapshots = ReadBinaryLittleEndian<uint64_t>(stream);
  if (!stream->good() || magic != kCameraRigBinaryMagic ||
      version != kCameraRigFormatVersion || num_cameras == 0 ||
      num_cameras > kMaxSerializedRigElements ||
      num_snapshots > kMaxSerializedRigElements) {
    return false;
  }

  CameraRig parsed;
  for (uint64_t i = 0; i < num_cameras; ++i) {
    const camera_t camera_id = ReadBinaryLittleEndian<camera_t>(stream);
    Eigen::Vector4d qvec;
    Eigen::Vector3d tvec;
    for (int j = 0; j < qvec.size(); ++j) {
      qvec(j) = ReadBinaryLittleEndian<double>(stream);
    }
    for (int j = 0; j < tvec.size(); ++j) {
      tvec(j) = ReadBinaryLittleEndian<double>(stream);
    }
    if (!stream->good() || parsed.HasCamera(camera_id) ||
        !IsValidRelativePose(qvec, tvec)) {
      return false;
    }
    parsed.rig_cameras_.emplace(camera_id, RigCamera{qvec, tvec});
  }
  if (!parsed.HasCamera(ref_camera_id)) {
    return false;
  }
  parsed.ref_camera_id_ = ref_camera_id;

  for (uint64_t i = 0; i < num_snapshots; ++i) {
    const uint64_t num_images = ReadBinaryLittleEndian<uint64_t>(stream);
    if (!stream->good() || num_images == 0 || num_images > num_cameras) {
      return false;
    }
    std::vector<image_t> snapshot(num_images);
    for (image_t& image_id : snapshot) {
      image_id = ReadBinaryLittleEndian<image_t>(stream);
    }
    if (!stream->good()) {
      return false;
    }
    parsed.snapshots_.push_back(std::move(snapshot));
  }
  if (!IsValidSnapshots(parsed.snapshots_, parsed.NumCameras())) {
    return false;
  }
  *this = std::move(parsed);
  return true;
}

void CameraRig::AddCamera(const camera_t camera_id,
                          const Eigen::Vector4d& rel_qvec,
                          const Eigen::Vector3d& rel_tvec) {
  CHECK(!HasCamera(camera_id));
  CHECK_EQ(NumSnapshots(), 0);
  RigCamera rig_camera;
  rig_camera.rel_qvec = rel_qvec;
  rig_camera.rel_tvec = rel_tvec;
  rig_cameras_.emplace(camera_id, rig_camera);
}

void CameraRig::AddSnapshot(const std::vector<image_t>& image_ids) {
  CHECK(!image_ids.empty());
  CHECK_LE(image_ids.size(), NumCameras());
  CHECK(!VectorContainsDuplicateValues(image_ids));
  snapshots_.push_back(image_ids);
}

void CameraRig::Check(const Reconstruction& reconstruction) const {
  CHECK(HasCamera(ref_camera_id_));

  for (const auto& rig_camera : rig_cameras_) {
    CHECK(reconstruction.ExistsCamera(rig_camera.first));
  }

  std::unordered_set<image_t> all_image_ids;
  for (const auto& snapshot : snapshots_) {
    CHECK(!snapshot.empty());
    CHECK_LE(snapshot.size(), NumCameras());
    bool has_ref_camera = false;
    for (const auto image_id : snapshot) {
      CHECK(reconstruction.ExistsImage(image_id));
      CHECK_EQ(all_image_ids.count(image_id), 0);
      all_image_ids.insert(image_id);
      const auto& image = reconstruction.Image(image_id);
      CHECK(HasCamera(image.CameraId()));
      if (image.CameraId() == ref_camera_id_) {
        has_ref_camera = true;
      }
    }
    CHECK(has_ref_camera);
  }
}

Eigen::Vector4d& CameraRig::RelativeQvec(const camera_t camera_id) {
  return rig_cameras_.at(camera_id).rel_qvec;
}

const Eigen::Vector4d& CameraRig::RelativeQvec(const camera_t camera_id) const {
  return rig_cameras_.at(camera_id).rel_qvec;
}

Eigen::Vector3d& CameraRig::RelativeTvec(const camera_t camera_id) {
  return rig_cameras_.at(camera_id).rel_tvec;
}

const Eigen::Vector3d& CameraRig::RelativeTvec(const camera_t camera_id) const {
  return rig_cameras_.at(camera_id).rel_tvec;
}

double CameraRig::ComputeScale(const Reconstruction& reconstruction) const {
  CHECK_GT(NumSnapshots(), 0);
  CHECK_GT(NumCameras(), 0);
  double scaling_factor = 0;
  size_t num_dists = 0;
  std::vector<Eigen::Vector3d> rel_proj_centers(NumCameras());
  std::vector<Eigen::Vector3d> abs_proj_centers(NumCameras());
  for (const auto& snapshot : snapshots_) {
    // Compute the projection relative and absolute projection centers.
    for (size_t i = 0; i < NumCameras(); ++i) {
      const auto& image = reconstruction.Image(snapshot[i]);
      rel_proj_centers[i] = ProjectionCenterFromPose(
          RelativeQvec(image.CameraId()), RelativeTvec(image.CameraId()));
      abs_proj_centers[i] = image.ProjectionCenter();
    }

    // Accumulate the scaling factor for all pairs of camera distances.
    for (size_t i = 0; i < NumCameras(); ++i) {
      for (size_t j = 0; j < i; ++j) {
        const double rel_dist =
            (rel_proj_centers[i] - rel_proj_centers[j]).norm();
        const double abs_dist =
            (abs_proj_centers[i] - abs_proj_centers[j]).norm();
        const double kMinDist = 1e-6;
        if (rel_dist > kMinDist && abs_dist > kMinDist) {
          scaling_factor += rel_dist / abs_dist;
          num_dists += 1;
        }
      }
    }
  }

  if (num_dists == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  return scaling_factor / num_dists;
}

bool CameraRig::ComputeRelativePoses(const Reconstruction& reconstruction) {
  CHECK_GT(NumSnapshots(), 0);
  CHECK_NE(ref_camera_id_, kInvalidCameraId);

  for (auto& rig_camera : rig_cameras_) {
    rig_camera.second.rel_tvec = Eigen::Vector3d::Zero();
  }

  std::unordered_map<camera_t, std::vector<Eigen::Vector4d>> rel_qvecs;

  for (const auto& snapshot : snapshots_) {
    // Find the image of the reference camera in the current snapshot.
    const Image* ref_image = nullptr;
    for (const auto image_id : snapshot) {
      const auto& image = reconstruction.Image(image_id);
      if (image.CameraId() == ref_camera_id_) {
        ref_image = &image;
      }
    }

    CHECK_NOTNULL(ref_image);

    // Compute the relative poses from all cameras in the current snapshot to
    // the reference camera.
    for (const auto image_id : snapshot) {
      const auto& image = reconstruction.Image(image_id);
      if (image.CameraId() != ref_camera_id_) {
        Eigen::Vector4d rel_qvec;
        Eigen::Vector3d rel_tvec;
        ComputeRelativePose(ref_image->Qvec(), ref_image->Tvec(), image.Qvec(),
                            image.Tvec(), &rel_qvec, &rel_tvec);

        rel_qvecs[image.CameraId()].push_back(rel_qvec);
        RelativeTvec(image.CameraId()) += rel_tvec;
      }
    }
  }

  RelativeQvec(ref_camera_id_) = ComposeIdentityQuaternion();
  RelativeTvec(ref_camera_id_) = Eigen::Vector3d(0, 0, 0);

  // Compute the average relative poses.
  for (auto& rig_camera : rig_cameras_) {
    if (rig_camera.first != ref_camera_id_) {
      if (rel_qvecs.count(rig_camera.first) == 0) {
        std::cout << "Need at least one snapshot with an image of camera "
                  << rig_camera.first << " and the reference camera "
                  << ref_camera_id_
                  << " to compute its relative pose in the camera rig"
                  << std::endl;
        return false;
      }
      const std::vector<Eigen::Vector4d>& camera_rel_qvecs =
          rel_qvecs.at(rig_camera.first);
      const std::vector<double> rel_qvec_weights(camera_rel_qvecs.size(), 1.0);
      rig_camera.second.rel_qvec =
          AverageQuaternions(camera_rel_qvecs, rel_qvec_weights);
      rig_camera.second.rel_tvec /= camera_rel_qvecs.size();
    }
  }
  return true;
}

void CameraRig::ComputeAbsolutePose(const size_t snapshot_idx,
                                    const Reconstruction& reconstruction,
                                    Eigen::Vector4d* abs_qvec,
                                    Eigen::Vector3d* abs_tvec) const {
  const auto& snapshot = snapshots_.at(snapshot_idx);

  std::vector<Eigen::Vector4d> abs_qvecs;
  *abs_tvec = Eigen::Vector3d::Zero();

  for (const auto image_id : snapshot) {
    const auto& image = reconstruction.Image(image_id);
    Eigen::Vector4d inv_rel_qvec;
    Eigen::Vector3d inv_rel_tvec;
    InvertPose(RelativeQvec(image.CameraId()), RelativeTvec(image.CameraId()),
               &inv_rel_qvec, &inv_rel_tvec);

    const Eigen::Vector4d qvec =
        ConcatenateQuaternions(image.Qvec(), inv_rel_qvec);
    const Eigen::Vector3d tvec = QuaternionRotatePoint(
        inv_rel_qvec, image.Tvec() - RelativeTvec(image.CameraId()));

    abs_qvecs.push_back(qvec);
    *abs_tvec += tvec;
  }

  const std::vector<double> abs_qvec_weights(snapshot.size(), 1);
  *abs_qvec = AverageQuaternions(abs_qvecs, abs_qvec_weights);
  *abs_tvec /= snapshot.size();
}

}  // namespace colmap
