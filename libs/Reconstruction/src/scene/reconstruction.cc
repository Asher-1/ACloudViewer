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
// Author: Johannes L. schoenberg (jsch-at-demuc-dot-de)

#include "scene/reconstruction.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>

#include "scene/database_cache.h"
#include "scene/reconstruction_io.h"
#include "geometry/gps.h"
#include "geometry/normalization.h"
#include "geometry/pose.h"
#include "scene/projection.h"
#include "geometry/triangulation.h"
#include "sensor/bitmap.h"
#include "util/misc.h"
#include "util/ply.h"

namespace colmap {

Reconstruction::Reconstruction()
    : correspondence_graph_(nullptr), num_added_points3D_(0) {}

Reconstruction::Reconstruction(const Reconstruction& other)
    : correspondence_graph_(other.GetCorrespondenceGraph()),
      cameras_(other.Cameras()),
      images_(other.Images()),
      rigs_(other.Rigs()),
      frames_(other.Frames()),
      points3D_(other.Points3D()),
      image_pair_stats_(other.ImagePairs()),
      reg_image_ids_(other.RegImageIds()),
      reg_frame_ids_(other.reg_frame_ids_),
      num_added_points3D_(other.NumAddedPoints3D()) {
    RewireObjectPointers();
}

Reconstruction& Reconstruction::operator=(const Reconstruction& other) {
    if (this != &other) {
        correspondence_graph_ = other.GetCorrespondenceGraph();
        cameras_ = other.Cameras();
        images_ = other.Images();
        rigs_ = other.Rigs();
        frames_ = other.Frames();
        points3D_ = other.Points3D();
        image_pair_stats_ = other.ImagePairs();
        reg_image_ids_ = other.RegImageIds();
        reg_frame_ids_ = other.reg_frame_ids_;
        num_added_points3D_ = other.NumAddedPoints3D();
        RewireObjectPointers();
    }
    return *this;
}

void Reconstruction::RewireObjectPointers() {
    // Upstream COLMAP dbb41680 parity (scene/reconstruction.cc copy ctor and
    // assignment): the copied frames/images must point into this object's
    // rigs/cameras/frames, not into the source object's. Without this the
    // pose stored in this->rigs_ is invisible through image.CamFromWorld(),
    // which silently reads the source object's (stale) rig.
    for (auto& [_, frame] : frames_) {
        frame.ResetRigPtr();
        frame.SetRigPtr(&Rig(frame.RigId()));
    }
    for (auto& [_, image] : images_) {
        image.ResetCameraPtr();
        image.SetCameraPtr(&Camera(image.CameraId()));
        image.ResetFramePtr();
        image.SetFramePtr(&Frame(image.FrameId()));
    }
}

std::unordered_set<point3D_t> Reconstruction::Point3DIds() const {
    std::unordered_set<point3D_t> point3D_ids;
    point3D_ids.reserve(points3D_.size());

    for (const auto& point3D : points3D_) {
        point3D_ids.insert(point3D.first);
    }

    return point3D_ids;
}

void Reconstruction::Load(const DatabaseCache& database_cache) {
    correspondence_graph_ = nullptr;

    // Add cameras.
    cameras_.reserve(database_cache.NumCameras());
    for (const auto& [camera_id, camera] : database_cache.Cameras()) {
        if (ExistsCamera(camera_id)) {
            class Camera& existing_camera = Camera(camera_id);
            THROW_CHECK_EQ(existing_camera.ModelId(), camera.ModelId());
            THROW_CHECK_EQ(existing_camera.Width(), camera.Width());
            THROW_CHECK_EQ(existing_camera.Height(), camera.Height());
        } else {
            AddCamera(camera);
        }
    }

    // Add rigs.
    rigs_.reserve(database_cache.NumRigs());
    for (const auto& [rig_id, rig] : database_cache.Rigs()) {
        if (ExistsRig(rig_id)) {
            class Rig& existing_rig = Rig(rig_id);
            THROW_CHECK(existing_rig.RefSensorId() == rig.RefSensorId());
            THROW_CHECK(existing_rig.SensorIds() == rig.SensorIds());
        } else {
            AddRig(rig);
        }
    }

    // Add frames.
    frames_.reserve(database_cache.NumFrames());
    for (const auto& [frame_id, frame] : database_cache.Frames()) {
        if (ExistsFrame(frame_id)) {
            class Frame& existing_frame = Frame(frame_id);
            THROW_CHECK(existing_frame.RigId() == frame.RigId());
            THROW_CHECK(existing_frame.DataIds() == frame.DataIds());
        } else {
            AddFrame(frame);
        }
    }

    // Add images.
    images_.reserve(database_cache.NumImages());

    for (const auto& [image_id, image] : database_cache.Images()) {
        if (ExistsImage(image_id)) {
            class Image& existing_image = Image(image_id);
            THROW_CHECK_EQ(existing_image.Name(), image.Name());
            if (existing_image.NumPoints2D() == 0) {
                existing_image.SetPoints2D(image.Points2D());
            } else {
                THROW_CHECK_EQ(image.NumPoints2D(),
                               existing_image.NumPoints2D());
            }
            // Fork bridge: keep the legacy per-image counters in sync from
            // the cache (upstream derives them from the correspondence
            // graph alone).
            existing_image.SetNumObservations(image.NumObservations());
            existing_image.SetNumCorrespondences(
                    image.NumCorrespondences());
        } else {
            AddImage(image);
        }
    }

    // Add image pairs (fork-specific statistics kept for the legacy
    // observation bookkeeping).
    for (const auto& image_pair :
         database_cache.CorrespondenceGraph()->NumMatchesBetweenAllImages()) {
        ImagePairStat image_pair_stat;
        image_pair_stat.num_total_corrs = image_pair.second;
        image_pair_stats_.emplace(image_pair.first, image_pair_stat);
    }
}

void Reconstruction::SetUp(const CorrespondenceGraph* correspondence_graph) {
    CHECK_NOTNULL(correspondence_graph);
    for (auto& image : images_) {
        image.second.SetUp(Camera(image.second.CameraId()));
    }
    correspondence_graph_ = correspondence_graph;

    // If an existing model was loaded from disk and there were already images
    // registered previously, we need to set observations as triangulated.
    for (const auto image_id : reg_image_ids_) {
        const class Image& image = Image(image_id);
        for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
             ++point2D_idx) {
            if (image.Point2D(point2D_idx).HasPoint3D()) {
                const bool kIsContinuedPoint3D = false;
                SetObservationAsTriangulated(image_id, point2D_idx,
                                             kIsContinuedPoint3D);
            }
        }
    }
}

void Reconstruction::TearDown() {
    // Upstream parity (dbb41680 scene/reconstruction.cc): frame-level
    // teardown. Frames without a pose are removed together with their
    // images; rigs without kept frames are removed together with their
    // sensors' cameras.
    (void)correspondence_graph_;
    image_pair_stats_.clear();

    // Remove all non-registered frames/images.
    FlatHashSet<rig_t> keep_rig_ids;
    for (auto frame_it = frames_.begin(); frame_it != frames_.end();) {
        for (const image_t image_id : frame_it->second.ImageIds()) {
            auto image_it = images_.find(image_id);
            if (!frame_it->second.HasPose() && image_it != images_.end()) {
                images_.erase(image_it);
            }
        }
        if (frame_it->second.HasPose()) {
            keep_rig_ids.insert(frame_it->second.RigId());
            ++frame_it;
        } else {
            // erase(it++) rather than it = erase(it): portable across hash map
            // backends; frames_ is node-based.
            frames_.erase(frame_it++);
        }
    }

    // Remove all unused rigs and corresponding sensors.
    for (auto it = rigs_.begin(); it != rigs_.end();) {
        if (keep_rig_ids.count(it->first) == 0) {
            for (const sensor_t& sensor_id : it->second.SensorIds()) {
                switch (sensor_id.type) {
                    case SensorType::CAMERA:
                        cameras_.erase(sensor_id.id);
                        break;
                    case SensorType::IMU:
                    case SensorType::INVALID:
                        break;
                }
            }
            rigs_.erase(it++);
        } else {
            ++it;
        }
    }

    // Compress tracks.
    for (auto& point3D : points3D_) {
        point3D.second.Track().Compress();
    }
}

void Reconstruction::AddCamera(const class Camera& camera) {
    CHECK(!ExistsCamera(camera.CameraId()));
    CHECK(camera.VerifyParams());
    cameras_.emplace(camera.CameraId(), camera);
}

void Reconstruction::AddImage(class Image image) {
  THROW_CHECK(image.HasCameraId());
  auto& camera = Camera(image.CameraId());
  if (image.HasCameraPtr()) {
    THROW_CHECK_EQ(image.CameraPtr(), &camera);
  } else {
    image.SetCameraPtr(&camera);
  }
  THROW_CHECK(image.HasFrameId());
  auto& frame = Frame(image.FrameId());
  THROW_CHECK(frame.HasDataId(image.DataId()));
  if (image.HasFramePtr()) {
    THROW_CHECK_EQ(image.FramePtr(), &frame);
  } else {
    image.SetFramePtr(&frame);
  }
  const image_t image_id = image.ImageId();
  THROW_CHECK(images_.emplace(image_id, std::move(image)).second);
}

void Reconstruction::AddRig(class Rig rig) {
  auto check_exists_sensor = [&](const auto& sensor_id) {
    switch (sensor_id.type) {
      case SensorType::CAMERA:
        THROW_CHECK(ExistsCamera(sensor_id.id))
            << "Camera " << sensor_id.id << " from rig " << rig.RigId()
            << " not found in the reconstruction. Note that AddCamera "
               "should be called before AddRig.";
        break;
      case SensorType::IMU:
      case SensorType::INVALID:
        break;
    }
  };

  check_exists_sensor(rig.RefSensorId());
  for (const auto& [sensor_id, _] : rig.NonRefSensors()) {
    check_exists_sensor(sensor_id);
  }

  const rig_t rig_id = rig.RigId();
  THROW_CHECK(rigs_.emplace(rig_id, std::move(rig)).second);
}

void Reconstruction::AddFrame(class Frame frame) {
  THROW_CHECK(frame.HasRigId());
  auto& rig = Rig(frame.RigId());
  for (const auto& data_id : frame.DataIds()) {
    switch (data_id.sensor_id.type) {
      case SensorType::CAMERA:
        THROW_CHECK(rig.HasSensor(data_id.sensor_id));
        break;
      case SensorType::IMU:
        // Note that we do not (yet) support IMU measurement data.
        break;
      case SensorType::INVALID:
        LOG(FATAL) << "Invalid sensor type: "
                      << static_cast<int>(data_id.sensor_id.type);
        break;
    }
  }
  if (frame.HasRigPtr()) {
    THROW_CHECK_EQ(frame.RigPtr(), &rig);
  } else {
    frame.SetRigPtr(&rig);
  }
  const bool is_registered = frame.HasPose();
  const frame_t frame_id = frame.FrameId();
  auto [it, inserted] = frames_.emplace(frame_id, std::move(frame));
  THROW_CHECK(inserted);
  // Upstream parity (d3ccaf35 scene/reconstruction.cc L436): a frame that
  // carries a pose is registered on insertion. The fork's disk-read path
  // calls ResetRegistrationState() afterwards to preserve the
  // fresh-load-starts-unregistered semantics of the fork's database pose
  // columns.
  if (is_registered) {
    THROW_CHECK_NE(frame_id, kInvalidFrameId);
    RegisterFrame(frame_id);
  }
}

point3D_t Reconstruction::AddPoint3D(const Eigen::Vector3d& xyz,
                                     const Track& track,
                                     const Eigen::Vector3ub& color) {
    const point3D_t point3D_id = ++num_added_points3D_;
    CHECK(!ExistsPoint3D(point3D_id));

    class Point3D& point3D = points3D_[point3D_id];

    point3D.SetXYZ(xyz);
    point3D.SetTrack(track);
    point3D.SetColor(color);

    for (const auto& track_el : track.Elements()) {
        class Image& image = Image(track_el.image_id);
        CHECK(!image.Point2D(track_el.point2D_idx).HasPoint3D());
        image.SetPoint3DForPoint2D(track_el.point2D_idx, point3D_id);
        CHECK_LE(image.NumPoints3D(), image.NumPoints2D());
    }

    const bool kIsContinuedPoint3D = false;

    for (const auto& track_el : track.Elements()) {
        SetObservationAsTriangulated(track_el.image_id, track_el.point2D_idx,
                                     kIsContinuedPoint3D);
    }

    return point3D_id;
}

// Add new 3D point with known ID (upstream parity, dbb41680 scene/reconstruction.cc).
void Reconstruction::AddPoint3D(const point3D_t point3D_id,
                                struct Point3D point3D) {
    // Fork parity: the fork allocates ids from num_added_points3D_; bump it
    // so subsequent allocating AddPoint3D calls never collide with loaded ids.
    num_added_points3D_ = std::max(num_added_points3D_, point3D_id);

    for (const auto& track_el : point3D.Track().Elements()) {
        class Image& image = Image(track_el.image_id);
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);
        if (point2D.HasPoint3D()) {
            THROW_CHECK_EQ(point2D.Point3DId(), point3D_id);
        } else {
            image.SetPoint3DForPoint2D(track_el.point2D_idx, point3D_id);
        }
        THROW_CHECK_LE(image.NumPoints3D(), image.NumPoints2D());
    }
    THROW_CHECK(points3D_.emplace(point3D_id, std::move(point3D)).second);
}

void Reconstruction::AddObservation(const point3D_t point3D_id,
                                    const TrackElement& track_el) {
    class Image& image = Image(track_el.image_id);
    CHECK(!image.Point2D(track_el.point2D_idx).HasPoint3D());

    image.SetPoint3DForPoint2D(track_el.point2D_idx, point3D_id);
    CHECK_LE(image.NumPoints3D(), image.NumPoints2D());

    class Point3D& point3D = Point3D(point3D_id);
    point3D.Track().AddElement(track_el);

    const bool kIsContinuedPoint3D = true;
    SetObservationAsTriangulated(track_el.image_id, track_el.point2D_idx,
                                 kIsContinuedPoint3D);
}

point3D_t Reconstruction::MergePoints3D(const point3D_t point3D_id1,
                                        const point3D_t point3D_id2) {
    const class Point3D& point3D1 = Point3D(point3D_id1);
    const class Point3D& point3D2 = Point3D(point3D_id2);

    const Eigen::Vector3d merged_xyz =
            (point3D1.Track().Length() * point3D1.XYZ() +
             point3D2.Track().Length() * point3D2.XYZ()) /
            (point3D1.Track().Length() + point3D2.Track().Length());
    const Eigen::Vector3d merged_rgb =
            (point3D1.Track().Length() * point3D1.Color().cast<double>() +
             point3D2.Track().Length() * point3D2.Color().cast<double>()) /
            (point3D1.Track().Length() + point3D2.Track().Length());

    Track merged_track;
    merged_track.Reserve(point3D1.Track().Length() + point3D2.Track().Length());
    merged_track.AddElements(point3D1.Track().Elements());
    merged_track.AddElements(point3D2.Track().Elements());

    DeletePoint3D(point3D_id1);
    DeletePoint3D(point3D_id2);

    const point3D_t merged_point3D_id =
            AddPoint3D(merged_xyz, merged_track, merged_rgb.cast<uint8_t>());

    return merged_point3D_id;
}

void Reconstruction::DeletePoint3D(const point3D_t point3D_id) {
    // Note: Do not change order of these instructions, especially with respect
    // to `Reconstruction::ResetTriObservations`

    const class Track& track = Point3D(point3D_id).Track();

    const bool kIsDeletedPoint3D = true;

    for (const auto& track_el : track.Elements()) {
        ResetTriObservations(track_el.image_id, track_el.point2D_idx,
                             kIsDeletedPoint3D);
    }

    for (const auto& track_el : track.Elements()) {
        class Image& image = Image(track_el.image_id);
        image.ResetPoint3DForPoint2D(track_el.point2D_idx);
    }

    points3D_.erase(point3D_id);
}

void Reconstruction::DeleteObservation(const image_t image_id,
                                       const point2D_t point2D_idx) {
    // Note: Do not change order of these instructions, especially with respect
    // to `Reconstruction::ResetTriObservations`

    class Image& image = Image(image_id);
    const point3D_t point3D_id = image.Point2D(point2D_idx).Point3DId();
    class Point3D& point3D = Point3D(point3D_id);

    if (point3D.Track().Length() <= 2) {
        DeletePoint3D(point3D_id);
        return;
    }

    point3D.Track().DeleteElement(image_id, point2D_idx);

    const bool kIsDeletedPoint3D = false;
    ResetTriObservations(image_id, point2D_idx, kIsDeletedPoint3D);

    image.ResetPoint3DForPoint2D(point2D_idx);
}

void Reconstruction::DeleteAllPoints2DAndPoints3D() {
    points3D_.clear();
    for (auto& image : images_) {
        // Upstream parity: only clear the 2D points; images keep their
        // identity, poses, and back-pointers (a full Image rebuild would
        // drop the camera/frame wiring).
        image.second.SetPoints2D(std::vector<Eigen::Vector2d>(0));
        // Fork parity: reset the legacy observation counter as well.
        image.second.SetNumObservations(0);
    }
}

void Reconstruction::RegisterFrame(const frame_t frame_id) {
  // Upstream parity (d3ccaf35 scene/reconstruction.cc L638): registration
  // requires a pose and is idempotent. The fork keeps the image-level set in
  // lockstep for the images that already exist (upstream counts
  // num_reg_images_ here; the fork derives it from reg_image_ids_).
  const class Frame& frame = Frame(frame_id);
  THROW_CHECK(frame.HasPose());
  if (reg_frame_ids_.insert(frame_id).second) {
    for (const image_t image_id : frame.ImageIds()) {
      if (ExistsImage(image_id) && !Image(image_id).IsRegistered()) {
        Image(image_id).SetRegistered(true);
        reg_image_ids_.push_back(image_id);
      }
    }
  }
}

void Reconstruction::RegisterImage(const image_t image_id) {
    class Image& image = Image(image_id);
    if (!image.IsRegistered()) {
        image.SetRegistered(true);
        reg_image_ids_.push_back(image_id);
        // Frame-level lockstep (upstream registers at frame level only).
        if (image.HasFrameId() && ExistsFrame(image.FrameId()) &&
            Frame(image.FrameId()).HasPose()) {
            reg_frame_ids_.insert(image.FrameId());
        }
    }
}

void Reconstruction::DeRegisterImage(const image_t image_id) {
    class Image& image = Image(image_id);

    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
        if (image.Point2D(point2D_idx).HasPoint3D()) {
            DeleteObservation(image_id, point2D_idx);
        }
    }

    image.SetRegistered(false);

    reg_image_ids_.erase(
            std::remove(reg_image_ids_.begin(), reg_image_ids_.end(), image_id),
            reg_image_ids_.end());
    // Frame-level lockstep: drop the frame when none of its images is
    // registered anymore.
    if (image.HasFrameId() && ExistsFrame(image.FrameId())) {
        bool any_image_registered = false;
        for (const image_t other_id : Frame(image.FrameId()).ImageIds()) {
            if (ExistsImage(other_id) && Image(other_id).IsRegistered()) {
                any_image_registered = true;
                break;
            }
        }
        if (!any_image_registered) {
            reg_frame_ids_.erase(image.FrameId());
        }
    }
}

void Reconstruction::SetRigsAndFrames(std::vector<class Rig> rigs,
                                      std::vector<class Frame> frames) {
  rigs_.clear();
  rigs_.reserve(rigs.size());
  for (auto& rig : rigs) {
    AddRig(std::move(rig));
  }

  frames_.clear();
  frames_.reserve(frames.size());
  // Reset the registration state first, then rebuild it consistently: the
  // imported frames carry poses and are registered on insertion (upstream
  // parity; AddFrame registers posed frames), which also re-registers their
  // existing images in lockstep.
  reg_image_ids_.clear();
  reg_frame_ids_.clear();
  for (auto& [image_id, image] : images_) {
    image.SetRegistered(false);
  }
  NodeHashMap<image_t, frame_t> image_to_frame_ids;
  for (auto& frame : frames) {
    for (const image_t image_id : frame.ImageIds()) {
      THROW_CHECK(image_to_frame_ids.emplace(image_id, frame.FrameId()).second);
    }
    AddFrame(std::move(frame));
  }

  for (auto& [image_id, image] : images_) {
    image.ResetFramePtr();
    image.SetFrameId(image_to_frame_ids.at(image_id));
    image.SetFramePtr(&Frame(image.FrameId()));
  }
}

void Reconstruction::DeRegisterFrame(const frame_t frame_id) {
    if (!ExistsFrame(frame_id) || !Frame(frame_id).HasPose()) {
        LOG(WARNING) << "Ignoring de-registration of frame " << frame_id
                     << ", which is not registered.";
        return;
    }

    class Frame& frame = Frame(frame_id);
    for (const image_t image_id : frame.ImageIds()) {
        if (ExistsImage(image_id) && Image(image_id).IsRegistered()) {
            DeRegisterImage(image_id);
        }
    }
    reg_frame_ids_.erase(frame_id);
    frame.ResetPose();
}

void Reconstruction::Normalize(const double extent,
                               const double p0,
                               const double p1,
                               const bool use_images) {
    CHECK_GT(extent, 0);

    if ((use_images && reg_image_ids_.size() < 2) ||
        (!use_images && points3D_.size() < 2)) {
        return;
    }

    auto bound = ComputeBoundsAndCentroid(p0, p1, use_images);

    // Calculate scale and translation, such that
    // translation is applied before scaling.
    const double old_extent = (std::get<1>(bound) - std::get<0>(bound)).norm();
    double scale;
    if (old_extent < std::numeric_limits<double>::epsilon()) {
        scale = 1;
    } else {
        scale = extent / old_extent;
    }

    SimilarityTransform3 tform(scale, ComposeIdentityQuaternion(),
                               -scale * std::get<2>(bound));
    Transform(tform);
}

Eigen::Vector3d Reconstruction::ComputeCentroid(const double p0,
                                                const double p1) const {
    return std::get<2>(ComputeBoundsAndCentroid(p0, p1, false));
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> Reconstruction::ComputeBoundingBox(
        const double p0, const double p1) const {
    auto bound = ComputeBoundsAndCentroid(p0, p1, false);
    return std::make_pair(std::get<0>(bound), std::get<1>(bound));
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>
Reconstruction::ComputeBoundsAndCentroid(const double p0,
                                         const double p1,
                                         const bool use_images) const {
    CHECK_GE(p0, 0);
    CHECK_LE(p0, 1);
    CHECK_GE(p1, 0);
    CHECK_LE(p1, 1);
    CHECK_LE(p0, p1);

    const size_t num_elements =
            use_images ? reg_image_ids_.size() : points3D_.size();
    if (num_elements == 0) {
        return std::make_tuple(Eigen::Vector3d(0, 0, 0),
                               Eigen::Vector3d(0, 0, 0),
                               Eigen::Vector3d(0, 0, 0));
    }

    // Coordinates of image centers or point locations.
    // Upstream parity (d3ccaf35 scene/reconstruction.cc +
    // geometry/normalization.cc): collect doubles and delegate to the shared
    // ComputeBoundingBoxAndCentroid, whose percentile indices use
    // floor(min)/ceil(max). The old fork version truncated both indices
    // (P1 = trunc(0.9*(n-1)) dropped the largest camera for n=6) and stored
    // coordinates as floats, which shifted the normalization basis and
    // produced an exported gauge x1.37 off upstream.
    std::vector<double> coords_x;
    std::vector<double> coords_y;
    std::vector<double> coords_z;
    coords_x.reserve(num_elements);
    coords_y.reserve(num_elements);
    coords_z.reserve(num_elements);
    if (use_images) {
        for (const image_t im_id : reg_image_ids_) {
            const Eigen::Vector3d proj_center = Image(im_id).ProjectionCenter();
            coords_x.push_back(proj_center(0));
            coords_y.push_back(proj_center(1));
            coords_z.push_back(proj_center(2));
        }
    } else {
        coords_x.reserve(points3D_.size());
        coords_y.reserve(points3D_.size());
        coords_z.reserve(points3D_.size());
        for (const auto& point3D : points3D_) {
            coords_x.push_back(point3D.second.X());
            coords_y.push_back(point3D.second.Y());
            coords_z.push_back(point3D.second.Z());
        }
    }

    const auto [bbox, centroid] =
            ComputeBoundingBoxAndCentroid(p0, p1, std::move(coords_x),
                                          std::move(coords_y),
                                          std::move(coords_z));
    return std::make_tuple(bbox.min(), bbox.max(), centroid);
}

void Reconstruction::Transform(const SimilarityTransform3& tform) {
    // Upstream COLMAP dbb41680 parity: a single Transform implementation
    // updates the whole frame-aware object graph (rigs, frames, images,
    // points). The fork's legacy SimilarityTransform3 overload previously
    // only rewrote the per-image qvec/tvec members, leaving the frame and
    // rig poses stale.
    const Eigen::Vector4d qvec = tform.Rotation();  // [w, x, y, z]
    Transform(Sim3d(tform.Scale(),
                    Eigen::Quaterniond(qvec(0), qvec(1), qvec(2), qvec(3)),
                    tform.Translation()));
}

// Upstream COLMAP dbb41680 scene/reconstruction.cc parity: summary printing
// (required by the test matchers in scene/reconstruction_matchers.h).
std::ostream& operator<<(std::ostream& stream,
                         const Reconstruction& reconstruction) {
  stream << "Reconstruction(" << "num_rigs=" << reconstruction.NumRigs()
         << ", num_cameras=" << reconstruction.NumCameras()
         << ", num_frames=" << reconstruction.NumFrames()
         << ", num_reg_frames=" << reconstruction.NumRegFrames()
         << ", num_images=" << reconstruction.NumImages()
         << ", num_points3D=" << reconstruction.NumPoints3D() << ")";
  return stream;
}

// Upstream COLMAP dbb41680 scene/reconstruction.cc. The fork additionally
// keeps the legacy per-image qvec/tvec pose members in sync (the upstream
// 4.x model stores the pose on the frame only).
void Reconstruction::Transform(const Sim3d& new_from_old_world) {
    for (auto& [rig_id, rig] : rigs_) {
        (void)rig_id;
        for (const sensor_t& sensor_id : rig.SensorIds()) {
            if (rig.IsRefSensor(sensor_id) || !rig.HasSensorFromRig(sensor_id)) {
                continue;
            }
            Rigid3d sensor_from_rig = rig.SensorFromRig(sensor_id);
            sensor_from_rig.translation() *= new_from_old_world.scale();
            // Update-in-place semantics: the sensor already exists in the
            // rig (upstream mutates the optional Rigid3d reference).
            rig.SetSensorFromRig(sensor_id, sensor_from_rig);
        }
    }
    for (auto& [frame_id, frame] : frames_) {
        (void)frame_id;
        if (frame.HasPose()) {
            const Rigid3d transformed = TransformCameraWorld(
                    new_from_old_world, frame.RigFromWorld());
            frame.SetRigFromWorld(transformed);
        }
    }
    for (auto& [image_id, image] : images_) {
        (void)image_id;
        // Frame-wired images without a pose have nothing to transform. All
        // other images (frame-wired with pose, or fork-legacy standalone
        // images whose pose lives only in qvec/tvec) need their legacy
        // members transformed; frame-wired ones keep them in sync with the
        // frame pose updated above.
        if (image.HasFramePtr() && !image.HasPose()) {
            continue;
        }
        // Fork qvec convention is [w, x, y, z]: construct from the four
        // scalars (Eigen's Vector4d constructor assumes [x, y, z, w]).
        const Eigen::Vector4d& qvec = image.Qvec();
        const Rigid3d cam_from_world = TransformCameraWorld(
                new_from_old_world,
                Rigid3d(Eigen::Quaterniond(qvec(0), qvec(1), qvec(2),
                                           qvec(3)),
                        image.Tvec()));
        const Eigen::Quaterniond& q = cam_from_world.rotation();
        image.SetQvec(Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()));
        image.SetTvec(cam_from_world.translation());
    }
    for (auto& point3D : points3D_) {
        point3D.second.XYZ() = new_from_old_world * point3D.second.XYZ();
    }
}

Reconstruction Reconstruction::Crop(
        const std::pair<Eigen::Vector3d, Eigen::Vector3d>& bbox) const {
    // add all cameras and images. Only the registered images will be used.
    Reconstruction reconstruction;
    for (const auto& camera_el : cameras_) {
        reconstruction.AddCameraWithTrivialRig(camera_el.second);
    }
    for (const auto& image_el : images_) {
        // The copied image carries back pointers into *this* reconstruction;
        // reset them so AddImage re-wires them into the cropped copy. The
        // trivial frame is seeded from the image's legacy pose.
        class Image image = image_el.second;
        image.ResetCameraPtr();
        image.ResetFramePtr();
        reconstruction.AddImageWithTrivialFrame(image);
        auto& cropped_image = reconstruction.Image(image_el.first);
        cropped_image.SetRegistered(false);
        for (point2D_t pid = 0; pid < cropped_image.NumPoints2D(); ++pid) {
            cropped_image.ResetPoint3DForPoint2D(pid);
        }
    }
    for (const auto& point_el : points3D_) {
        const auto& point = point_el.second;
        if ((point.XYZ().array() >= bbox.first.array()).all() &&
            (point.XYZ().array() <= bbox.second.array()).all()) {
            for (const auto& track_el : point.Track().Elements()) {
                reconstruction.RegisterImage(track_el.image_id);
            }
            reconstruction.AddPoint3D(point.XYZ(), point.Track(),
                                      point.Color());
        }
    }
    return reconstruction;
}

bool Reconstruction::Merge(const Reconstruction& reconstruction,
                           const double max_reproj_error) {
    const double kMinInlierObservations = 0.3;

    Eigen::Matrix3x4d alignment;
    if (!ComputeAlignmentBetweenReconstructions(reconstruction, *this,
                                                kMinInlierObservations,
                                                max_reproj_error, &alignment)) {
        return false;
    }

    const SimilarityTransform3 tform(alignment);

    // Find common and missing images in the two reconstructions.

    std::unordered_set<image_t> common_image_ids;
    common_image_ids.reserve(reconstruction.NumRegImages());
    std::unordered_set<image_t> missing_image_ids;
    missing_image_ids.reserve(reconstruction.NumRegImages());

    for (const auto& image_id : reconstruction.RegImageIds()) {
        if (ExistsImage(image_id)) {
            common_image_ids.insert(image_id);
        } else {
            missing_image_ids.insert(image_id);
        }
    }

    // Register the missing images in this reconstruction.

    for (const auto image_id : missing_image_ids) {
        auto reg_image = reconstruction.Image(image_id);
        if (!ExistsCamera(reg_image.CameraId())) {
            AddCameraWithTrivialRig(reconstruction.Camera(reg_image.CameraId()));
        } else if (!ExistsRig(reg_image.CameraId())) {
            class Rig rig;
            rig.SetRigId(reg_image.CameraId());
            rig.AddRefSensor(
                sensor_t(SensorType::CAMERA, reg_image.CameraId()));
            AddRig(std::move(rig));
        }
        // Reset the source reconstruction's back pointers so AddImage re-wires
        // them here; the trivial frame is seeded from the legacy pose.
        reg_image.ResetCameraPtr();
        reg_image.ResetFramePtr();
        reg_image.SetRegistered(false);
        AddImageWithTrivialFrame(reg_image);
        RegisterImage(image_id);
        auto& image = Image(image_id);
        tform.TransformPose(&image.Qvec(), &image.Tvec());
    }

    // Merge the two point clouds using the following two rules:
    //    - copy points to this reconstruction with non-conflicting tracks,
    //      i.e. points that do not have an already triangulated observation
    //      in this reconstruction.
    //    - merge tracks that are unambiguous, i.e. only merge points in the two
    //      reconstructions if they have a one-to-one mapping.
    // Note that in both cases no cheirality or reprojection test is performed.

    for (const auto& point3D : reconstruction.Points3D()) {
        Track new_track;
        Track old_track;
        std::unordered_set<point3D_t> old_point3D_ids;
        for (const auto& track_el : point3D.second.Track().Elements()) {
            if (common_image_ids.count(track_el.image_id) > 0) {
                const auto& point2D =
                        Image(track_el.image_id).Point2D(track_el.point2D_idx);
                if (point2D.HasPoint3D()) {
                    old_track.AddElement(track_el);
                    old_point3D_ids.insert(point2D.Point3DId());
                } else {
                    new_track.AddElement(track_el);
                }
            } else if (missing_image_ids.count(track_el.image_id) > 0) {
                Image(track_el.image_id)
                        .ResetPoint3DForPoint2D(track_el.point2D_idx);
                new_track.AddElement(track_el);
            }
        }

        const bool create_new_point = new_track.Length() >= 2;
        const bool merge_new_and_old_point =
                (new_track.Length() + old_track.Length()) >= 2 &&
                old_point3D_ids.size() == 1;
        if (create_new_point || merge_new_and_old_point) {
            Eigen::Vector3d xyz = point3D.second.XYZ();
            tform.TransformPoint(&xyz);
            const auto point3D_id =
                    AddPoint3D(xyz, new_track, point3D.second.Color());
            if (old_point3D_ids.size() == 1) {
                MergePoints3D(point3D_id, *old_point3D_ids.begin());
            }
        }
    }

    FilterPoints3DWithLargeReprojectionError(max_reproj_error, Point3DIds());

    return true;
}

const class Image* Reconstruction::FindImageWithName(
        const std::string& name) const {
    for (const auto& image : images_) {
        if (image.second.Name() == name) {
            return &image.second;
        }
    }
    return nullptr;
}

// Upstream COLMAP dbb41680 semantics: common registered images matched by
// name, returned as (this_id, other_id) pairs.
std::vector<std::pair<image_t, image_t>> Reconstruction::FindCommonRegImageIds(
        const Reconstruction& other) const {
    std::vector<std::pair<image_t, image_t>> common_reg_image_ids;
    for (const image_t image_id : reg_image_ids_) {
        const class Image& image = Image(image_id);
        const class Image* other_image = other.FindImageWithName(image.Name());
        if (other_image != nullptr && other_image->HasPose()) {
            common_reg_image_ids.emplace_back(image.ImageId(),
                                              other_image->ImageId());
        }
    }
    return common_reg_image_ids;
}

void Reconstruction::TranscribeImageIdsToDatabase(const Database& database) {
    std::unordered_map<image_t, image_t> old_to_new_image_ids;
    old_to_new_image_ids.reserve(NumImages());

    std::unordered_map<image_t, class Image> new_images;
    new_images.reserve(NumImages());

    for (auto& image : images_) {
        if (!database.ExistsImageWithName(image.second.Name())) {
            LOG(FATAL) << "Image with name " << image.second.Name()
                       << " does not exist in database";
        }

        const auto database_image =
                database.ReadImageWithName(image.second.Name());
        old_to_new_image_ids.emplace(image.second.ImageId(),
                                     database_image.ImageId());
        image.second.SetImageId(database_image.ImageId());
        new_images.emplace(database_image.ImageId(), image.second);
    }

    images_ = std::move(new_images);

    // Upstream parity (d3ccaf35 scene/reconstruction.cc): transcribe the
    // frame data ids, otherwise the frames keep referencing the old image
    // ids and every frame-to-image lookup (e.g. ExtractImageToFramePtr,
    // correspondence-graph bookkeeping) misses the transcribed images.
    for (auto& [_, frame] : frames_) {
        class Frame new_frame = frame;
        new_frame.ClearDataIds();
        for (const data_t& data_id : frame.DataIds()) {
            data_t transcribed_id = data_id;
            if (data_id.sensor_id.type == SensorType::CAMERA) {
                transcribed_id.id = old_to_new_image_ids.at(data_id.id);
            }
            new_frame.AddDataId(transcribed_id);
        }
        frame = std::move(new_frame);
    }

    // Upstream parity: the Image copy constructor resets the camera/frame
    // back-pointers (they refer to the source container), so they must be
    // re-wired against the surviving rigs_/frames_/cameras_ after the
    // transcribed images are swapped in; otherwise every later CameraPtr()
    // dereference throws (point_triangulator --clear_points crash).
    RewireObjectPointers();

    for (auto& image_id : reg_image_ids_) {
        image_id = old_to_new_image_ids.at(image_id);
    }

    for (auto& point3D : points3D_) {
        for (auto& track_el : point3D.second.Track().Elements()) {
            track_el.image_id = old_to_new_image_ids.at(track_el.image_id);
        }
    }
}

size_t Reconstruction::FilterPoints3D(
        const double max_reproj_error,
        const double min_tri_angle,
        const std::unordered_set<point3D_t>& point3D_ids) {
    size_t num_filtered = 0;
    num_filtered += FilterPoints3DWithLargeReprojectionError(max_reproj_error,
                                                             point3D_ids);
    num_filtered += FilterPoints3DWithSmallTriangulationAngle(min_tri_angle,
                                                              point3D_ids);
    return num_filtered;
}

size_t Reconstruction::FilterPoints3DInImages(
        const double max_reproj_error,
        const double min_tri_angle,
        const std::unordered_set<image_t>& image_ids) {
    std::unordered_set<point3D_t> point3D_ids;
    for (const image_t image_id : image_ids) {
        const class Image& image = Image(image_id);
        for (const Point2D& point2D : image.Points2D()) {
            if (point2D.HasPoint3D()) {
                point3D_ids.insert(point2D.Point3DId());
            }
        }
    }
    return FilterPoints3D(max_reproj_error, min_tri_angle, point3D_ids);
}

size_t Reconstruction::FilterAllPoints3D(const double max_reproj_error,
                                         const double min_tri_angle) {
    // Important: First filter observations and points with large reprojection
    // error, so that observations with large reprojection error do not make
    // a point stable through a large triangulation angle.
    const std::unordered_set<point3D_t>& point3D_ids = Point3DIds();
    size_t num_filtered = 0;
    num_filtered += FilterPoints3DWithLargeReprojectionError(max_reproj_error,
                                                             point3D_ids);
    num_filtered += FilterPoints3DWithSmallTriangulationAngle(min_tri_angle,
                                                              point3D_ids);
    return num_filtered;
}

size_t Reconstruction::FilterObservationsWithNegativeDepth() {
    size_t num_filtered = 0;
    for (const auto image_id : reg_image_ids_) {
        const class Image& image = Image(image_id);
        const Eigen::Matrix3x4d proj_matrix = image.ProjectionMatrix();
        for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
             ++point2D_idx) {
            const Point2D& point2D = image.Point2D(point2D_idx);
            if (point2D.HasPoint3D()) {
                const class Point3D& point3D = Point3D(point2D.Point3DId());
                if (!HasPointPositiveDepth(proj_matrix, point3D.XYZ())) {
                    DeleteObservation(image_id, point2D_idx);
                    num_filtered += 1;
                }
            }
        }
    }
    return num_filtered;
}

std::vector<image_t> Reconstruction::FilterImages(
        const double min_focal_length_ratio,
        const double max_focal_length_ratio,
        const double max_extra_param) {
    std::vector<image_t> filtered_image_ids;
    for (const image_t image_id : RegImageIds()) {
        const class Image& image = Image(image_id);
        const class Camera& camera = Camera(image.CameraId());
        if (image.NumPoints3D() == 0) {
            filtered_image_ids.push_back(image_id);
        } else if (camera.HasBogusParams(min_focal_length_ratio,
                                         max_focal_length_ratio,
                                         max_extra_param)) {
            filtered_image_ids.push_back(image_id);
        }
    }

    // Only de-register after iterating over reg_image_ids_ to avoid
    // simultaneous iteration and modification of the vector.
    for (const image_t image_id : filtered_image_ids) {
        DeRegisterImage(image_id);
    }

    return filtered_image_ids;
}

size_t Reconstruction::ComputeNumObservations() const {
    size_t num_obs = 0;
    for (const image_t image_id : reg_image_ids_) {
        num_obs += Image(image_id).NumPoints3D();
    }
    return num_obs;
}

double Reconstruction::ComputeMeanTrackLength() const {
    if (points3D_.empty()) {
        return 0.0;
    } else {
        return ComputeNumObservations() / static_cast<double>(points3D_.size());
    }
}

double Reconstruction::ComputeMeanObservationsPerRegImage() const {
    if (reg_image_ids_.empty()) {
        return 0.0;
    } else {
        return ComputeNumObservations() /
               static_cast<double>(reg_image_ids_.size());
    }
}

double Reconstruction::ComputeMeanReprojectionError() const {
    double error_sum = 0.0;
    size_t num_valid_errors = 0;
    for (const auto& point3D : points3D_) {
        if (point3D.second.HasError()) {
            error_sum += point3D.second.Error();
            num_valid_errors += 1;
        }
    }

    if (num_valid_errors == 0) {
        return 0.0;
    } else {
        return error_sum / num_valid_errors;
    }
}

void Reconstruction::Read(const std::filesystem::path& path) {
    if (ExistsFile(path / "cameras.bin") &&
        ExistsFile(path / "images.bin") &&
        ExistsFile(path / "points3D.bin")) {
        ReadBinary(path);
    } else if (ExistsFile(path / "cameras.txt") &&
               ExistsFile(path / "images.txt") &&
               ExistsFile(path / "points3D.txt")) {
        ReadText(path);
    } else {
        LOG(FATAL) << "cameras, images, points3D files do not exist at "
                   << path;
    }
}

void Reconstruction::Write(const std::filesystem::path& path) const {
    WriteBinary(path);
}

void Reconstruction::ReadText(const std::filesystem::path& path) {
    cameras_.clear();
    rigs_.clear();
    frames_.clear();
    images_.clear();
    points3D_.clear();
    ReadCamerasText(*this, path / "cameras.txt");
    const auto rigs_path = path / "rigs.txt";
    if (ExistsFile(rigs_path)) {
        ReadRigsText(*this, rigs_path);
    }
    const auto frames_path = path / "frames.txt";
    if (ExistsFile(frames_path)) {
        ReadFramesText(*this, frames_path);
    }
    ReadImagesText(*this, path / "images.txt");
    ReadPoints3DText(*this, path / "points3D.txt");
}

void Reconstruction::ReadBinary(const std::filesystem::path& path) {
    cameras_.clear();
    rigs_.clear();
    frames_.clear();
    images_.clear();
    points3D_.clear();
    ReadCamerasBinary(*this, path / "cameras.bin");
    const auto rigs_path = path / "rigs.bin";
    if (ExistsFile(rigs_path)) {
        ReadRigsBinary(*this, rigs_path);
    }
    const auto frames_path = path / "frames.bin";
    if (ExistsFile(frames_path)) {
        ReadFramesBinary(*this, frames_path);
    }
    ReadImagesBinary(*this, path / "images.bin");
    ReadPoints3DBinary(*this, path / "points3D.bin");

}

void Reconstruction::WriteText(const std::filesystem::path& path) const {
    THROW_CHECK(ExistsDir(path))
            << "Directory does not exist: " << path;
    WriteRigsText(*this, path / "rigs.txt");
    WriteCamerasText(*this, path / "cameras.txt");
    WriteFramesText(*this, path / "frames.txt");
    WriteImagesText(*this, path / "images.txt");
    WritePoints3DText(*this, path / "points3D.txt");
}

void Reconstruction::WriteBinary(const std::filesystem::path& path) const {
    THROW_CHECK(ExistsDir(path))
            << "Directory does not exist: " << path;
    WriteRigsBinary(*this, path / "rigs.bin");
    WriteCamerasBinary(*this, path / "cameras.bin");
    WriteFramesBinary(*this, path / "frames.bin");
    WriteImagesBinary(*this, path / "images.bin");
    WritePoints3DBinary(*this, path / "points3D.bin");
}









std::vector<PlyPoint> Reconstruction::ConvertToPLY() const {
    std::vector<PlyPoint> ply_points;
    ply_points.reserve(points3D_.size());

    for (const auto& point3D : points3D_) {
        PlyPoint ply_point;
        ply_point.x = point3D.second.X();
        ply_point.y = point3D.second.Y();
        ply_point.z = point3D.second.Z();
        ply_point.r = point3D.second.Color(0);
        ply_point.g = point3D.second.Color(1);
        ply_point.b = point3D.second.Color(2);
        ply_points.push_back(ply_point);
    }

    return ply_points;
}

void Reconstruction::ImportPLY(const std::filesystem::path& path) {
    points3D_.clear();

    const auto ply_points = ReadPly(path);

    points3D_.reserve(ply_points.size());

    for (const auto& ply_point : ply_points) {
        AddPoint3D(Eigen::Vector3d(ply_point.x, ply_point.y, ply_point.z),
                   Track(),
                   Eigen::Vector3ub(ply_point.r, ply_point.g, ply_point.b));
    }
}

void Reconstruction::ImportPLY(const std::vector<PlyPoint>& ply_points) {
    points3D_.clear();
    points3D_.reserve(ply_points.size());
    for (const auto& ply_point : ply_points) {
        AddPoint3D(Eigen::Vector3d(ply_point.x, ply_point.y, ply_point.z),
                   Track(),
                   Eigen::Vector3ub(ply_point.r, ply_point.g, ply_point.b));
    }
}







bool Reconstruction::ExtractColorsForImage(const image_t image_id,
                                           const std::filesystem::path& path) {
    const class Image& image = Image(image_id);

    Bitmap bitmap;
    if (!bitmap.Read(path / image.Name())) {
        return false;
    }

    const Eigen::Vector3ub kBlackColor(0, 0, 0);
    for (const Point2D& point2D : image.Points2D()) {
        if (point2D.HasPoint3D()) {
            class Point3D& point3D = Point3D(point2D.Point3DId());
            if (point3D.Color() == kBlackColor) {
                BitmapColor<float> color;
                // COLMAP assumes that the upper left pixel center is (0.5,
                // 0.5).
                if (bitmap.InterpolateBilinear(point2D.X() - 0.5,
                                               point2D.Y() - 0.5, &color)) {
                    const BitmapColor<uint8_t> color_ub = color.Cast<uint8_t>();
                    point3D.SetColor(Eigen::Vector3ub(color_ub.r, color_ub.g,
                                                      color_ub.b));
                }
            }
        }
    }

    return true;
}

void Reconstruction::ExtractColorsForAllImages(
        const std::filesystem::path& path) {
    std::unordered_map<point3D_t, Eigen::Vector3d> color_sums;
    std::unordered_map<point3D_t, size_t> color_counts;

    for (size_t i = 0; i < reg_image_ids_.size(); ++i) {
        const class Image& image = Image(reg_image_ids_[i]);
        const std::filesystem::path image_path = path / image.Name();

        Bitmap bitmap;
        if (!bitmap.Read(image_path)) {
            std::cout << StringPrintf("Could not read image %s at path %s.",
                                      image.Name().c_str(),
                                      image_path.string().c_str())
                      << std::endl;
            continue;
        }

        for (const Point2D& point2D : image.Points2D()) {
            if (point2D.HasPoint3D()) {
                BitmapColor<float> color;
                // COLMAP assumes that the upper left pixel center is (0.5,
                // 0.5).
                if (bitmap.InterpolateBilinear(point2D.X() - 0.5,
                                               point2D.Y() - 0.5, &color)) {
                    if (color_sums.count(point2D.Point3DId())) {
                        Eigen::Vector3d& color_sum =
                                color_sums[point2D.Point3DId()];
                        color_sum(0) += color.r;
                        color_sum(1) += color.g;
                        color_sum(2) += color.b;
                        color_counts[point2D.Point3DId()] += 1;
                    } else {
                        color_sums.emplace(
                                point2D.Point3DId(),
                                Eigen::Vector3d(color.r, color.g, color.b));
                        color_counts.emplace(point2D.Point3DId(), 1);
                    }
                }
            }
        }
    }

    const Eigen::Vector3ub kBlackColor = Eigen::Vector3ub::Zero();
    for (auto& point3D : points3D_) {
        if (color_sums.count(point3D.first)) {
            Eigen::Vector3d color =
                    color_sums[point3D.first] / color_counts[point3D.first];
            for (Eigen::Index i = 0; i < color.size(); ++i) {
                color[i] = std::round(color[i]);
            }
            point3D.second.SetColor(color.cast<uint8_t>());
        } else {
            point3D.second.SetColor(kBlackColor);
        }
    }
}

void Reconstruction::CreateImageDirs(const std::filesystem::path& path) const {
    std::set<std::filesystem::path> image_dirs;
    for (const auto& image : images_) {
        const std::vector<std::string> name_split =
                StringSplit(image.second.Name(), "/");
        if (name_split.size() > 1) {
            std::filesystem::path dir = path;
            for (size_t i = 0; i < name_split.size() - 1; ++i) {
                dir = dir / name_split[i];
                image_dirs.insert(dir);
            }
        }
    }
    for (const auto& dir : image_dirs) {
        CreateDirIfNotExists(dir, /*recursive=*/true);
    }
}

size_t Reconstruction::FilterPoints3DWithSmallTriangulationAngle(
        const double min_tri_angle,
        const std::unordered_set<point3D_t>& point3D_ids) {
    // Number of filtered points.
    size_t num_filtered = 0;

    // Minimum triangulation angle in radians.
    const double min_tri_angle_rad = DegToRad(min_tri_angle);

    // Cache for image projection centers.
    std::unordered_map<image_t, Eigen::Vector3d> proj_centers;

    for (const auto point3D_id : point3D_ids) {
        if (!ExistsPoint3D(point3D_id)) {
            continue;
        }

        const class Point3D& point3D = Point3D(point3D_id);

        // Calculate triangulation angle for all pairwise combinations of image
        // poses in the track. Only delete point if none of the combinations
        // has a sufficient triangulation angle.
        bool keep_point = false;
        for (size_t i1 = 0; i1 < point3D.Track().Length(); ++i1) {
            const image_t image_id1 = point3D.Track().Element(i1).image_id;

            Eigen::Vector3d proj_center1;
            if (proj_centers.count(image_id1) == 0) {
                const class Image& image1 = Image(image_id1);
                proj_center1 = image1.ProjectionCenter();
                proj_centers.emplace(image_id1, proj_center1);
            } else {
                proj_center1 = proj_centers.at(image_id1);
            }

            for (size_t i2 = 0; i2 < i1; ++i2) {
                const image_t image_id2 = point3D.Track().Element(i2).image_id;
                const Eigen::Vector3d proj_center2 = proj_centers.at(image_id2);

                const double tri_angle = CalculateTriangulationAngle(
                        proj_center1, proj_center2, point3D.XYZ());

                if (tri_angle >= min_tri_angle_rad) {
                    keep_point = true;
                    break;
                }
            }

            if (keep_point) {
                break;
            }
        }

        if (!keep_point) {
            num_filtered += 1;
            DeletePoint3D(point3D_id);
        }
    }

    return num_filtered;
}

size_t Reconstruction::FilterPoints3DWithLargeReprojectionError(
        const double max_reproj_error,
        const std::unordered_set<point3D_t>& point3D_ids) {
    const double max_squared_reproj_error = max_reproj_error * max_reproj_error;

    // Number of filtered points.
    size_t num_filtered = 0;

    for (const auto point3D_id : point3D_ids) {
        if (!ExistsPoint3D(point3D_id)) {
            continue;
        }

        class Point3D& point3D = Point3D(point3D_id);

        if (point3D.Track().Length() < 2) {
            DeletePoint3D(point3D_id);
            num_filtered += point3D.Track().Length();
            continue;
        }

        double reproj_error_sum = 0.0;

        std::vector<TrackElement> track_els_to_delete;

        for (const auto& track_el : point3D.Track().Elements()) {
            const class Image& image = Image(track_el.image_id);
            const class Camera& camera = Camera(image.CameraId());
            const Point2D& point2D = image.Point2D(track_el.point2D_idx);
            const double squared_reproj_error =
                    CalculateSquaredReprojectionError(
                            point2D.XY(), point3D.XYZ(), image.Qvec(),
                            image.Tvec(), camera);
            if (squared_reproj_error > max_squared_reproj_error) {
                track_els_to_delete.push_back(track_el);
            } else {
                reproj_error_sum += std::sqrt(squared_reproj_error);
            }
        }

        if (track_els_to_delete.size() >= point3D.Track().Length() - 1) {
            num_filtered += point3D.Track().Length();
            DeletePoint3D(point3D_id);
        } else {
            num_filtered += track_els_to_delete.size();
            for (const auto& track_el : track_els_to_delete) {
                DeleteObservation(track_el.image_id, track_el.point2D_idx);
            }
            point3D.SetError(reproj_error_sum / point3D.Track().Length());
        }
    }

    return num_filtered;
}













void Reconstruction::SetObservationAsTriangulated(
        const image_t image_id,
        const point2D_t point2D_idx,
        const bool is_continued_point3D) {
    if (correspondence_graph_ == nullptr) {
        return;
    }

    const class Image& image = Image(image_id);
    const Point2D& point2D = image.Point2D(point2D_idx);
    const CorrespondenceGraph::CorrespondenceRange corrs =
            correspondence_graph_->FindCorrespondences(image_id, point2D_idx);

    CHECK(image.IsRegistered());
    CHECK(point2D.HasPoint3D());

    for (const CorrespondenceGraph::Correspondence* corr = corrs.beg;
         corr < corrs.end; ++corr) {
        class Image& corr_image = Image(corr->image_id);
        const Point2D& corr_point2D = corr_image.Point2D(corr->point2D_idx);
        corr_image.IncrementCorrespondenceHasPoint3D(corr->point2D_idx);
        // Update number of shared 3D points between image pairs and make sure
        // to only count the correspondences once (not twice forward and
        // backward).
        if (point2D.Point3DId() == corr_point2D.Point3DId() &&
            (is_continued_point3D || image_id < corr->image_id)) {
            const image_pair_t pair_id =
                    Database::ImagePairToPairId(image_id, corr->image_id);
            image_pair_stats_[pair_id].num_tri_corrs += 1;
            CHECK_LE(image_pair_stats_[pair_id].num_tri_corrs,
                     image_pair_stats_[pair_id].num_total_corrs)
                    << "The correspondence graph graph must not contain "
                       "duplicate "
                       "matches";
        }
    }
}

void Reconstruction::ResetTriObservations(const image_t image_id,
                                          const point2D_t point2D_idx,
                                          const bool is_deleted_point3D) {
    if (correspondence_graph_ == nullptr) {
        return;
    }

    const class Image& image = Image(image_id);
    const Point2D& point2D = image.Point2D(point2D_idx);
    const CorrespondenceGraph::CorrespondenceRange corrs =
            correspondence_graph_->FindCorrespondences(image_id, point2D_idx);

    CHECK(image.IsRegistered());
    CHECK(point2D.HasPoint3D());

    for (const CorrespondenceGraph::Correspondence* corr = corrs.beg;
         corr < corrs.end; ++corr) {
        class Image& corr_image = Image(corr->image_id);
        const Point2D& corr_point2D = corr_image.Point2D(corr->point2D_idx);
        corr_image.DecrementCorrespondenceHasPoint3D(corr->point2D_idx);
        // Update number of shared 3D points between image pairs and make sure
        // to only count the correspondences once (not twice forward and
        // backward).
        if (point2D.Point3DId() == corr_point2D.Point3DId() &&
            (!is_deleted_point3D || image_id < corr->image_id)) {
            const image_pair_t pair_id =
                    Database::ImagePairToPairId(image_id, corr->image_id);
            image_pair_stats_[pair_id].num_tri_corrs -= 1;
            CHECK_GE(image_pair_stats_[pair_id].num_tri_corrs, 0)
                    << "The scene graph graph must not contain duplicate "
                       "matches";
        }
    }
}


std::unordered_set<frame_t> Reconstruction::RegFrameIds() const {
  // Upstream d3ccaf35 semantics: registration is an explicit state held in
  // reg_frame_ids_; AddFrame registers frames that carry a pose, and
  // RegisterImage keeps the legacy image-level set in lockstep. The fork
  // database round-trip persists poses (a fork extension), so the disk-read
  // path calls ResetRegistrationState() to start unregistered.
  return reg_frame_ids_;
}

void Reconstruction::ResetRegistrationState() {
  reg_frame_ids_.clear();
  reg_image_ids_.clear();
  for (auto& [image_id, image] : images_) {
    image.SetRegistered(false);
  }
}

void Reconstruction::AddCameraWithTrivialRig(class Camera camera) {
  THROW_CHECK(!ExistsRig(camera.CameraId()))
      << "AddCameraWithTrivialRig tried to add a rig with the same id as the "
         "camera, but failed because Rig "
      << camera.CameraId() << "already exists in the reconstruction. ";
  class Rig rig;
  rig.SetRigId(camera.CameraId());
  rig.AddRefSensor(camera.SensorId());
  AddCamera(std::move(camera));
  AddRig(std::move(rig));
}


void Reconstruction::AddImageWithTrivialFrame(class Image image) {
  THROW_CHECK(!ExistsFrame(image.ImageId()))
      << "AddImageWithTrivialFrame tried to add a frame with the same id as "
         "the image, but failed because Frame "
      << image.ImageId() << "already exists in the reconstruction.";
  THROW_CHECK(ExistsRig(image.CameraId()))
      << "Rig " << image.CameraId() << " that contains Camera "
      << image.CameraId() << " does not exist in the reconstruction.";
  auto& rig = Rig(image.CameraId());
  THROW_CHECK_EQ(rig.NumSensors(), 1)
      << "AddImageWithTrivialFrame requires that the camera is from a rig that "
         "contains exactly one sensor (the camera itself).";
  THROW_CHECK(rig.IsRefSensor(Camera(image.CameraId()).SensorId()));
  class Frame frame;
  frame.SetFrameId(image.ImageId());
  frame.SetRigId(image.CameraId());
  frame.AddDataId(image.DataId());
  if (image.HasFrameId()) {
    THROW_CHECK_EQ(image.FrameId(), frame.FrameId());
  } else {
    image.SetFrameId(frame.FrameId());
  }
  // Seed the frame pose from the image's legacy qvec/tvec pose so the
  // trivial frame mirrors the image-level pose of the legacy model.
  frame.SetRigFromWorld(image.Qvec(), image.Tvec());
  AddFrame(std::move(frame));
  AddImage(std::move(image));
}

void Reconstruction::AddImageWithTrivialFrame(class Image image,
                                              const Rigid3d& cam_from_world) {
  const frame_t frame_id = image.ImageId();
  AddImageWithTrivialFrame(std::move(image));
  Frame(frame_id).SetRigFromWorld(cam_from_world);
  // Frame-level registration lands with W3-2b; image-level
        // registration below keeps the legacy model.
        RegisterImage(frame_id);
}


void Reconstruction::UpdatePoint3DErrors() {
  for (auto& [_, point3D] : points3D_) {
    if (point3D.Track().Length() == 0) {
      point3D.SetError(0);
      continue;
    }
    double error_sum = 0;
    for (const auto& track_el : point3D.Track().Elements()) {
      const auto& image = Image(track_el.image_id);
      const auto& point2D = image.Point2D(track_el.point2D_idx);
      const auto& camera = *image.CameraPtr();
      error_sum += std::sqrt(CalculateSquaredReprojectionError(
          point2D.XY(), point3D.XYZ(), image.CamFromWorld(), camera));
    }
    point3D.SetError(error_sum / point3D.Track().Length());
  }
}
}  // namespace colmap
