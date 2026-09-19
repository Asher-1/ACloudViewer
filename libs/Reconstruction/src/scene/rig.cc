// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------

#include "scene/rig.h"

#include <istream>
#include <limits>
#include <ostream>
#include <utility>

#include "util/endian.h"
#include "util/hash_containers.h"
#include "util/logging.h"
#include "util/string.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <functional>
#include <set>

#include "sensor/models.h"
#include "scene/database.h"
#include "scene/frame.h"
#include "scene/image.h"
#include "geometry/pose.h"
#include "scene/reconstruction.h"

namespace colmap {

namespace {

constexpr uint32_t kRigBinaryMagic = 0x52475631;  // RGV1
constexpr uint32_t kRigFormatVersion = 2;
constexpr uint64_t kMaxSerializedRigSensors = 1000000;

bool IsValidPose(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec) {
    return qvec.allFinite() && tvec.allFinite() &&
           qvec.squaredNorm() > std::numeric_limits<double>::epsilon();
}

}  // namespace

rig_t Rig::RigId() const { return rig_id_; }

void Rig::SetRigId(const rig_t rig_id) {
    CHECK_NE(rig_id, kInvalidRigId);
    rig_id_ = rig_id;
}

void Rig::AddRefSensor(const sensor_t& sensor_id) {
    CHECK_EQ(static_cast<int>(ref_sensor_id_.type), static_cast<int>(SensorType::INVALID));
    CHECK_NE(static_cast<int>(sensor_id.type), static_cast<int>(SensorType::INVALID));
    CHECK_NE(sensor_id.id, std::numeric_limits<uint32_t>::max());
    CHECK(!HasSensor(sensor_id));
    ref_sensor_id_ = sensor_id;
    sensors_.emplace(sensor_id, std::nullopt);
}

void Rig::AddSensor(const sensor_t& sensor_id,
                    const std::optional<Eigen::Vector4d>& sensor_from_rig_qvec,
                    const std::optional<Eigen::Vector3d>& sensor_from_rig_tvec) {
    CHECK_NE(static_cast<int>(ref_sensor_id_.type), static_cast<int>(SensorType::INVALID));
    CHECK_NE(static_cast<int>(sensor_id.type), static_cast<int>(SensorType::INVALID));
    CHECK(!HasSensor(sensor_id));
    CHECK_EQ(sensor_from_rig_qvec.has_value(), sensor_from_rig_tvec.has_value());
    if (sensor_from_rig_qvec) {
        CHECK(IsValidPose(*sensor_from_rig_qvec, *sensor_from_rig_tvec));
        sensors_.emplace(sensor_id, SensorPose{NormalizeQuaternion(*sensor_from_rig_qvec),
                                                *sensor_from_rig_tvec});
    } else {
        sensors_.emplace(sensor_id, std::nullopt);
    }
}

bool Rig::HasSensor(const sensor_t& sensor_id) const {
    return sensors_.count(sensor_id) != 0;
}

size_t Rig::NumSensors() const { return sensors_.size(); }

bool Rig::IsRefSensor(const sensor_t& sensor_id) const {
    return sensor_id == ref_sensor_id_;
}

const sensor_t& Rig::RefSensorId() const { return ref_sensor_id_; }

std::vector<sensor_t> Rig::SensorIds() const {
    std::vector<sensor_t> sensor_ids;
    sensor_ids.reserve(sensors_.size());
    for (const auto& sensor : sensors_) sensor_ids.push_back(sensor.first);
    return sensor_ids;
}

bool Rig::HasSensorFromRig(const sensor_t& sensor_id) const {
    const auto it = sensors_.find(sensor_id);
    return sensor_id != ref_sensor_id_ && it != sensors_.end() &&
           it->second.has_value();
}

const Eigen::Vector4d& Rig::SensorFromRigQvec(const sensor_t& sensor_id) const {
    return sensors_.at(sensor_id).value().qvec;
}

const Eigen::Vector3d& Rig::SensorFromRigTvec(const sensor_t& sensor_id) const {
    return sensors_.at(sensor_id).value().tvec;
}

void Rig::AddRefCamera(const camera_t camera_id) {
    AddRefSensor(sensor_t(SensorType::CAMERA, camera_id));
}

void Rig::AddCamera(const camera_t camera_id,
                    const Eigen::Vector4d& cam_from_rig_qvec,
                    const Eigen::Vector3d& cam_from_rig_tvec) {
    AddSensor(sensor_t(SensorType::CAMERA, camera_id), cam_from_rig_qvec,
              cam_from_rig_tvec);
}

bool Rig::HasCamera(const camera_t camera_id) const {
    return HasSensor(sensor_t(SensorType::CAMERA, camera_id));
}

size_t Rig::NumCameras() const {
    size_t num_cameras = 0;
    for (const auto& sensor : sensors_) num_cameras += sensor.first.type == SensorType::CAMERA;
    return num_cameras;
}

camera_t Rig::RefCameraId() const {
    return ref_sensor_id_.type == SensorType::CAMERA ? ref_sensor_id_.id : kInvalidCameraId;
}

std::vector<camera_t> Rig::CameraIds() const {
    std::vector<camera_t> camera_ids;
    camera_ids.reserve(NumCameras());
    for (const auto& sensor : sensors_) {
        if (sensor.first.type == SensorType::CAMERA) camera_ids.push_back(sensor.first.id);
    }
    return camera_ids;
}

const Eigen::Vector4d& Rig::CamFromRigQvec(const camera_t camera_id) const {
    const sensor_t sensor_id(SensorType::CAMERA, camera_id);
    if (sensor_id == ref_sensor_id_) {
        static const Eigen::Vector4d kIdentity = ComposeIdentityQuaternion();
        return kIdentity;
    }
    return SensorFromRigQvec(sensor_id);
}

Eigen::Vector4d& Rig::CamFromRigQvec(const camera_t camera_id) {
    CHECK(!(sensor_t(SensorType::CAMERA, camera_id) == ref_sensor_id_));
    return sensors_.at(sensor_t(SensorType::CAMERA, camera_id)).value().qvec;
}

const Eigen::Vector3d& Rig::CamFromRigTvec(const camera_t camera_id) const {
    const sensor_t sensor_id(SensorType::CAMERA, camera_id);
    if (sensor_id == ref_sensor_id_) {
        static const Eigen::Vector3d kZero = Eigen::Vector3d::Zero();
        return kZero;
    }
    return SensorFromRigTvec(sensor_id);
}

Eigen::Vector3d& Rig::CamFromRigTvec(const camera_t camera_id) {
    CHECK(!(sensor_t(SensorType::CAMERA, camera_id) == ref_sensor_id_));
    return sensors_.at(sensor_t(SensorType::CAMERA, camera_id)).value().tvec;
}

void Rig::WriteText(std::ostream* stream) const {
    CHECK_NOTNULL(stream);
    CHECK(stream->good());
    CHECK_NE(rig_id_, kInvalidRigId);
    CHECK_NE(static_cast<int>(ref_sensor_id_.type), static_cast<int>(SensorType::INVALID));
    stream->precision(17);
    *stream << "RIG 2 " << rig_id_ << " "
            << static_cast<int>(ref_sensor_id_.type) << " " << ref_sensor_id_.id
            << " " << NumSensors() << "\n";
    for (const sensor_t& sensor_id : SensorIds()) {
        const bool has_pose = HasSensorFromRig(sensor_id);
        *stream << "SENSOR " << static_cast<int>(sensor_id.type) << " "
                << sensor_id.id << " " << (has_pose ? 1 : 0);
        if (has_pose) {
            const auto& qvec = SensorFromRigQvec(sensor_id);
            const auto& tvec = SensorFromRigTvec(sensor_id);
            *stream << " " << qvec(0) << " " << qvec(1) << " " << qvec(2)
                    << " " << qvec(3) << " " << tvec(0) << " " << tvec(1)
                    << " " << tvec(2);
        }
        *stream << "\n";
    }
}

bool Rig::ReadText(std::istream* stream) {
    if (stream == nullptr || !stream->good()) {
        return false;
    }
    std::string tag;
    uint32_t version = 0;
    uint64_t num_sensors = 0;
    Rig parsed;
    camera_t ref_camera_id = kInvalidCameraId;
    if (!(*stream >> tag >> version >> parsed.rig_id_) || tag != "RIG" ||
        parsed.rig_id_ == kInvalidRigId) {
        return false;
    }
    if (version == 2) {
        int ref_type = static_cast<int>(SensorType::INVALID);
        uint32_t ref_id = std::numeric_limits<uint32_t>::max();
        if (!(*stream >> ref_type >> ref_id >> num_sensors) || num_sensors == 0 ||
            num_sensors > kMaxSerializedRigSensors) return false;
        parsed.AddRefSensor(sensor_t(static_cast<SensorType>(ref_type), ref_id));
        for (uint64_t i = 0; i < num_sensors; ++i) {
            int type = static_cast<int>(SensorType::INVALID);
            uint32_t id = std::numeric_limits<uint32_t>::max();
            int has_pose = 0;
            if (!(*stream >> tag >> type >> id >> has_pose) || tag != "SENSOR" ||
                (has_pose != 0 && has_pose != 1)) return false;
            const sensor_t sensor_id(static_cast<SensorType>(type), id);
            if (sensor_id == parsed.RefSensorId()) {
                if (has_pose != 0) return false;
                continue;
            }
            if (parsed.HasSensor(sensor_id)) return false;
            if (has_pose == 0) { parsed.AddSensor(sensor_id, std::nullopt, std::nullopt); continue; }
            SensorPose pose;
            if (!(*stream >> pose.qvec(0) >> pose.qvec(1) >> pose.qvec(2) >> pose.qvec(3) >>
                  pose.tvec(0) >> pose.tvec(1) >> pose.tvec(2)) || !IsValidPose(pose.qvec, pose.tvec)) return false;
            parsed.AddSensor(sensor_id, pose.qvec, pose.tvec);
        }
        if (parsed.NumSensors() != num_sensors) return false;
        *this = std::move(parsed);
        return true;
    }
    if (version != 1 || !(*stream >> ref_camera_id >> num_sensors) ||
        ref_camera_id == kInvalidCameraId || num_sensors == 0 ||
        num_sensors > kMaxSerializedRigSensors) return false;
    parsed.AddRefCamera(ref_camera_id);
    for (uint64_t i = 0; i < num_sensors; ++i) {
        camera_t camera_id = kInvalidCameraId;
        SensorPose pose;
        if (!(*stream >> tag >> camera_id >> pose.qvec(0) >> pose.qvec(1) >>
              pose.qvec(2) >> pose.qvec(3) >> pose.tvec(0) >> pose.tvec(1) >>
              pose.tvec(2)) ||
            tag != "CAMERA" || (camera_id != ref_camera_id && parsed.HasCamera(camera_id)) ||
            !IsValidPose(pose.qvec, pose.tvec)) {
            return false;
        }
        if (camera_id != ref_camera_id) parsed.AddCamera(camera_id, pose.qvec, pose.tvec);
    }
    if (parsed.NumCameras() != num_sensors) {
        return false;
    }
    *this = std::move(parsed);
    return true;
}

void Rig::WriteBinary(std::ostream* stream) const {
    CHECK_NOTNULL(stream);
    CHECK(stream->good());
    CHECK_NE(rig_id_, kInvalidRigId);
    CHECK_NE(static_cast<int>(ref_sensor_id_.type), static_cast<int>(SensorType::INVALID));
    WriteBinaryLittleEndian<uint32_t>(stream, kRigBinaryMagic);
    WriteBinaryLittleEndian<uint32_t>(stream, kRigFormatVersion);
    WriteBinaryLittleEndian<rig_t>(stream, rig_id_);
    WriteBinaryLittleEndian<int32_t>(stream, static_cast<int32_t>(ref_sensor_id_.type));
    WriteBinaryLittleEndian<uint32_t>(stream, ref_sensor_id_.id);
    WriteBinaryLittleEndian<uint64_t>(stream, NumSensors());
    for (const sensor_t& sensor_id : SensorIds()) {
        WriteBinaryLittleEndian<int32_t>(stream, static_cast<int32_t>(sensor_id.type));
        WriteBinaryLittleEndian<uint32_t>(stream, sensor_id.id);
        const uint8_t has_pose = HasSensorFromRig(sensor_id) ? 1 : 0;
        WriteBinaryLittleEndian<uint8_t>(stream, has_pose);
        if (has_pose == 0) continue;
        for (const double value : SensorFromRigQvec(sensor_id)) {
            WriteBinaryLittleEndian<double>(stream, value);
        }
        for (const double value : SensorFromRigTvec(sensor_id)) {
            WriteBinaryLittleEndian<double>(stream, value);
        }
    }
}

bool Rig::ReadBinary(std::istream* stream) {
    if (stream == nullptr || !stream->good()) {
        return false;
    }
    const uint32_t magic = ReadBinaryLittleEndian<uint32_t>(stream);
    const uint32_t version = ReadBinaryLittleEndian<uint32_t>(stream);
    Rig parsed;
    parsed.rig_id_ = ReadBinaryLittleEndian<rig_t>(stream);
    if (!stream->good() || magic != kRigBinaryMagic || parsed.rig_id_ == kInvalidRigId) {
        return false;
    }
    if (version == 2) {
        const int32_t ref_sensor_type = ReadBinaryLittleEndian<int32_t>(stream);
        const uint32_t ref_sensor_index = ReadBinaryLittleEndian<uint32_t>(stream);
        const sensor_t ref_sensor_id(static_cast<SensorType>(ref_sensor_type),
                                     ref_sensor_index);
        const uint64_t num_sensors = ReadBinaryLittleEndian<uint64_t>(stream);
        if (!stream->good() || ref_sensor_id.type == SensorType::INVALID ||
            num_sensors == 0 || num_sensors > kMaxSerializedRigSensors) return false;
        parsed.AddRefSensor(ref_sensor_id);
        for (uint64_t i = 0; i < num_sensors; ++i) {
            const int32_t sensor_type = ReadBinaryLittleEndian<int32_t>(stream);
            const uint32_t sensor_index = ReadBinaryLittleEndian<uint32_t>(stream);
            const sensor_t sensor_id(static_cast<SensorType>(sensor_type),
                                     sensor_index);
            const uint8_t has_pose = ReadBinaryLittleEndian<uint8_t>(stream);
            if (!stream->good() || (has_pose != 0 && has_pose != 1)) return false;
            if (sensor_id == ref_sensor_id) { if (has_pose != 0) return false; continue; }
            if (parsed.HasSensor(sensor_id)) return false;
            if (has_pose == 0) { parsed.AddSensor(sensor_id, std::nullopt, std::nullopt); continue; }
            SensorPose pose;
            for (double& value : pose.qvec) value = ReadBinaryLittleEndian<double>(stream);
            for (double& value : pose.tvec) value = ReadBinaryLittleEndian<double>(stream);
            if (!stream->good() || !IsValidPose(pose.qvec, pose.tvec)) return false;
            parsed.AddSensor(sensor_id, pose.qvec, pose.tvec);
        }
        if (parsed.NumSensors() != num_sensors) return false;
        *this = std::move(parsed);
        return true;
    }
    const camera_t ref_camera_id = ReadBinaryLittleEndian<camera_t>(stream);
    const uint64_t num_cameras = ReadBinaryLittleEndian<uint64_t>(stream);
    if (!stream->good() || version != 1 || ref_camera_id == kInvalidCameraId ||
        num_cameras == 0 || num_cameras > kMaxSerializedRigSensors) return false;
    parsed.AddRefCamera(ref_camera_id);
    for (uint64_t i = 0; i < num_cameras; ++i) {
        const camera_t camera_id = ReadBinaryLittleEndian<camera_t>(stream);
        SensorPose pose;
        for (double& value : pose.qvec) {
            value = ReadBinaryLittleEndian<double>(stream);
        }
        for (double& value : pose.tvec) {
            value = ReadBinaryLittleEndian<double>(stream);
        }
        if (!stream->good() || (camera_id != ref_camera_id && parsed.HasCamera(camera_id)) ||
            !IsValidPose(pose.qvec, pose.tvec)) {
            return false;
        }
        if (camera_id != ref_camera_id) parsed.AddCamera(camera_id, pose.qvec, pose.tvec);
    }
    if (parsed.NumCameras() != num_cameras) {
        return false;
    }
    *this = std::move(parsed);
    return true;
}

std::map<sensor_t, std::optional<Rigid3d>> Rig::NonRefSensors() const {
    std::map<sensor_t, std::optional<Rigid3d>> non_ref_sensors;
    for (const auto& [sensor_id, pose] : sensors_) {
        if (sensor_id == ref_sensor_id_) {
            continue;
        }
        if (pose.has_value()) {
            const Eigen::Quaterniond rotation(
                pose->qvec(0), pose->qvec(1), pose->qvec(2), pose->qvec(3));
            non_ref_sensors.emplace(sensor_id,
                                    Rigid3d(rotation, pose->tvec));
        } else {
            non_ref_sensors.emplace(sensor_id, std::nullopt);
        }
    }
    return non_ref_sensors;
}


namespace {

// Update the database with extracted rig and calibrations from the given
// reconstruction derived as follows:
//   * Compute the sensor_from_rig poses as the average of the relative
//     poses between registered sensors in the reconstruction.
//   * Set the camera calibration parameters from the first frame with an image
//     of the camera.
void UpdateRigAndCameraCalibsFromReconstruction(
    const Reconstruction& reconstruction,
    const std::map<std::string, std::vector<const Image*>>&
        frame_name_to_images,
    Rig& rig,
    Database& database) {
  NodeHashMap<camera_t,
              std::pair<std::vector<Eigen::Quaterniond>, Eigen::Vector3d>>
      rig_from_cams;
  std::set<camera_t> updated_cameras;
  for (auto& [_, images] : frame_name_to_images) {
    const Image* ref_image = nullptr;
    for (const Image* image : images) {
      if (rig.IsRefSensor(image->DataId().sensor_id)) {
        ref_image = image;
      }
    }

    if (ref_image == nullptr) {
      continue;
    }

    const Image* rig_calib_ref_image =
        reconstruction.FindImageWithName(ref_image->Name());
    if (rig_calib_ref_image == nullptr || !rig_calib_ref_image->HasPose()) {
      continue;
    }

    const Rigid3d ref_cam_from_world = rig_calib_ref_image->CamFromWorld();
    if (updated_cameras.insert(rig_calib_ref_image->CameraId()).second) {
      Camera ref_camera = *rig_calib_ref_image->CameraPtr();
      ref_camera.SetCameraId(ref_image->CameraId());
      database.UpdateCamera(ref_camera);
    }

    for (const Image* image : images) {
      if (image->CameraId() != ref_image->CameraId()) {
        const Image* rig_calib_image =
            reconstruction.FindImageWithName(image->Name());
        if (rig_calib_image == nullptr || !rig_calib_image->HasPose()) {
          continue;
        }
        const Rigid3d rig_from_cam =
            ref_cam_from_world * Inverse(rig_calib_image->CamFromWorld());
        auto& [rig_from_cam_rotations, rig_from_cam_translation] =
            rig_from_cams[image->CameraId()];
        if (updated_cameras.insert(rig_calib_image->CameraId()).second) {
          Camera camera = *rig_calib_image->CameraPtr();
          camera.SetCameraId(image->CameraId());
          database.UpdateCamera(camera);
        }
        if (rig_from_cam_rotations.empty()) {
          rig_from_cam_translation = rig_from_cam.translation();
        } else {
          rig_from_cam_translation += rig_from_cam.translation();
        }
        rig_from_cam_rotations.push_back(rig_from_cam.rotation());
      }
    }
  }

  // Compute the average sensor_from_rig poses over all frames.
  for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
    if (sensor_from_rig.has_value()) {
      // Do not compute it for explicitly provided poses in the config.
      continue;
    }

    const auto it = rig_from_cams.find(sensor_id.id);
    if (it == rig_from_cams.end()) {
      LOG(WARNING)
          << "Failed to derive sensor_from_rig transformation for camera "
          << sensor_id.id
          << ", because the image was not registered in the given "
             "reconstruction.";
      continue;
    }

    const auto& [rig_from_cam_rotations, rig_from_cam_translation] = it->second;
    const Rigid3d rig_from_cam(
        AverageQuaternions(
            rig_from_cam_rotations,
            std::vector<double>(rig_from_cam_rotations.size(), 1.0)),
        rig_from_cam_translation / rig_from_cam_rotations.size());
    rig.SetSensorFromRig(sensor_id, Inverse(rig_from_cam));
  }

  database.UpdateRig(rig);
}

void UpdateRigsAndFramesFromDatabase(const Database& database,
                                     Reconstruction* reconstruction) {
  const std::vector<Frame> database_frames = database.ReadAllFrames();

  NodeHashMap<rig_t, Rig> database_rigs;
  database_rigs.reserve(database.NumRigs());
  for (auto& rig : database.ReadAllRigs()) {
    database_rigs.emplace(rig.RigId(), std::move(rig));
  }

  NodeHashMap<rig_t, Rig> reconstruction_rigs;
  reconstruction_rigs.reserve(database_rigs.size());

  // Create O(1) lookup table from image names to images.
  NodeHashMap<std::string, const Image*> image_name_to_image;
  image_name_to_image.reserve(reconstruction->NumImages());
  for (const auto& [_, image] : reconstruction->Images()) {
    image_name_to_image.emplace(image.Name(), &image);
  }

  auto visit_frame_data =
      [&database, &image_name_to_image, &database_frames](
          const std::function<void(const Frame&, const Image&, const Image&)>&
              visitor) {
        for (const Frame& database_frame : database_frames) {
          for (const image_t image_id : database_frame.ImageIds()) {
            const Image database_image = database.ReadImage(image_id);
            const auto reconstruction_image =
                image_name_to_image.find(database_image.Name());
            if (reconstruction_image == image_name_to_image.end()) {
              continue;
            }
            visitor(
                database_frame, database_image, *reconstruction_image->second);
          }
        }
      };

  // Update reference sensors in reconstruction rigs.
  // (must be done before updating the non-reference sensors).
  visit_frame_data([&](const Frame& database_frame,
                       const Image& database_image,
                       const Image& reconstruction_image) {
    const Rig& database_rig = database_rigs.at(database_frame.RigId());
    Rig& reconstruction_rig = reconstruction_rigs[database_frame.RigId()];
    reconstruction_rig.SetRigId(database_frame.RigId());

    const sensor_t& database_sensor_id = database_image.DataId().sensor_id;
    const sensor_t& reconstruction_sensor_id =
        reconstruction_image.CameraPtr()->SensorId();

    if (!reconstruction_rig.IsRefSensor(reconstruction_sensor_id) &&
        database_rig.IsRefSensor(database_sensor_id)) {
      reconstruction_rig.AddRefSensor(reconstruction_sensor_id);
    }
  });

  // Update non-reference sensors in reconstruction rigs.
  visit_frame_data([&](const Frame& database_frame,
                       const Image& database_image,
                       const Image& reconstruction_image) {
    const Rig& database_rig = database_rigs.at(database_frame.RigId());
    Rig& reconstruction_rig = reconstruction_rigs[database_frame.RigId()];
    reconstruction_rig.SetRigId(database_frame.RigId());

    const sensor_t& database_sensor_id = database_image.DataId().sensor_id;
    const sensor_t& reconstruction_sensor_id =
        reconstruction_image.CameraPtr()->SensorId();

    // Upstream parity: add the non-reference sensor only when the
    // reconstruction rig does not have it yet and the database rig does.
    if (!reconstruction_rig.HasSensor(reconstruction_sensor_id) &&
        !database_rig.IsRefSensor(database_sensor_id) &&
        database_rig.HasSensor(database_sensor_id)) {
      reconstruction_rig.AddSensor(
          reconstruction_sensor_id,
          database_rig.SensorFromRig(database_sensor_id));
    }
  });

  // Update reconstruction frames.
  NodeHashMap<frame_t, Frame> reconstruction_frames;
  reconstruction_frames.reserve(database_frames.size());
  visit_frame_data([&](const Frame& database_frame,
                       const Image& database_image,
                       const Image& reconstruction_image) {
    const Rig& database_rig = database_rigs.at(database_frame.RigId());
    const sensor_t& database_sensor_id = database_image.DataId().sensor_id;
    Frame& reconstruction_frame =
        reconstruction_frames[database_frame.FrameId()];
    reconstruction_frame.SetFrameId(database_frame.FrameId());
    reconstruction_frame.SetRigId(database_frame.RigId());
    reconstruction_frame.AddDataId(reconstruction_image.DataId());
    if (reconstruction_image.HasPose()) {
      if (database_rig.IsRefSensor(database_sensor_id)) {
        reconstruction_frame.SetRigFromWorld(
            reconstruction_image.CamFromWorld());
      } else {
        reconstruction_frame.SetRigFromWorld(
            Inverse(database_rig.SensorFromRig(database_sensor_id)) *
            reconstruction_image.CamFromWorld());
      }
    }
  });

  std::vector<Rig> rigs;
  rigs.reserve(reconstruction_rigs.size());
  for (auto& [_, rig] : reconstruction_rigs) {
    rigs.push_back(std::move(rig));
  }

  std::vector<Frame> frames;
  frames.reserve(reconstruction_frames.size());
  for (auto& [_, frame] : reconstruction_frames) {
    frames.push_back(std::move(frame));
  }

  reconstruction->SetRigsAndFrames(std::move(rigs), std::move(frames));
}

void CopyCameraIntrinsics(const Camera& src, Camera& dst) {
  dst.SetModelId(src.ModelId());
  dst.SetParams(src.Params());
  dst.SetPriorFocalLength(src.HasPriorFocalLength());
}

}  // namespace

std::vector<RigConfig> ReadRigConfig(
    const std::filesystem::path& rig_config_path) {
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(rig_config_path.string().c_str(), pt);

  std::vector<RigConfig> configs;
  for (const auto& rig_node : pt) {
    RigConfig& config = configs.emplace_back();
    bool has_ref_sensor = false;
    for (const auto& camera : rig_node.second.get_child("cameras")) {
      RigConfig::RigCamera& config_camera = config.cameras.emplace_back();

      config_camera.image_prefix =
          camera.second.get<std::string>("image_prefix");

      auto cam_from_rig_rotation_node =
          camera.second.get_child_optional("cam_from_rig_rotation");
      auto cam_from_rig_translation_node =
          camera.second.get_child_optional("cam_from_rig_translation");
      if (cam_from_rig_rotation_node && cam_from_rig_translation_node) {
        Rigid3d cam_from_rig;

        int index = 0;
        Eigen::Vector4d cam_from_rig_wxyz;
        for (const auto& node : cam_from_rig_rotation_node.get()) {
          cam_from_rig_wxyz[index++] = node.second.get_value<double>();
        }
        cam_from_rig.rotation() = Eigen::Quaterniond(cam_from_rig_wxyz(0),
                                                     cam_from_rig_wxyz(1),
                                                     cam_from_rig_wxyz(2),
                                                     cam_from_rig_wxyz(3));

        THROW_CHECK(cam_from_rig_translation_node);
        index = 0;
        for (const auto& node : cam_from_rig_translation_node.get()) {
          cam_from_rig.translation()(index++) = node.second.get_value<double>();
        }
        config_camera.cam_from_rig = cam_from_rig;
      }

      auto ref_sensor_node = camera.second.get_child_optional("ref_sensor");
      if (ref_sensor_node && ref_sensor_node.get().get_value<bool>()) {
        THROW_CHECK(!cam_from_rig_rotation_node &&
                    !cam_from_rig_translation_node)
            << "Reference sensor must not have cam_from_rig";
        THROW_CHECK(!has_ref_sensor)
            << "Rig must only have one reference sensor";
        config_camera.ref_sensor = true;
        has_ref_sensor = true;
      }

      auto camera_model_name_node =
          camera.second.get_child_optional("camera_model_name");
      auto camera_params_node =
          camera.second.get_child_optional("camera_params");
      if (camera_model_name_node && camera_params_node) {
        config_camera.camera = std::make_optional<Camera>();
        config_camera.camera->SetModelId(CameraModelNameToId(
            camera.second.get<std::string>("camera_model_name")));
        config_camera.camera->SetPriorFocalLength(true);
        // Fork adaptation: Camera::SetModelId pre-fills params with zeros
        // (upstream keeps them empty), so reset before appending.
        config_camera.camera->SetParams(std::vector<double>());
        for (const auto& node : camera_params_node.get()) {
          config_camera.camera->Params().push_back(
              node.second.get_value<double>());
        }
      }
    }

    THROW_CHECK(has_ref_sensor) << "Rig must have one reference sensor";
  }

  return configs;
}

void ApplyRigConfig(const std::vector<RigConfig>& configs,
                    Database& database,
                    Reconstruction* reconstruction) {
  database.ClearFrames();
  database.ClearRigs();

  const std::vector<Image> images = database.ReadAllImages();
  std::set<image_t> configured_image_ids;

  for (const RigConfig& config : configs) {
    Rig rig;

    const size_t num_cameras = config.cameras.size();

    std::vector<std::optional<camera_t>> camera_ids(num_cameras);
    std::map<std::string, std::vector<const Image*>> frame_name_to_images;
    for (const Image& image : images) {
      for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
        const auto& config_camera = config.cameras[camera_idx];
        if (StringStartsWith(image.Name(), config_camera.image_prefix)) {
          const std::string frame_name =
              StringGetAfter(image.Name(), config_camera.image_prefix);
          frame_name_to_images[frame_name].push_back(&image);
          std::optional<camera_t>& camera_id = camera_ids[camera_idx];
          if (camera_id.has_value()) {
            THROW_CHECK_EQ(*camera_id, image.CameraId())
                << "Inconsistent cameras for images with prefix: "
                << config_camera.image_prefix
                << ". Consider setting --ImageReader.single_camera_per_folder "
                   "during feature extraction or manually assign consistent "
                   "camera_id's.";
          } else {
            camera_id = image.CameraId();
            if (config_camera.camera.has_value()) {
              Camera database_camera = database.ReadCamera(image.CameraId());
              CopyCameraIntrinsics(*config_camera.camera, database_camera);
              database.UpdateCamera(database_camera);
              if (reconstruction != nullptr) {
                auto& reconstruction_camera =
                    reconstruction->Camera(image.CameraId());
                CopyCameraIntrinsics(*config_camera.camera,
                                     reconstruction_camera);
              }
            }
          }
        }
      }
    }

    std::set<camera_t> unique_camera_ids;
    for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
      const auto& config_camera = config.cameras[camera_idx];
      std::optional<camera_t>& camera_id = camera_ids[camera_idx];
      THROW_CHECK(camera_id.has_value())
          << "At least one image must exist for each camera in the rig";
      if (!unique_camera_ids.insert(*camera_id).second) {
        // Clone the camera, if multiple cameras in the rig share a camera.
        *camera_id = database.WriteCamera(database.ReadCamera(*camera_id));
      }
      if (config_camera.ref_sensor) {
        rig.AddRefSensor(sensor_t(SensorType::CAMERA, *camera_id));
      } else {
        rig.AddSensor(sensor_t(SensorType::CAMERA, *camera_id),
                      config_camera.cam_from_rig);
      }
    }

    rig.SetRigId(database.WriteRig(rig));
    LOG(INFO) << "Configured rig_id=" << rig.RigId();

    for (auto& [frame_name, frame_images] : frame_name_to_images) {
      Frame frame;
      frame.SetRigId(rig.RigId());
      for (const Image* image : frame_images) {
        const data_t& data_id = image->DataId();
        THROW_CHECK(rig.HasSensor(data_id.sensor_id))
            << "rig_id=" << rig.RigId() << " must not contain Image(image_id="
            << image->ImageId() << ", camera_id=" << image->CameraId()
            << ", name=" << image->Name() << ")";
        frame.AddDataId(data_id);
        configured_image_ids.insert(image->ImageId());
      }
      frame.SetFrameId(database.WriteFrame(frame));
      LOG(INFO) << "Configured frame_id=" << frame.FrameId();
    }

    if (reconstruction != nullptr) {
      UpdateRigAndCameraCalibsFromReconstruction(
          *reconstruction, frame_name_to_images, rig, database);
    }
  }

  // Create trivial rigs/frames for images without configuration.
  // This is necessary because we clear rigs/frames above.
  NodeHashMap<camera_t, rig_t> camera_to_rig_id;
  for (const Image& image : images) {
    if (configured_image_ids.count(image.ImageId()) > 0) {
      continue;
    }
    const sensor_t sensor_id(SensorType::CAMERA, image.CameraId());
    auto rig_id_it = camera_to_rig_id.find(image.CameraId());
    if (rig_id_it == camera_to_rig_id.end()) {
      Rig rig;
      rig.AddRefSensor(sensor_id);
      rig_id_it =
          camera_to_rig_id.emplace(image.CameraId(), database.WriteRig(rig))
              .first;
    }
    Frame frame;
    frame.SetRigId(rig_id_it->second);
    frame.AddDataId(data_t(sensor_id, image.ImageId()));
    frame.SetFrameId(database.WriteFrame(frame));
  }

  if (reconstruction != nullptr) {
    UpdateRigsAndFramesFromDatabase(database, reconstruction);
  }
}

}  // namespace colmap
