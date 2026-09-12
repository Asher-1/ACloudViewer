// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------

#include "base/rig.h"

#include <istream>
#include <limits>
#include <ostream>
#include <utility>

#include "util/endian.h"
#include "util/logging.h"

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

}  // namespace colmap
