// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------

#include "base/frame.h"

#include <istream>
#include <limits>
#include <ostream>
#include <utility>

#include "util/endian.h"
#include "util/logging.h"

namespace colmap {

namespace {

constexpr uint32_t kFrameBinaryMagic = 0x46525631;  // FRV1
constexpr uint32_t kFrameFormatVersion = 2;
constexpr uint64_t kMaxSerializedFrameImages = 1000000;

bool IsValidPose(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec) {
    return qvec.allFinite() && tvec.allFinite() &&
           qvec.squaredNorm() > std::numeric_limits<double>::epsilon();
}

}  // namespace

frame_t Frame::FrameId() const { return frame_id_; }
void Frame::SetFrameId(const frame_t frame_id) {
    CHECK_NE(frame_id, kInvalidFrameId);
    frame_id_ = frame_id;
}

rig_t Frame::RigId() const { return rig_id_; }
void Frame::SetRigId(const rig_t rig_id) {
    CHECK_NE(rig_id, kInvalidRigId);
    rig_id_ = rig_id;
}

void Frame::AddDataId(const data_t& data_id) {
    CHECK_NE(static_cast<int>(data_id.sensor_id.type),
             static_cast<int>(SensorType::INVALID));
    CHECK_NE(data_id.id, std::numeric_limits<uint64_t>::max());
    data_ids_.insert(data_id);
    if (data_id.sensor_id.type == SensorType::CAMERA &&
        data_id.id <= std::numeric_limits<image_t>::max()) {
        image_ids_.insert(static_cast<image_t>(data_id.id));
    }
}

bool Frame::HasDataId(const data_t& data_id) const {
    return data_ids_.count(data_id) != 0;
}

const std::set<data_t>& Frame::DataIds() const { return data_ids_; }

void Frame::AddImageId(const image_t image_id) {
    CHECK_NE(image_id, kInvalidImageId);
    image_ids_.insert(image_id);
    // Fork-legacy bridge: image ids imply a camera data entry under the
    // historical image_id == camera_id assumption. Database::ReadFrame only
    // falls back to this path for legacy rows that lack frame_data entries.
    data_ids_.insert(data_t(sensor_t(SensorType::CAMERA, image_id), image_id));
}

bool Frame::HasImageId(const image_t image_id) const {
    return image_ids_.count(image_id) != 0;
}

const std::set<image_t>& Frame::ImageIds() const { return image_ids_; }

bool Frame::HasPose() const { return has_pose_; }

void Frame::SetRigFromWorld(const Eigen::Vector4d& qvec,
                            const Eigen::Vector3d& tvec) {
    // Upstream parity (dbb41680): no finiteness checks here. The rotation
    // averaging stack legally seeds un-estimated frames with a NaN
    // "unknown pose" placeholder before overwriting it with the solution.
    // Fork note: normalize idempotently. The fork stores the pose as a
    // qvec/tvec pair while callers hand over Rigid3d values, so every
    // round-trip through RigFromWorld()/SetRigFromWorld() re-enters here;
    // re-normalizing an already-normalized quaternion perturbs the
    // components at the 1e-16 level and breaks bit-exact equality (e.g. the
    // reconstruction_io round-trip tests). Skip the divide when the input is
    // already unit-length within double rounding.
    if (std::abs(qvec.norm() - 1.0) > 1e-12) {
        rig_from_world_qvec_ = NormalizeQuaternion(qvec);
    } else {
        rig_from_world_qvec_ = qvec;
    }
    rig_from_world_tvec_ = tvec;
    has_pose_ = true;
}

void Frame::ResetPose() { has_pose_ = false; }

const Eigen::Vector4d& Frame::RigFromWorldQvec() const {
    CHECK(has_pose_);
    return rig_from_world_qvec_;
}

const Eigen::Vector3d& Frame::RigFromWorldTvec() const {
    CHECK(has_pose_);
    return rig_from_world_tvec_;
}

void Frame::WriteText(std::ostream* stream) const {
    CHECK_NOTNULL(stream);
    CHECK(stream->good());
    CHECK_NE(frame_id_, kInvalidFrameId);
    CHECK_NE(rig_id_, kInvalidRigId);
    stream->precision(17);
    *stream << "FRAME " << kFrameFormatVersion << " " << frame_id_ << " " << rig_id_ << " "
            << (has_pose_ ? 1 : 0) << " " << data_ids_.size() << "\n";
    if (has_pose_) {
        *stream << "POSE " << rig_from_world_qvec_(0) << " "
                << rig_from_world_qvec_(1) << " " << rig_from_world_qvec_(2)
                << " " << rig_from_world_qvec_(3) << " "
                << rig_from_world_tvec_(0) << " " << rig_from_world_tvec_(1)
                << " " << rig_from_world_tvec_(2) << "\n";
    }
    for (const data_t& data_id : data_ids_) {
        *stream << "DATA " << static_cast<int>(data_id.sensor_id.type) << " "
                << data_id.sensor_id.id << " " << data_id.id << "\n";
    }
}

bool Frame::ReadText(std::istream* stream) {
    if (stream == nullptr || !stream->good()) {
        return false;
    }
    std::string tag;
    uint32_t version = 0;
    int has_pose = 0;
    uint64_t num_data = 0;
    Frame parsed;
    if (!(*stream >> tag >> version >> parsed.frame_id_ >> parsed.rig_id_ >>
          has_pose >> num_data) ||
        tag != "FRAME" || (version != 1 && version != 2) || parsed.frame_id_ == kInvalidFrameId ||
        parsed.rig_id_ == kInvalidRigId || (has_pose != 0 && has_pose != 1) ||
        num_data == 0 || num_data > kMaxSerializedFrameImages) {
        return false;
    }
    if (has_pose != 0) {
        Eigen::Vector4d qvec;
        Eigen::Vector3d tvec;
        if (!(*stream >> tag >> qvec(0) >> qvec(1) >> qvec(2) >> qvec(3) >>
              tvec(0) >> tvec(1) >> tvec(2)) ||
            tag != "POSE" || !IsValidPose(qvec, tvec)) {
            return false;
        }
        parsed.SetRigFromWorld(qvec, tvec);
    }
    for (uint64_t i = 0; i < num_data; ++i) {
        if (version == 2) {
            int sensor_type = static_cast<int>(SensorType::INVALID);
            uint32_t sensor_id = std::numeric_limits<uint32_t>::max();
            uint64_t data_id = std::numeric_limits<uint64_t>::max();
            if (!(*stream >> tag >> sensor_type >> sensor_id >> data_id) || tag != "DATA") return false;
            const data_t data(sensor_t(static_cast<SensorType>(sensor_type), sensor_id), data_id);
            if (parsed.HasDataId(data)) return false;
            parsed.AddDataId(data);
            continue;
        }
        image_t image_id = kInvalidImageId;
        if (!(*stream >> tag >> image_id) || tag != "IMAGE" ||
            image_id == kInvalidImageId || parsed.HasImageId(image_id)) {
            return false;
        }
        parsed.AddImageId(image_id);
    }
    *this = std::move(parsed);
    return true;
}

void Frame::WriteBinary(std::ostream* stream) const {
    CHECK_NOTNULL(stream);
    CHECK(stream->good());
    CHECK_NE(frame_id_, kInvalidFrameId);
    CHECK_NE(rig_id_, kInvalidRigId);
    WriteBinaryLittleEndian<uint32_t>(stream, kFrameBinaryMagic);
    WriteBinaryLittleEndian<uint32_t>(stream, kFrameFormatVersion);
    WriteBinaryLittleEndian<frame_t>(stream, frame_id_);
    WriteBinaryLittleEndian<rig_t>(stream, rig_id_);
    WriteBinaryLittleEndian<uint8_t>(stream, has_pose_ ? 1 : 0);
    WriteBinaryLittleEndian<uint64_t>(stream, data_ids_.size());
    if (has_pose_) {
        for (const double value : rig_from_world_qvec_) {
            WriteBinaryLittleEndian<double>(stream, value);
        }
        for (const double value : rig_from_world_tvec_) {
            WriteBinaryLittleEndian<double>(stream, value);
        }
    }
    for (const data_t& data_id : data_ids_) {
        WriteBinaryLittleEndian<int32_t>(stream, static_cast<int32_t>(data_id.sensor_id.type));
        WriteBinaryLittleEndian<uint32_t>(stream, data_id.sensor_id.id);
        WriteBinaryLittleEndian<uint64_t>(stream, data_id.id);
    }
}

bool Frame::ReadBinary(std::istream* stream) {
    if (stream == nullptr || !stream->good()) {
        return false;
    }
    const uint32_t magic = ReadBinaryLittleEndian<uint32_t>(stream);
    const uint32_t version = ReadBinaryLittleEndian<uint32_t>(stream);
    Frame parsed;
    parsed.frame_id_ = ReadBinaryLittleEndian<frame_t>(stream);
    parsed.rig_id_ = ReadBinaryLittleEndian<rig_t>(stream);
    const uint8_t has_pose = ReadBinaryLittleEndian<uint8_t>(stream);
    const uint64_t num_data = ReadBinaryLittleEndian<uint64_t>(stream);
    if (!stream->good() || magic != kFrameBinaryMagic ||
        (version != 1 && version != kFrameFormatVersion) || parsed.frame_id_ == kInvalidFrameId ||
        parsed.rig_id_ == kInvalidRigId || (has_pose != 0 && has_pose != 1) ||
        num_data == 0 || num_data > kMaxSerializedFrameImages) {
        return false;
    }
    if (has_pose != 0) {
        Eigen::Vector4d qvec;
        Eigen::Vector3d tvec;
        for (double& value : qvec) {
            value = ReadBinaryLittleEndian<double>(stream);
        }
        for (double& value : tvec) {
            value = ReadBinaryLittleEndian<double>(stream);
        }
        if (!stream->good() || !IsValidPose(qvec, tvec)) {
            return false;
        }
        parsed.SetRigFromWorld(qvec, tvec);
    }
    for (uint64_t i = 0; i < num_data; ++i) {
        if (version == 2) {
            const int32_t sensor_type = ReadBinaryLittleEndian<int32_t>(stream);
            const uint32_t sensor_index = ReadBinaryLittleEndian<uint32_t>(stream);
            const uint64_t data_index = ReadBinaryLittleEndian<uint64_t>(stream);
            const data_t data(sensor_t(static_cast<SensorType>(sensor_type),
                                       sensor_index),
                              data_index);
            if (!stream->good() || parsed.HasDataId(data)) return false;
            parsed.AddDataId(data);
            continue;
        }
        const image_t image_id = ReadBinaryLittleEndian<image_t>(stream);
        if (!stream->good() || image_id == kInvalidImageId ||
            parsed.HasImageId(image_id)) {
            return false;
        }
        parsed.AddImageId(image_id);
    }
    *this = std::move(parsed);
    return true;
}

}  // namespace colmap
