// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "util/alignment.h"

#ifdef _MSC_VER
#if _MSC_VER >= 1600
#include <cstdint>
#else
typedef __int8 int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#endif
#elif __GNUC__ >= 3
#include <cstdint>
#endif

// Define non-copyable or non-movable classes.
#define NON_COPYABLE(class_name)            \
    class_name(class_name const&) = delete; \
    void operator=(class_name const& obj) = delete;
#define NON_MOVABLE(class_name) class_name(class_name&&) = delete;

#include <Eigen/Core>

#include <tuple>

namespace Eigen {
using Matrix3x4f = Matrix<float, 3, 4>;
using Matrix3x4d = Matrix<double, 3, 4>;
using Matrix2x3d = Matrix<double, 2, 3>;
using Matrix6d = Matrix<double, 6, 6>;
using Vector3ub = Matrix<uint8_t, 3, 1>;
using Vector4ub = Matrix<uint8_t, 4, 1>;
using Vector6d = Matrix<double, 6, 1>;
using RowMajorMatrixXf = Matrix<float, Dynamic, Dynamic, RowMajor>;
using RowMajorMatrixXd = Matrix<double, Dynamic, Dynamic, RowMajor>;
using RowMajorMatrixXi = Matrix<int, Dynamic, Dynamic, RowMajor>;

}  // namespace Eigen

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// Index types, determines the maximum number of objects.
////////////////////////////////////////////////////////////////////////////////

// Unique identifier for cameras.
typedef uint32_t camera_t;

// Unique identifier for images.
typedef uint32_t image_t;

// Unique identifier for a rigid camera calibration and one of its captures.
typedef uint32_t rig_t;
typedef uint32_t frame_t;

// A rig is a calibration container for all sensors, not just cameras. Keep
// these identifiers separate from camera/image ids so database records can
// represent upstream COLMAP's generic sensor/data associations.
enum class SensorType : int32_t {
    INVALID = -1,
    CAMERA = 0,
    IMU = 1,
};

struct sensor_t {
    SensorType type = SensorType::INVALID;
    uint32_t id = std::numeric_limits<uint32_t>::max();

    constexpr sensor_t() = default;
    constexpr sensor_t(const SensorType sensor_type, const uint32_t sensor_id)
        : type(sensor_type), id(sensor_id) {}

    bool operator==(const sensor_t& other) const {
        return type == other.type && id == other.id;
    }
    bool operator!=(const sensor_t& other) const { return !(*this == other); }
    bool operator<(const sensor_t& other) const {
        return std::tie(type, id) < std::tie(other.type, other.id);
    }
};

struct data_t {
    sensor_t sensor_id;
    uint64_t id = std::numeric_limits<uint64_t>::max();

    constexpr data_t() = default;
    constexpr data_t(const sensor_t& sensor, const uint64_t data_id)
        : sensor_id(sensor), id(data_id) {}

    bool operator==(const data_t& other) const {
        return sensor_id == other.sensor_id && id == other.id;
    }
    bool operator!=(const data_t& other) const { return !(*this == other); }
    bool operator<(const data_t& other) const {
        return std::tie(sensor_id, id) < std::tie(other.sensor_id, other.id);
    }
};

// Each image pair gets a unique ID, see `Database::ImagePairToPairId`.
typedef uint64_t image_pair_t;

// Index per image, i.e. determines maximum number of 2D points per image.
typedef uint32_t point2D_t;

// Unique identifier per added 3D point. Since we add many 3D points,
// delete them, and possibly re-add them again, the maximum number of allowed
// unique indices should be large.
typedef uint64_t point3D_t;

// Values for invalid identifiers or indices.
const camera_t kInvalidCameraId = std::numeric_limits<camera_t>::max();
const image_t kInvalidImageId = std::numeric_limits<image_t>::max();
const rig_t kInvalidRigId = std::numeric_limits<rig_t>::max();
const frame_t kInvalidFrameId = std::numeric_limits<frame_t>::max();
const image_pair_t kInvalidImagePairId =
        std::numeric_limits<image_pair_t>::max();
const point2D_t kInvalidPoint2DIdx = std::numeric_limits<point2D_t>::max();
const point3D_t kInvalidPoint3DId = std::numeric_limits<point3D_t>::max();

}  // namespace colmap

// This file provides specializations of the templated hash function for
// custom types. These are used for comparison in unordered sets/maps.
namespace std {

// Hash function specialization for uint32_t pairs, e.g., image_t or camera_t.
template <>
struct hash<std::pair<uint32_t, uint32_t>> {
    std::size_t operator()(const std::pair<uint32_t, uint32_t>& p) const {
        const uint64_t s = (static_cast<uint64_t>(p.first) << 32) +
                           static_cast<uint64_t>(p.second);
        return std::hash<uint64_t>()(s);
    }
};

}  // namespace std
