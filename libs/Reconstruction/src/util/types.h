// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "util/alignment.h"
#include "util/enum_utils.h"

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
using Matrix3x2d = Matrix<double, 3, 2>;
using Matrix6d = Matrix<double, 6, 6>;
using Vector3ub = Matrix<uint8_t, 3, 1>;
using Vector4ub = Matrix<uint8_t, 4, 1>;
using Vector6d = Matrix<double, 6, 1>;
using Vector7d = Matrix<double, 7, 1>;
// Upstream COLMAP dbb41680 parity (Sim3d packed params, W4).
using Vector8d = Matrix<double, 8, 1>;
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
#ifdef __CUDACC__
enum class SensorType : int32_t {
    INVALID = -1,
    CAMERA = 0,
    IMU = 1,
};
#else
MAKE_ENUM_CLASS_OVERLOAD_STREAM(SensorType, -1, INVALID, CAMERA, IMU);
#endif

struct sensor_t {
    constexpr static uint32_t kInvalidId = std::numeric_limits<uint32_t>::max();

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
    constexpr static uint32_t kInvalidId = std::numeric_limits<uint32_t>::max();

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

constexpr sensor_t kInvalidSensorId =
        sensor_t(SensorType::INVALID, sensor_t::kInvalidId);

// The maximum number of images that can be stored in the database, as we
// generate unique image_pair_ids based on the image ids (upstream parity,
// dbb41680 util/types.h).
constexpr size_t kMaxNumImages =
        static_cast<size_t>(std::numeric_limits<int32_t>::max());

// Each image pair gets a unique ID, see `ImagePairToPairId`.
typedef uint64_t image_pair_t;

// Return true if image pairs should be swapped. Used to enforce a specific
// image order to generate unique image pair identifiers independent of the
// order in which the image identifiers are used. Upstream COLMAP dbb41680
// util/types.h parity.
inline bool ShouldSwapImagePair(image_t image_id1, image_t image_id2) {
    return image_id1 > image_id2;
}

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

inline void ThrowIfGtMaxImages(image_t image_id) {
    if (image_id >= kMaxNumImages) {
        throw std::runtime_error("image_id=" + std::to_string(image_id) +
                                 " >= kMaxNumImages.");
    }
}

// Convert pair of image identifiers to unique image pair identifier.
inline image_pair_t ImagePairToPairId(image_t image_id1, image_t image_id2) {
    ThrowIfGtMaxImages(image_id1);
    ThrowIfGtMaxImages(image_id2);
    if (ShouldSwapImagePair(image_id1, image_id2)) {
        return static_cast<image_pair_t>(kMaxNumImages) * image_id2 + image_id1;
    } else {
        return static_cast<image_pair_t>(kMaxNumImages) * image_id1 + image_id2;
    }
}

// Convert unique image pair identifier to pair of image identifiers.
inline std::pair<image_t, image_t> PairIdToImagePair(image_pair_t pair_id) {
    const image_t image_id2 = static_cast<image_t>(pair_id % kMaxNumImages);
    const image_t image_id1 =
            static_cast<image_t>((pair_id - image_id2) / kMaxNumImages);
    ThrowIfGtMaxImages(image_id1);
    ThrowIfGtMaxImages(image_id2);
    return std::make_pair(image_id1, image_id2);
}

const point2D_t kInvalidPoint2DIdx = std::numeric_limits<point2D_t>::max();
const point3D_t kInvalidPoint3DId = std::numeric_limits<point3D_t>::max();

using pose_prior_t = uint32_t;
constexpr pose_prior_t kInvalidPosePriorId =
        std::numeric_limits<pose_prior_t>::max();
constexpr data_t kInvalidDataId = data_t(kInvalidSensorId, data_t::kInvalidId);

// Hash functor for (image_t, image_t) pairs and generic uint64 pairs
// (upstream parity, COLMAP 4.x util/types.h).
// Upstream-parity alias: camera model identifiers are plain ints in this
// fork (see base/camera_models.h CAMERA_MODEL_DEFINITIONS ids).
using CameraModelId = int;

struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const {
        const auto h1 = std::hash<T1>()(pair.first);
        const auto h2 = std::hash<T2>()(pair.second);
        return h1 ^ (h2 << (sizeof(std::size_t) * 4));
    }
};

// Simple implementation of C++20's std::ranges::filter_view.
// Upstream COLMAP dbb41680 util/types.h parity (W4 GLomap consumers).

template <class Iterator, class Predicate>
struct filter_iterator {
    template <class OtherIterator, class OtherPredicate>
    friend struct filter_iterator;

    using base_category =
            typename std::iterator_traits<Iterator>::iterator_category;
    using iterator_category = typename std::conditional<
            std::is_same<base_category, std::random_access_iterator_tag>::value,
            std::bidirectional_iterator_tag,
            base_category>::type;

    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using reference = typename std::iterator_traits<Iterator>::reference;
    using pointer = typename std::iterator_traits<Iterator>::pointer;
    using difference_type =
            typename std::iterator_traits<Iterator>::difference_type;

    filter_iterator() = default;
    filter_iterator(const Predicate& filter, Iterator it, Iterator end)
        : filter_(filter), it_(std::move(it)), end_(std::move(end)) {
        while (it_ != end_ && !filter_(*it_)) {
            ++it_;
        }
    }

    // Enable conversion from const to non-const iterator and vice versa.
    template <class OtherIterator>
    explicit filter_iterator(
            const filter_iterator<OtherIterator, Predicate>& f,
            typename std::enable_if<
                    std::is_convertible<OtherIterator,
                                        Iterator>::value>::type* = nullptr)
        : filter_(f.filter_), it_(f.it_), end_(f.end_) {}

    reference operator*() const { return *it_; }
    pointer operator->() { return std::addressof(*it_); }

    filter_iterator& operator++() {
        do {
            ++it_;
        } while (it_ != end_ && !filter_(*it_));
        return *this;
    }

    filter_iterator operator++(int) {
        filter_iterator copy = *this;
        ++it_;
        return copy;
    }

    inline friend bool operator==(const filter_iterator& left,
                                  const filter_iterator& right) {
        return left.it_ == right.it_;
    }

    inline friend bool operator!=(const filter_iterator& left,
                                  const filter_iterator& right) {
        return left.it_ != right.it_;
    }

private:
    const Predicate& filter_;
    Iterator it_;
    const Iterator end_;
};

template <class Iterator, class Predicate>
struct filter_view {
public:
    filter_view(Predicate filter, Iterator beg, Iterator end)
        : filter_(std::move(filter)),
          beg_(filter_, beg, end),
          end_(filter_, end, end) {}

    filter_iterator<Iterator, Predicate> begin() const { return beg_; }
    filter_iterator<Iterator, Predicate> end() const { return end_; }

private:
    const Predicate filter_;
    const filter_iterator<Iterator, Predicate> beg_;
    const filter_iterator<Iterator, Predicate> end_;
};

// Folds `value` into `seed`, following the classic boost::hash_combine spread.
inline std::size_t HashCombine(std::size_t seed, std::size_t value) {
    return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

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

// Upstream COLMAP dbb41680 parity: hash specializations for the typed
// sensor/data identifiers (required by std::unordered_map<sensor_t, ...>
// consumers such as the W4 global positioning stack).
template <>
struct hash<colmap::sensor_t> {
    std::size_t operator()(const colmap::sensor_t& s) const noexcept {
        return hash<std::pair<uint32_t, uint32_t>>()(
                std::make_pair(static_cast<uint32_t>(s.type), s.id));
    }
};

template <>
struct hash<colmap::data_t> {
    std::size_t operator()(const colmap::data_t& d) const noexcept {
        const size_t h1 = hash<colmap::sensor_t>()(d.sensor_id);
        const size_t h2 = std::hash<uint64_t>()(d.id);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

}  // namespace std
